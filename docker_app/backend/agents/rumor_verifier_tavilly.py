#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rumor_verifier_tavily.py

Full, self-contained Rumor verifier:
- Loads claims from TTL (auto-detects text predicate)
- Groups them into clusters (default 64)
- For each cluster:
    - Summarizes cluster topic (Pydantic-validated)
    - Builds compact keyword query (<= 400 chars) for Tavily
    - Retrieves evidence snippets from Tavily
    - Runs Bedrock/Agent structured reasoning on the cluster (one call)
    - Robustly recovers/heuristic-fallbacks if structured parsing fails
- Writes results back to TTL (_verified.ttl)
"""

import os
import re
import sys
import json
import time
import logging
import datetime
import requests
from typing import List, Tuple, Dict, Any
from collections import Counter
from pydantic import BaseModel, Field, ValidationError
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD


from strands import Agent
from strands.models import BedrockModel
from common.llm_models import get_default_model
from common.aws_clients import get_secret


try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_DATA_PATH = os.getenv("NLTK_DATA", os.path.join(os.path.dirname(__file__), "data/nltk_data"))
    nltk.data.path.append(NLTK_DATA_PATH)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set([
        "the","and","that","this","with","from","have","were","their","which","when",
        "what","about","also","such","these","those","after","before","because","over",
    ])

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rumor_verifier_tavily_complete")

# Config: Tavily endpoint + API key (env var)
region = os.getenv("AWS_REGION", "us-east-1")
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_API_KEY = get_secret('api_secret', region, 'tavily')

# ---------------------------
# Pydantic models
# ---------------------------
class RumorResult(BaseModel):
    id: str
    label: str
    score: float
    reasoning: str

class RumorBatchResult(BaseModel):
    results: List[RumorResult]

class TopicSummary(BaseModel):
    summary: str = Field(..., description="A concise cluster topic summary (short)")

# ---------------------------
# Helpers
# ---------------------------
def extract_keywords(claim_texts: List[str], top_k: int = 10) -> str:
    """Return top_k keywords from claim_texts joined as a single string."""
    text = " ".join(claim_texts).lower()
    # keep words of length >=4
    tokens = re.findall(r"[a-z]{4,}", text)
    filtered = [t for t in tokens if t not in STOPWORDS]
    counts = Counter(filtered)
    kws = [w for w, _ in counts.most_common(top_k)]
    return " ".join(kws)

def tavily_post(payload: dict, timeout: int = 30) -> Dict[str, Any]:
    """Post to Tavily with basic error handling and return parsed JSON or {}."""
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY environment variable not set")
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(TAVILY_ENDPOINT, headers=headers, json=payload, timeout=timeout)
        try:
            return r.json()
        except Exception:
            logger.error("Tavily returned non-JSON response (status=%d): %s", r.status_code, r.text[:400])
            return {}
    except requests.RequestException as e:
        logger.error("Tavily request failed: %s", e)
        return {}

def extract_agent_text(agent_result) -> str:
    """
    Extract string text from an AgentResult-like object:
    - if has .message and message is dict with content list [{'text':...}], return that text
    - else fall back to str(agent_result)
    """
    try:
        msg = getattr(agent_result, "message", None)
        if isinstance(msg, dict):
            cont = msg.get("content")
            if isinstance(cont, list) and len(cont) > 0:
                first = cont[0]
                if isinstance(first, dict) and "text" in first and isinstance(first["text"], str):
                    return first["text"].strip()
        # Try str(agent_result)
        return str(agent_result).strip()
    except Exception:
        return str(agent_result)

def safe_truncate(s: str, limit: int) -> str:
    if not s:
        return s
    return s if len(s) <= limit else s[:limit-1] + "…"


def _clean_claim_text(text: str) -> str:
    """
    Normalize claim text to avoid JSON parse issues downstream.
    Removes nested quotes, smart quotes, and weird punctuation that break LLM output.
    """
    # Normalize unicode quotes and dashes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stray quotation marks or partial JSON-like patterns
    text = re.sub(r'["“”]+', '', text)  # strip all remaining quotes

    # Optionally sentence-normalize (nltk)
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
        text = " ".join(sents)
    except Exception:
        pass

    # Remove trailing punctuation-only tokens
    text = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text)
    return text

# ---------------------------
# Main verifier class
# ---------------------------
class RumorVerifierBatchLLM:
    def __init__(self, ttl_path: str):
        self.ttl_path = ttl_path
        self.graph = Graph()
        if not os.path.exists(ttl_path):
            raise FileNotFoundError(f"TTL file not found: {ttl_path}")
        self.graph.parse(ttl_path, format="turtle")
        self.agent = Agent(model=get_default_model()) if 'Agent' in globals() and get_default_model is not None else None
        self.PRED_NS = Namespace("http://example.org/pred/")
        logger.info("Initialized verifier for TTL: %s", ttl_path)
        self.TAVILY_API_KEY = TAVILY_API_KEY


    # ---------- Claim loader with auto-detection ----------
    def _load_claims(self) -> List[Tuple[str, str]]:
        """Load only literal textual claims (not URIs or usernames)."""
        g = self.graph
        textlike_preds = {}
        for s, p, o in g:
            if isinstance(o, Literal):
                val = str(o)
                if len(val.split()) >= 5 and re.search(r"[A-Za-z]", val):
                    textlike_preds.setdefault(p, []).append((s, val))

        if not textlike_preds:
            logger.error("No literal text-like claims found in TTL!")
            return []

        # Pick the predicate with the most literal values
        chosen, claims = max(textlike_preds.items(), key=lambda kv: len(kv[1]))
        logger.info("Chosen predicate for claim text: %s (%d claims)", chosen, len(claims))

        # Return only (URI, text)
        return [(str(s), str(text)) for s, text in claims]



    def _tavily_search(self, cluster_claims: List[Tuple[str, str]]) -> str:
        """Summarize claims & query Tavily using compact, keyword-driven search."""
        try:
            # --- summarize cluster topic ---
            resp = self.agent(
                "Summarize the main concrete topic of these claims in ≤12 words. "
                "Use nouns, skip filler like 'main topic is':\n\n" +
                "\n".join(c[1] for c in cluster_claims)
            )
            msg = resp.message
            if isinstance(msg, dict):
                content = msg.get("content", [])
                text = ""
                if content and isinstance(content, list) and "text" in content[0]:
                    text = content[0]["text"]
                else:
                    text = str(resp)
            else:
                text = str(resp)

            summary = TopicSummary(summary=text.strip()).summary
        except Exception as e:
            logger.warning("Topic summarization failed: %s", e)
            summary = "Nepal current events and tourism"

        # --- extract compact keywords for search ---
        keywords = extract_keywords([c[1] for c in cluster_claims], top_k=8)
        query_base = f"{summary}. {keywords} Nepal"
        safe_query = query_base[:350]
        query = f"Recent credible English news (past 7 days) about: {safe_query}"

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=7)

        payload = {
            "query": query,
            "search_depth": "advanced",          # stronger coverage
            "max_results": 20,
            "structured_response": False,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }

        try:
            r = requests.post(
                TAVILY_ENDPOINT,
                headers={"Authorization": f"Bearer {self.TAVILY_API_KEY}"},
                json=payload,
                timeout=45,
            )
            data = r.json()
            if "results" not in data:
                logger.error("Tavily error: %s", data)
                return ""

            texts = [
                (res.get("content") or res.get("snippet") or "").strip()
                for res in data["results"]
                if (res.get("content") or res.get("snippet"))
            ]
            context = "\n\n".join(texts)
            logger.info(
                "Tavily returned %d snippets (context %.1f KB)",
                len(texts), len(context) / 1024,
            )
            return context
        except Exception as e:
            logger.error("Tavily call failed: %s", e)
            return ""
            
    def _bedrock_reason(self, context: str, cluster_claims: List[Tuple[str, str]]) -> List[Dict]:
        """Use Bedrock reasoning to verify claims robustly and tolerate malformed JSON."""

        # --- normalize input ---
        normalized_claims = []
        for item in cluster_claims:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                normalized_claims.append(item)
            else:
                cid = f"claim_{abs(hash(item)) % (10**8)}"
                normalized_claims.append((cid, str(item)))
        cluster_claims = normalized_claims
        # --- sanitize all claims ---
        cleaned_claims = [(cid, _clean_claim_text(text)) for cid, text in cluster_claims]
        cluster_claims = cleaned_claims

        # --- setup model ---
        model = BedrockModel(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            temperature=0.0,
            streaming=False,
        )
        agent = Agent(model=model)

        # --- build prompt ---
        prompt = f"""
        You are a professional fact-verification model.
        Using the provided CONTEXT, verify each claim and return a JSON object matching this schema:

        {{
          "results": [
            {{
              "id": "claim_uri",
              "label": "SUPPORTED" | "REFUTED" | "UNCERTAIN",
              "score": float,
              "reasoning": string
            }}
          ]
        }}

        CONTEXT:
        {context}

        CLAIMS:
        """ + "\n".join(f"- {cid}: {text}" for cid, text in cluster_claims)

        # --- try structured output first ---
        try:
            result = agent.structured_output(RumorBatchResult, prompt)
            if result and result.results:
                logger.info("Structured output parsed (%d results)", len(result.results))
                return [r.dict() for r in result.results]
        except Exception as e:
            logger.warning("Structured output failed: %s", e)

        # --- fallback JSON parse (robust repair) ---
        try:
            response = agent(prompt)
            text = str(response)

            # isolate JSON section
            start_idx = text.find("{")
            if start_idx == -1:
                start_idx = text.find("[")
            if start_idx > 0:
                text = text[start_idx:]
            text = text.strip()

            # 🧩 clean up malformed JSON
            text = re.sub(r'(?<!\\)"(.*?)"(?=\s*:)', lambda m: '"' + m.group(1).replace('"', '\\"') + '"', text)
            text = text.replace('“', '"').replace('”', '"')
            text = text.replace("True", "true").replace("False", "false")
            text = text.replace("\n", " ")
            text = re.sub(r",\s*}", "}", text)
            text = re.sub(r",\s*]", "]", text)

            # forgiving parse
            try:
                import json5
                data = json5.loads(text)
            except Exception:
                data = json.loads(text)

            if isinstance(data, dict) and "results" in data:
                logger.info("Recovered valid JSON fallback (%d results)", len(data["results"]))
                return data["results"]
            elif isinstance(data, list):
                logger.info("Recovered list JSON fallback (%d results)", len(data))
                return data
        except Exception as e:
            logger.warning("Fallback JSON parse failed: %s", e)

        # --- final fallback ---
        logger.info("Heuristic fallback: structured output + parse failed")
        return [
            {
                "id": cid,
                "label": "UNCERTAIN",
                "score": 0.5,
                "reasoning": "Structured output failed; default fallback.",
            }
            for cid, _ in cluster_claims
        ]

    # ---------- Append results to TTL ----------
    def _append_results_to_ttl(self, results: List[Dict[str, Any]]):
        """
        Append verification results (label, confidence, reasoning) to the TTL graph and write file.
        Uses predicate namespace http://example.org/pred/
        """
        PRED = self.PRED_NS
        g = self.graph
        for item in results:
            cid = URIRef(item.get("id"))
            # create simple triples (label, confidence, reasoning)
            g.set((cid, PRED["label"], Literal(item.get("label", "UNCERTAIN"))))
            g.set((cid, PRED["confidence"], Literal(float(item.get("score", 0.5)), datatype=XSD.float)))
            g.set((cid, PRED["reasoning"], Literal(item.get("reasoning", "")[:1000])))
        out_path = self.ttl_path.replace(".ttl", "_verified.ttl")
        try:
            g.serialize(destination=out_path, format="turtle")
            logger.info("Wrote verified TTL --> %s", out_path)
            return out_path # <-- THE FIX: Return the path of the new file
        except Exception as e:
            logger.error("Failed to write TTL: %s", e)
            return None # Return None on failure
        

    # ---------- Orchestration ----------
    def verify_clusters(self, cluster_size: int = 64) -> List[Dict[str, Any]]:
        # 1) load claims
        claims = self._load_claims()
        logger.info("Sample claim text: %s", claims[0][1][:120] if claims else "(none)")

        if len(claims) == 0:
            logger.warning("No claims found in TTL. Exiting.")
            return []

        # 2) create clusters
        clusters = [claims[i:i + cluster_size] for i in range(0, len(claims), cluster_size)]
        logger.info("Processing %d clusters (cluster_size=%d)", len(clusters), cluster_size)

        all_results: List[Dict[str, Any]] = []
        for idx, cluster in enumerate(clusters, start=1):
            logger.info("Cluster %d/%d: retrieving evidence (Tavily)...", idx, len(clusters))
            context = self._tavily_search(cluster)
            logger.info("Cluster %d/%d: reasoning with Bedrock...", idx, len(clusters))
            cluster_results = self._bedrock_reason(context, cluster)
            all_results.extend(cluster_results)

        # write back to TTL

        verified_ttl_path = self._append_results_to_ttl(all_results)
        return all_results, verified_ttl_path
        
# ---------------------------
# CLI entrypoint
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rumor verifier - Tavily retrieval + Bedrock reasoning")
    parser.add_argument("--ttl_path", required=True, help="Path to TTL file containing claims")
    parser.add_argument("--cluster_size", type=int, default=64, help="Claims per cluster (default 64)")
    args = parser.parse_args()

    verifier = RumorVerifierBatchLLM(args.ttl_path)
    start = time.time()
    results, verified_ttl_path = verifier.verify_clusters(cluster_size=args.cluster_size)
    elapsed = time.time() - start
    logger.info("Done: processed %d claims in %.1f seconds", len(results), elapsed)
    if results:
        print(json.dumps(results[:5], indent=2))
    print("Total results:", len(results), verified_ttl_path)

if __name__ == "__main__":
    main()
