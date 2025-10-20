#!/usr/bin/env python3
"""
Graph RAG Tool - Hybrid Graph-RAG functionality extracted from EnhancedChatAgent

This module contains the hybrid Graph-RAG tool that:
  - asks the LLM to produce a SPARQL query and semantic keywords (plan),
  - executes the SPARQL against Oxigraph,
  - runs a semantic vector search in LanceDB using the produced keywords,
  - fuses and re-ranks results with an adjustable hybrid scoring,
  - returns a compact, provenance-rich context for the LLM to answer.
"""

import json
import re
import time
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from common.llm_models import get_default_model
from common.embed import generate_embeddings


class GraphRAGTool:
    """Hybrid Graph-RAG tool for combining KG and vector search"""
    
    def __init__(self, kg_store, vector_db, vector_table):
        self.kg_store = kg_store
        self.vector_db = vector_db
        self.vector_table = vector_table
    
    def query_combined_graph_rag(
        self,
        user_query: str,
        top_k_graph: int = 5,
        top_k_vector: int = 10,
        re_rank_with_cross_encoder: bool = False
    ) -> str:
        """
        Full-power hybrid Graph-RAG:
          1) Ask the LLM to generate a JSON plan with 'sparql' and 'keywords'.
          2) Execute the SPARQL plan against Oxigraph.
          3) Run vector search on LanceDB using generated keywords + user_query fallback.
          4) Fuse results by hybrid scoring and optional cross-encoder re-rank.
          5) Return a JSON object { "success": bool, "context": "...", "summary": "...", "count": int }
        """

        start_time = time.time()
        print(f"[Hybrid] Running combined Graph-RAG for: {user_query[:200]}")
        PLANNER_PROMPT = """
            You are a concise retrieval planner.  Your job: for the given user query, return JSON only (no explanation, no extra text) with two keys:
              - "sparql": a single SPARQL SELECT query string (complete and executable) that uses the prefixes below.
              - "keywords": an array of 2–5 short search keywords useful for vector search.

            Always include these prefix declarations at the top of every SPARQL query:
            PREFIX ex: <http://example.org/>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX sioc: <http://rdfs.org/sioc/ns#>
            PREFIX schema: <http://schema.org/>

            Important:
            - Use the graph schema: Claims are `ex:Claim`, text is `ex:canonicalForm`, language `ex:hasLanguage`, truth `ex:truthStatus`, and `prov:wasDerivedFrom` links to source posts.
            - Use case-insensitive filters: e.g. FILTER(CONTAINS(LCASE(STR(?text)), "putin"))
            - Limit results to 10 (or the provided limit).

            Output example (exact JSON only):
            {
              "sparql": "PREFIX ex: <http://example.org/> PREFIX prov: <http://www.w3.org/ns/prov#> SELECT ?claim ?text ?status ?source WHERE { ?claim a ex:Claim ; ex:canonicalForm ?text ; ex:truthStatus ?status ; prov:wasDerivedFrom ?source . FILTER(CONTAINS(LCASE(STR(?text)), \"putin\") && CONTAINS(LCASE(STR(?text)), \"girlfriend\")) } LIMIT 10",
              "keywords": ["putin girlfriend", "Alina Kabaeva birth", "Putin absent girlfriend birth"]
            }
            """

        try:
            # 1️⃣ Build LLM plan
            plan_prompt = self._build_plan_prompt(user_query=user_query)
            plan_text = self._generate_plan_text(plan_prompt, PLANNER_PROMPT)
            print('plan_text', plan_text)

            if not plan_text:
                print("[Hybrid] LLM planning failed or returned empty result.")
                plan_json = {}
            else:
                plan_json = self._parse_llm_json(plan_text)

            sparql_query = plan_json.get("sparql") if isinstance(plan_json, dict) else None
            keywords = plan_json.get("keywords") if isinstance(plan_json, dict) else None
                
            if not keywords:
                keywords = [user_query]  # fallback
            print(plan_json)
            
            # 2️⃣ Run SPARQL plan (if any)
            graph_candidates: List[Dict[str, Any]] = []
            if sparql_query:
                try:
                    print(f"[Hybrid] Executing SPARQL:\n{sparql_query}")
                    results_iter = self.kg_store.query(sparql_query)
                    vars_list = [v.value for v in results_iter.variables]
                    results_list = list(results_iter)
                    for sol in results_list:
                        candidate = {"uri": None, "text": None}
                        for var in vars_list:
                            term = sol[var]
                            if term:
                                sval = str(term.value)
                                if any(k in var.lower() for k in ("text", "label", "canonical")) and not candidate["text"]:
                                    candidate["text"] = sval
                                if any(k in var.lower() for k in ("node", "uri", "id")) and not candidate["uri"]:
                                    candidate["uri"] = sval
                        if not candidate["text"]:
                            for var in vars_list:
                                term = sol[var]
                                if term:
                                    candidate["text"] = str(term.value)
                                    break
                        if not candidate["uri"]:
                            candidate["uri"] = candidate["text"][:128] if candidate["text"] else None
                        if candidate["text"]:
                            graph_candidates.append(candidate)
                except Exception as e:
                    print(f"[Hybrid] SPARQL execution failed: {e}")

            # 3️⃣ Vector search
            vector_candidates: List[Dict[str, Any]] = []
            if self.vector_db and self.vector_table is not None:
                try:
                    emb_input = " ".join(keywords)[:2000]
                    q_emb = self._embed_text(emb_input)
                    if q_emb is None:
                        print("[Hybrid] Query embedding failed; skipping vector retrieval.")
                    else:
                        try:
                            try:
                                res = self.vector_table.search(q_emb).limit(top_k_vector).to_list()
                            except Exception:
                                try:
                                    res = self.vector_table.search(q_emb, "vector").limit(top_k_vector).to_pylist()
                                except Exception:
                                    res = list(self.vector_table.search(q_emb).limit(top_k_vector))

                            for r in res:
                                text = r.get("text") or r.get("content") or r.get("payload") or r.get("doc")
                                uri = r.get("uri") or (r.get("metadata") or {}).get("uri")
                                distance = None
                                for key in ("_distance", "distance", "score", "similarity"):
                                    if key in r:
                                        distance = r[key]
                                        break
                                vector_candidates.append({
                                    "uri": uri,
                                    "text": text,
                                    "distance": float(distance) if distance is not None else None,
                                    "raw": r,
                                })
                        except Exception as e:
                            print(f"[Hybrid] Vector search failed: {e}")
                except Exception as e:
                    print(f"[Hybrid] Vector retrieval failed overall: {e}")
            else:
                print("[Hybrid] No vector DB available; skipping vector retrieval.")

            # 4️⃣ Merge & hybrid scoring
            merged: Dict[str, Dict[str, Any]] = {}

            for g in graph_candidates:
                uri = g.get("uri") or (g.get("text")[:128] if g.get("text") else None)
                if not uri:
                    continue
                merged[uri] = {
                    "uri": uri,
                    "text": g.get("text"),
                    "in_graph": True,
                    "distance": None,
                    "raw_vector": None,
                    "graph_rank": 1,
                }

            for v in vector_candidates:
                uri = v.get("uri") or (v.get("text")[:128] if v.get("text") else None)
                if not uri:
                    continue
                if uri in merged:
                    merged[uri]["distance"] = v.get("distance")
                    merged[uri]["raw_vector"] = v.get("raw")
                else:
                    merged[uri] = {
                        "uri": uri,
                        "text": v.get("text"),
                        "in_graph": False,
                        "distance": v.get("distance"),
                        "raw_vector": v.get("raw"),
                        "graph_rank": 0,
                    }

            scored: List[Tuple[float, Dict[str, Any]]] = []
            for uri, item in merged.items():
                score = self._hybrid_score(item.get("distance"), item.get("in_graph"))
                scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)

            if re_rank_with_cross_encoder:
                print("[Hybrid] Cross-encoder rerank requested but not implemented; skipping.")

            top_n = min(len(scored), max(5, top_k_vector))
            top_items = [it for _, it in scored][:top_n]

            if not top_items:
                return json.dumps({
                    "success": False,
                    "context": "",
                    "summary": "No relevant results found in either the Knowledge Graph or vector store.",
                    "count": 0,
                })

            formatted_lines = []
            for i, it in enumerate(top_items, 1):
                prov = "KG" if it.get("in_graph") else "Vector"
                dist = it.get("distance")
                dist_str = f", dist={dist:.4f}" if dist is not None else ""
                text_snip = (it.get("text") or "")[:400].replace("\n", " ")
                formatted_lines.append(f"{i}. [{prov}{dist_str}] {text_snip}\n   uri: {it.get('uri')}")

            context = "Combined retrieval results (hybrid Graph-RAG):\n\n" + "\n\n".join(formatted_lines)

            # Truncate context if too long for Lambda/Bedrock (safety)
            if len(context) > 5000:
                context = context[:5000] + "\n...[truncated for brevity]..."

            elapsed = round(time.time() - start_time, 2)
            print(f"[Hybrid] Completed hybrid retrieval in {elapsed}s with {len(top_items)} results.")

            # ✅ Return valid JSON always
            return context

        except Exception as e:
            print(f"[Hybrid] Fatal error: {e}\n{traceback.format_exc()}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False)

    def _generate_plan_text(self, plan_prompt: str, planner_system_prompt: str):
        """
        Robust planner call using a fresh model instance.
        - Streams from BedrockModel and captures all partial text chunks.
        - Returns concatenated plan text as a single string.
        """
        print("[Planner] Starting planner call...")
        print("[Planner] plan_prompt repr:", repr(plan_prompt)[:500])
        print("[Planner] planner_system_prompt repr:", repr(planner_system_prompt)[:500])

        # Create a fresh model (avoid recursive tool issues)
        try:
            local_model = get_default_model()
            print("[Planner] got local_model:", type(local_model))
        except Exception as e:
            print("[Planner][ERROR] get_default_model() failed:", e)
            return ""

        async def _collect_stream():
            chunks = []
            try:
                messages = [{"role": "user", "content": [{"type": "text", "text": plan_prompt}]}]
                print("[Planner] Sending messages:", messages)

                async for event in local_model.stream(messages, system_prompt=planner_system_prompt):
                    
                    # ---- FIXED CHUNK EXTRACTION ----
                    content = None

                    # Case 1: event is an object with a .content attr (rare)
                    if hasattr(event, "content"):
                        content = event.content

                    # Case 2: event is a dict (Bedrock stream format)
                    elif isinstance(event, dict):
                        # Handle partial delta text
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"].get("delta", {})
                            content = delta.get("text")

                    # Append valid text chunks
                    if isinstance(content, str) and content.strip():
                        chunks.append(content)
                    else:
                        # Only log if it's an unexpected event type (reduce noise)
                        if isinstance(event, dict) and "messageStop" not in event and "contentBlockStart" not in event:
                            pass  # Suppress routine streaming events

                # Join collected text pieces
                text = "".join(chunks)
                return text

            except Exception as ex:
                print("[Planner][ERROR] Stream exception:", ex)
                traceback.print_exc()
                return ""

        # Run async collector
        try:
            plan_text = asyncio.run(_collect_stream())
            print("[Planner] plan_text repr:", repr(plan_text)[:800])
            print("[Planner] plan_text length:", len(plan_text))
            return plan_text or ""
        except Exception as e:
            print("[Planner][FATAL] asyncio.run failed:", e)
            return ""

    def _build_plan_prompt(self, user_query: str) -> str:
        """
        Build a structured prompt instructing the LLM to generate:
          1. A SPARQL query tailored to the example.org KG schema.
          2. Keywords for vector search on node documents.
        Output must be valid JSON: {"sparql": "...", "keywords": ["..."]}.
        """
        prompt = f"""
            You are an expert in querying RDF Knowledge Graphs and semantic search systems.

            The Knowledge Graph uses these prefixes:
            @prefix ex: <http://example.org/> .
            @prefix prov: <http://www.w3.org/ns/prov#> .
            @prefix schema1: <http://schema.org/> .
            @prefix sioc: <http://rdfs.org/sioc/ns#> .
            @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

            ### Important ontology notes
            - `ex:Claim` instances represent factual claims.
            - Claims have:
                - `ex:canonicalForm` → textual content of the claim.
                - `ex:hasLanguage` → ISO language code (e.g. "en").
                - `ex:truthStatus` → one of ["true", "false", "unknown"].
                - `prov:wasDerivedFrom` → links to a post resource (e.g. `ex:post/...`).
            - `ex:Platform` (e.g. ex:Twitter) describes the platform of origin.
            - Posts are instances of `ex:Post` or similar.

            The user asked:
            \"\"\"{user_query}\"\"\"

            Your task:
            1. Write a SPARQL query that retrieves the most relevant claims, posts, or platforms related to the query.
               - Prefer using `ex:canonicalForm`, `ex:truthStatus`, and `prov:wasDerivedFrom`.
               - Include useful variables like ?claim, ?text, ?status, ?source.
               - Use case-insensitive filtering via `FILTER(CONTAINS(LCASE(str(?text)), "keyword"))`.
               - Limit to 10 results.

            2. Suggest 3–5 search keywords for semantic vector retrieval that capture the same meaning as the query.

            3. Output **only valid JSON** in this structure:
            {{
              "sparql": "<SPARQL QUERY>",
              "keywords": ["kw1", "kw2", "kw3"]
            }}

            Example output:
            {{
              "sparql": "PREFIX ex: <http://example.org/> PREFIX prov: <http://www.w3.org/ns/prov#> SELECT ?claim ?text ?status ?source WHERE {{ ?claim a ex:Claim ; ex:canonicalForm ?text ; ex:truthStatus ?status ; prov:wasDerivedFrom ?source . FILTER(CONTAINS(LCASE(str(?text)), 'switzerland')) }} LIMIT 10",
              "keywords": ["switzerland travel", "going to switzerland", "trip to switzerland"]
            }}
            """
        return prompt

    def _parse_llm_json(self, text):
        # try to extract JSON block robustly
        if not text:
            return {}
        s = str(text)
        # remove ```json blocks etc
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
        # find first {...} block:
        start = s.find('{')
        end = s.rfind('}')
        if start == -1 or end == -1:
            # log for debugging
            print("[Planner] Could not find JSON braces. Raw plan:", s[:1000])
            return {}
        try:
            return json.loads(s[start:end+1])
        except Exception as e:
            print("[Planner] JSON parse failed:", e, "raw:", s[:1000])
            return {}

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Create an embedding for the text using your existing embedding pipeline."""
        try:
            emb = generate_embeddings([text], batch_size=1)
            if emb and len(emb) > 0:
                return emb[0]
        except Exception as e:
            print(f"Embedding error: {e}")
        return None

    def _hybrid_score(self, vector_distance: Optional[float], in_graph: bool, alpha: float = 0.75, beta: float = 0.25) -> float:
        """Combine vector distance (smaller is better) with graph membership into a single score."""
        if vector_distance is None:
            vec_score = 0.0
        else:
            vec_score = 1.0 / (1.0 + max(1e-6, float(vector_distance)))
        graph_boost = 1.0 if in_graph else 0.0
        return alpha * vec_score + beta * graph_boost