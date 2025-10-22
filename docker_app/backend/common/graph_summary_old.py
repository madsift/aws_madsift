# ============================================================
# File: lambda_functions/common/graph_summary.py
# Purpose: Summarize a Knowledge Graph (TTL + LanceDB vectors)
# Author: Saket Kunwar
# ============================================================
import sys
import os
import json
import logging
import traceback
import boto3
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from strands import Agent
import networkx as nx
from pyoxigraph import Store  # ✅ Using pyoxigraph as requested

# Import storage + connection helpers
# for local testing

from common.storage_utils import load_knowledge_graph, load_vector_store
from common.lancedb_helper import connect_lancedb
from common.aws_clients import  bedrock_client
from common.llm_models import get_default_model
from common.embed import generate_embeddings  

# ============================================================
# CONFIG
# ============================================================

LANCE_S3_BUCKET = os.getenv("KG_BUCKET" ,"madsift")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Change to INFO to see info logs



# ============================================================
# GRAPH EXTRACTORS
# ============================================================

def extract_triples_oxigraph(ttl_path: str, limit: int = 200) -> List[Dict[str, str]]:
    """
    Load a TTL file via Oxigraph and extract up to `limit` triples.
    Falls back to rdflib if Oxigraph query fails.
    """
    triples = []
    try:
        store, local_path = load_knowledge_graph(ttl_path)
        logger.debug(f"[Oxigraph] Extracting triples from {local_path}")

        q = f"""
        SELECT ?s ?p ?o
        WHERE {{
            ?s ?p ?o .
        }} LIMIT {limit}
        """

        results = store.query(q)
        for binding in results:
            triples.append({
                "s": str(binding["s"]),
                "p": str(binding["p"]),
                "o": str(binding["o"])
            })

        logger.debug(f"[Oxigraph] Retrieved {len(triples)} triples via SPARQL")

    except Exception as e:
        logger.warning(f"[Oxigraph] SPARQL extract failed, attempting rdflib fallback: {e}")
        traceback.print_exc()
        try:
            from rdflib import Graph
            g = Graph()
            g.parse(local_path, format="turtle")
            for s, p, o in g:
                triples.append({"s": str(s), "p": str(p), "o": str(o)})
                if len(triples) >= limit:
                    break
            logger.debug(f"[Fallback] Retrieved {len(triples)} triples using rdflib fallback")
        except Exception as e2:
            logger.error(f"[Fallback] rdflib parse also failed: {e2}")
    return triples


def build_graph(triples: List[Dict[str, str]]) -> nx.DiGraph:
    """Build a directed graph from RDF triples."""
    G = nx.DiGraph()
    for t in triples:
        try:
            G.add_edge(t["s"], t["o"], predicate=t["p"])
        except Exception as e:
            logger.warning(f"[GraphBuild] Failed to add edge {t}: {e}")
    logger.debug(f"[GraphBuild] Graph nodes={len(G.nodes)}, edges={len(G.edges)}")
    return G


def top_centralities(G: nx.DiGraph, k: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Compute top degree and betweenness centralities."""
    if len(G) == 0:
        return [], []
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G)
    top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:k]
    top_btw = sorted(btw.items(), key=lambda x: x[1], reverse=True)[:k]
    logger.debug(f"[Centrality] Top degree nodes: {top_deg[:3]}")
    logger.debug(f"[Centrality] Top betweenness nodes: {top_btw[:3]}")
    return top_deg, top_btw


# ============================================================
# LANCEDB SEMANTIC CONTEXT
# ============================================================
def retrieve_semantic_context(query_text: str, ldb_path: str, k: int = 5) -> list[str]:
    """Retrieve semantically similar text snippets using LanceDB 0.10.2 (vector mode)."""
  

    context = []
    try:
        _, vector_table, _ = load_vector_store(ldb_path)
        if not vector_table:
            logger.warning("[LanceDB] No vector table found.")
            return []

        # ✅ Step 1: Embed the query text
        query_vec = generate_embeddings([query_text], batch_size=1)[0]

        # ✅ Step 2: Perform vector search using the correct column
        results = (
            vector_table.search(query_vec)
            .limit(k)
            .to_pandas()
        )

        if "text" not in results.columns:
            logger.warning("[LanceDB] No 'text' column found in results.")
            return []

        context = results["text"].astype(str).tolist()
        logger.info(f"[LanceDB] Retrieved {len(context)} context snippets.")
        return context

    except Exception as e:
        logger.warning(f"[LanceDB] Context retrieval failed: {e}")
        traceback.print_exc()
        return []



# ============================================================
# CLAUDE SUMMARIZATION (BEDROCK)
# ============================================================


def claude_summarize(prompt: str) -> str:
    """
    Summarize text using a simple Strands Agent with the default model.
    """
    try:
        model = get_default_model()   # automatically picks Claude or the configured Bedrock model
        agent = Agent(model=model)
        response = agent(prompt)
        # Agent returns a structured object; extract text content
        summary_text = (
            response.text if hasattr(response, "text") else str(response)
        )
        logger.debug(f"[Claude-Agent] Summary generated ({len(summary_text)} chars)")
        return summary_text
    except Exception as e:
        logger.exception(f"[Claude-Agent] Summarization failed: {e}")
        return f"[AUTO-SUMMARY FAILSAFE]\n{prompt[:1000]}"

# ============================================================
# TEXT HELPERS
# ============================================================

def format_list(pairs, limit=5):
    return "\n".join([f"- {n.split('/')[-1]} ({round(v,3)})" for n, v in pairs[:limit]])


def join_snippets(snips, limit=5):
    return "\n".join([f'"{s.strip()}"' for s in snips[:limit]])


# ============================================================
# MAIN SUMMARIZATION PIPELINE
# ============================================================

def summarize_graph(
    ttl_path: str,
    query_text: Optional[str] = None,
    ldb_path: Optional[str] = None,
    limit: int = 200,
    top_k: int = 5
) -> dict:
    """
    Summarize a knowledge graph using Oxigraph, NetworkX,
    and semantic context from LanceDB.
    """
    logger.info(f"[Summarize] Starting summarization for {ttl_path}")
    if query_text:
        logger.info(f"[Summarize] Focus query: '{query_text}'")

    # --- Step 1: Extract triples via Oxigraph ---
    triples = extract_triples_oxigraph(ttl_path, limit)
    if not triples:
        return {"error": "No triples extracted from graph."}

    # --- Step 2: Build NetworkX graph ---
    G = build_graph(triples)
    top_deg, top_btw = top_centralities(G, k=top_k)

    # --- Step 3: Retrieve semantic context (optional) ---
    context = []
    if ldb_path:
        context = retrieve_semantic_context(query_text or "rumor graph summary", ldb_path, k=top_k)

    # --- Step 4: Build prompt for summarization ---
    prompt = f"""
        You are an expert assistant for rumor verification and analysis.

        Summarize the knowledge graph{(' related to ' + query_text) if query_text else ''}.
        Each node represents a claim, post, or user derived from Reddit discussions.

        Top entities by degree centrality:
        {format_list(top_deg)}

        Key bridge nodes by betweenness:
        {format_list(top_btw)}

        Contextual evidence from semantic retrieval:
        {join_snippets(context)}

        Provide a concise, factual summary describing:
        - Main rumors or topics
        - Influential entities or accounts
        - Relationships or contradictions
        - Any observable propagation patterns
        """

    # --- Step 5: Generate Claude summary ---
    summary = claude_summarize(prompt)

    # --- Step 6: Return structured summary ---
    result = {
        "summary": summary,
        "top_degree": top_deg,
        "top_betweenness": top_btw,
        "context_samples": context,
        "graph_stats": {"nodes": len(G.nodes), "edges": len(G.edges)},
        "query": query_text,
        "timestamp": datetime.now().isoformat()
    }

    logger.info(f"[Summarize] Completed summarization for {ttl_path}: {result['graph_stats']}")
    return result


# ============================================================
# LOCAL TEST HARNESS
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl", required=True, help="Path or S3 URI to TTL file")
    parser.add_argument("--query", required=False, help="Optional query focus")
    parser.add_argument("--ldb", required=False, help="Optional LanceDB path (S3 or local)")
    args = parser.parse_args()

    output = summarize_graph(args.ttl, query_text=args.query, ldb_path=args.ldb)
    print(json.dumps(output, indent=2, ensure_ascii=False))
