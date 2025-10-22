# ============================================================
# File: lambda_functions/common/graph_summary.py
# Purpose: Summarize a Knowledge Graph (TTL + LanceDB vectors)
# Enhanced: verification inference, LanceDB autodetect, journalist-friendly output
# Author: adapted for Saket Kunwar
# ============================================================
import sys
import os
import json
import logging
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

from strands import Agent
import networkx as nx
from pyoxigraph import Store  # expecting pyoxigraph as in your environment

# Import storage + connection helpers (existing in your repo)
from common.storage_utils import load_knowledge_graph, load_vector_store
from common.lancedb_helper import connect_lancedb
from common.aws_clients import bedrock_client
from common.llm_models import get_default_model
from common.embed import generate_embeddings

# ============================================================
# CONFIG & LOGGER
# ============================================================
LANCE_S3_BUCKET = os.getenv("KG_BUCKET", "madsift")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ============================================================
# GRAPH EXTRACTORS (Oxigraph + fallback)
# ============================================================
def extract_triples_oxigraph(ttl_path: str, limit: int = 200) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    """
    Load a TTL file via Oxigraph and extract up to `limit` triples.
    Also collect per-subject metadata (confidence, label, reasoning) inferred from predicates.
    Returns: (triples, node_meta) where node_meta maps subject -> {confidence, label, reasoning, other}
    """
    triples: List[Dict[str, str]] = []
    node_meta: Dict[str, Dict[str, Any]] = {}

    local_path = ttl_path
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
            s = str(binding["s"])
            p = str(binding["p"])
            o = str(binding["o"])
            triples.append({"s": s, "p": p, "o": o})

            # Infer node metadata when predicate names suggest verification info
            key = s
            if key not in node_meta:
                node_meta[key] = {"confidence": None, "label": None, "reasoning": [], "other": {}}

            lp = p.lower()
            # common predicates for confidence/label/reasoning (heuristic)
            if "confidence" in lp or lp.endswith("confidence") or "score" in lp:
                try:
                    # try parse numeric literal
                    val = float(o.strip('"').split("^^")[0])
                except Exception:
                    try:
                        val = float(o)
                    except Exception:
                        val = o
                node_meta[key]["confidence"] = val
            elif "label" in lp or lp.endswith("label"):
                node_meta[key]["label"] = o.strip('"')
            elif "reason" in lp or "reasoning" in lp:
                node_meta[key]["reasoning"].append(o.strip('"'))
            else:
                # stash other preds for potential later use
                node_meta[key]["other"].setdefault(p, []).append(o)

        logger.debug(f"[Oxigraph] Retrieved {len(triples)} triples via SPARQL")

    except Exception as e:
        logger.warning(f"[Oxigraph] SPARQL extract failed, attempting rdflib fallback: {e}")
        traceback.print_exc()
        try:
            from rdflib import Graph
            g = Graph()
            g.parse(local_path, format="turtle")
            for idx, (s, p, o) in enumerate(g):
                s_s = str(s)
                p_s = str(p)
                o_s = str(o)
                triples.append({"s": s_s, "p": p_s, "o": o_s})

                key = s_s
                if key not in node_meta:
                    node_meta[key] = {"confidence": None, "label": None, "reasoning": [], "other": {}}
                lp = p_s.lower()
                if "confidence" in lp or lp.endswith("confidence") or "score" in lp:
                    try:
                        node_meta[key]["confidence"] = float(o_s)
                    except Exception:
                        node_meta[key]["confidence"] = o_s
                elif "label" in lp or lp.endswith("label"):
                    node_meta[key]["label"] = o_s.strip('"')
                elif "reason" in lp or "reasoning" in lp:
                    node_meta[key]["reasoning"].append(o_s.strip('"'))
                else:
                    node_meta[key]["other"].setdefault(p_s, []).append(o_s)

                if len(triples) >= limit:
                    break
            logger.debug(f"[Fallback] Retrieved {len(triples)} triples using rdflib fallback")
        except Exception as e2:
            logger.error(f"[Fallback] rdflib parse also failed: {e2}")
            traceback.print_exc()

    return triples, node_meta

def filter_triples_by_query(triples: List[Dict[str, str]], query_text: str, threshold: float = 0.3) -> List[Dict[str, str]]:
    """
    Filter triples to those relevant to the query using embedding similarity.
    Returns triples where subject, predicate, or object text matches query semantically.
    """
    if not query_text or not triples:
        return triples
    
    try:
        # Generate query embedding
        query_vec = generate_embeddings([query_text], batch_size=1)[0]
        
        # Generate embeddings for triple text representations
        triple_texts = []
        for t in triples:
            # Create readable representation
            s = t['s'].split('/')[-1] if '/' in t['s'] else t['s']
            p = t['p'].split('/')[-1] if '/' in t['p'] else t['p']
            o = t['o'].split('/')[-1] if '/' in t['o'] else t['o']
            triple_texts.append(f"{s} {p} {o}")
        
        triple_vecs = generate_embeddings(triple_texts, batch_size=32)
        
        # Calculate cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        
        filtered = []
        for i, t in enumerate(triples):
            similarity = dot(query_vec, triple_vecs[i]) / (norm(query_vec) * norm(triple_vecs[i]))
            if similarity >= threshold:
                filtered.append(t)
        
        logger.info(f"[QueryFilter] Filtered {len(filtered)}/{len(triples)} triples (threshold={threshold})")
        return filtered
    except Exception as e:
        logger.warning(f"[QueryFilter] Failed, returning all triples: {e}")
        return triples
        
def analyze_confidence_distribution(node_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the distribution of confidence scores across verified claims."""
    confidences = []
    for node, meta in node_meta.items():
        conf = meta.get("confidence")
        if conf is not None:
            try:
                confidences.append(float(conf))
            except:
                pass
    
    if not confidences:
        return {"available": False}
    
    import statistics
    return {
        "available": True,
        "count": len(confidences),
        "mean": round(statistics.mean(confidences), 3),
        "median": round(statistics.median(confidences), 3),
        "min": round(min(confidences), 3),
        "max": round(max(confidences), 3),
        "std_dev": round(statistics.stdev(confidences), 3) if len(confidences) > 1 else 0
    }
    
    
def extract_temporal_patterns(triples: List[Dict[str, str]], node_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Extract temporal information from triples and metadata."""
    timestamps = []
    date_keywords = ["timestamp", "date", "created", "published", "posted"]
    
    for t in triples:
        p = t.get("p", "").lower()
        if any(kw in p for kw in date_keywords):
            try:
                # Try to parse timestamp from object
                from dateutil import parser
                ts = parser.parse(t["o"])
                timestamps.append({"node": t["s"], "timestamp": ts})
            except:
                pass
    
    if not timestamps:
        return {"available": False}
    
    timestamps.sort(key=lambda x: x["timestamp"])
    return {
        "available": True,
        "earliest": timestamps[0]["timestamp"].isoformat(),
        "latest": timestamps[-1]["timestamp"].isoformat(),
        "span_hours": (timestamps[-1]["timestamp"] - timestamps[0]["timestamp"]).total_seconds() / 3600,
        "total_timestamped": len(timestamps)
    }
    
def analyze_sources(triples: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract and categorize sources from URIs."""
    from urllib.parse import urlparse
    from collections import Counter
    
    domains = []
    source_types = {"reddit": [], "twitter": [], "news": [], "other": []}
    
    for t in triples:
        for field in ["s", "o"]:
            uri = t.get(field, "")
            if uri.startswith("http"):
                try:
                    domain = urlparse(uri).netloc
                    domains.append(domain)
                    
                    # Categorize
                    if "reddit" in domain:
                        source_types["reddit"].append(uri)
                    elif "twitter" in domain or "x.com" in domain:
                        source_types["twitter"].append(uri)
                    elif any(news in domain for news in ["news", "times", "post", "bbc", "cnn"]):
                        source_types["news"].append(uri)
                    else:
                        source_types["other"].append(uri)
                except:
                    pass
    
    domain_counts = Counter(domains).most_common(10)
    
    return {
        "total_sources": len(domains),
        "unique_domains": len(set(domains)),
        "top_domains": [{"domain": d, "count": c} for d, c in domain_counts],
        "by_type": {k: len(v) for k, v in source_types.items()},
        "reddit_focus": len(source_types["reddit"]) / len(domains) if domains else 0
    }
    
    
def build_graph(triples: List[Dict[str, str]]) -> nx.DiGraph:
    """Build a directed graph from RDF triples."""
    G = nx.DiGraph()
    for t in triples:
        try:
            s = t["s"]
            o = t["o"]
            p = t.get("p")
            # add nodes explicitly to allow node attributes later
            if s not in G:
                G.add_node(s)
            if o not in G:
                G.add_node(o)
            # add edge with predicate meta
            G.add_edge(s, o, predicate=p)
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
# VERIFICATION INFERENCE
# ============================================================
def analyze_verification_triples(triples: List[Dict[str, str]], node_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Infer verification relationships and produce verification statistics.
    Heuristics:
      - If nodes have label/confidence/reasoning in node_meta, treat them as verification assertions.
      - If edge predicate contains words like 'refute', 'dispute', 'contradict' -> mark as dispute/contradiction.
      - If predicate contains 'support', 'confirm', 'verify', 'factcheck' -> mark as supporting/verified edge.
    Returns:
      {
        "verified_claims": [...],
        "disputed_claims": [...],
        "unverified_claims": [...],
        "verification_ratio": float,
        "verification_examples": [...]
      }
    """
    verified = set()
    disputed = set()
    asserted = set()
    # predicates heuristics
    dispute_keywords = {"refute", "refutes", "contradict", "contradicts", "dispute", "disputes", "challenge"}
    support_keywords = {"support", "supports", "confirm", "confirms", "verified", "verify", "factcheck", "fact-checked"}
    for t in triples:
        p = t.get("p", "").lower()
        s = t.get("s")
        o = t.get("o")
        # check edge-level cues
        if any(k in p for k in dispute_keywords):
            # the 'o' might be a claim node being disputed or the subject is disputing
            disputed.add(s)
            disputed.add(o)
        elif any(k in p for k in support_keywords):
            verified.add(s)
            verified.add(o)
        # collect nodes with explicit meta labels
        # note: node_meta keys are subjects; some verification info could be on object nodes as subjects elsewhere
        # We'll treat any node in node_meta with a label/confidence as asserted
    for node, meta in node_meta.items():
        label = meta.get("label")
        conf = meta.get("confidence")
        reasoning = meta.get("reasoning") or []
        if label:
            low = str(label).lower()
            if any(x in low for x in ["support", "supported", "true", "verified", "confirmed"]):
                verified.add(node)
            elif any(x in low for x in ["refute", "false", "unsupported", "disputed", "contradict"]):
                disputed.add(node)
            else:
                asserted.add(node)
        elif conf is not None:
            # use threshold heuristics: > 0.6 as likely verified
            try:
                f = float(conf)
                if f >= 0.6:
                    verified.add(node)
                elif f <= 0.4:
                    disputed.add(node)
                else:
                    asserted.add(node)
            except Exception:
                asserted.add(node)
    # determine unverified: nodes that appear in graph but not in verified/disputed/asserted
    nodes_in_triples = set([t["s"] for t in triples] + [t["o"] for t in triples])
    unverified = nodes_in_triples - (verified | disputed | asserted)
    # sample examples: include reasoning where available
    examples = []
    for v in list(verified)[:5]:
        r = node_meta.get(v, {}).get("reasoning", [])
        examples.append({"node": v, "status": "verified", "reasoning": r})
    for d in list(disputed)[:5]:
        r = node_meta.get(d, {}).get("reasoning", [])
        examples.append({"node": d, "status": "disputed", "reasoning": r})
    total_claims = len(nodes_in_triples)
    verification_ratio = (len(verified) / total_claims) if total_claims > 0 else 0.0
    return {
        "verified_claims": list(verified),
        "disputed_claims": list(disputed),
        "asserted_claims": list(asserted),
        "unverified_claims": list(unverified),
        "verification_ratio": round(verification_ratio, 3),
        "verification_examples": examples,
        "total_claims": total_claims
    }


# ============================================================
# LANCEDB SEMANTIC CONTEXT (autodetect fix)
# ============================================================
def retrieve_semantic_context(query_text: str, ldb_path: str, k: int = 5) -> list[str]:
    """
    Retrieve semantically similar text snippets using LanceDB.
    
    Self-contained function that handles both S3 and local LanceDB paths.
    
    Args:
        query_text: Query string to search for
        ldb_path: Path to LanceDB (S3 URI like 's3://bucket/path/' or local path)
        k: Number of results to retrieve
        
    Returns:
        List of text snippets
    """
    import lancedb
    from pathlib import Path
    
    context: list[str] = []
    if not ldb_path:
        return context

    try:
        # ====================================================================
        # Step 1: Connect to LanceDB (S3 or local)
        # ====================================================================
        logger.info(f"[LanceDB] Connecting to: {ldb_path}")
        
        # Determine if S3 or local path
        is_s3 = ldb_path.startswith('s3://')
        
        if is_s3:
            # S3 path - connect to database directory
            # Remove trailing .lance if present (table name, not db path)
            db_path = ldb_path.rstrip('/')
            if db_path.endswith('.lance'):
                # Extract database path (parent directory)
                db_path = '/'.join(db_path.split('/')[:-1]) + '/'
            
            db = lancedb.connect(db_path)
            logger.info(f"[LanceDB] Connected to S3 database: {db_path}")
        else:
            # Local path
            local_path = Path(ldb_path)
            if not local_path.exists():
                logger.warning(f"[LanceDB] Local path does not exist: {ldb_path}")
                return []
            
            # If path points to a .lance directory, use parent as database
            if local_path.name.endswith('.lance'):
                db_path = str(local_path.parent)
            else:
                db_path = str(local_path)
            
            db = lancedb.connect(db_path)
            logger.info(f"[LanceDB] Connected to local database: {db_path}")
        
        # ====================================================================
        # Step 2: Open a table
        # ====================================================================
        vector_table = None
        
        try:
            tables = db.table_names()
            logger.info(f"[LanceDB] Available tables: {tables}")
            
            if not tables:
                logger.warning("[LanceDB] No tables found in database.")
                return []
            
            # If original path specified a table name, try to extract it
            table_name = None
            if is_s3 and ldb_path.endswith('.lance/'):
                # Extract table name from S3 path
                table_name = ldb_path.rstrip('/').split('/')[-1].replace('.lance', '')
            elif not is_s3:
                local_path = Path(ldb_path)
                if local_path.name.endswith('.lance'):
                    table_name = local_path.name.replace('.lance', '')
            
            # Try to open the specified table, or use first available
            if table_name and table_name in tables:
                vector_table = db.open_table(table_name)
                logger.info(f"[LanceDB] Opened specified table: {table_name}")
            else:
                # Use first available table
                table_name = tables[0]
                vector_table = db.open_table(table_name)
                logger.info(f"[LanceDB] Opened first available table: {table_name}")
                
        except Exception as e:
            logger.warning(f"[LanceDB] Failed to open table: {e}")
            return []
        
        if not vector_table:
            logger.warning("[LanceDB] No vector table available.")
            return []
        
        # ====================================================================
        # Step 3: Generate query embedding
        # ====================================================================
        try:
            query_vec = generate_embeddings([query_text], batch_size=1)[0]
        except Exception as e:
            logger.warning(f"[LanceDB] Embedding generation failed: {e}")
            return []
        
        # ====================================================================
        # Step 4: Perform vector search
        # ====================================================================
        try:
            results = vector_table.search(query_vec).limit(k).to_pandas()
            logger.info(f"[LanceDB] Search returned {len(results)} results")
        except AttributeError:
            # Try alternate query method
            try:
                results = vector_table.query(query_vec).limit(k).to_pandas()
                logger.info(f"[LanceDB] Query returned {len(results)} results")
            except Exception as e:
                logger.warning(f"[LanceDB] Vector search/query failed: {e}")
                return []
        except Exception as e:
            logger.warning(f"[LanceDB] Vector search failed: {e}")
            traceback.print_exc()
            return []
        
        if results.empty:
            logger.info("[LanceDB] Search returned no results.")
            return []
        
        # ====================================================================
        # Step 5: Extract text columns
        # ====================================================================
        # Look for common text column names
        text_keywords = ["text", "content", "snippet", "body", "doc", "description"]
        text_cols = [c for c in results.columns if any(kw in c.lower() for kw in text_keywords)]
        
        if not text_cols:
            # Fallback to first string-like column
            str_cols = [c for c in results.columns if results[c].dtype == object]
            if str_cols:
                text_cols = [str_cols[0]]
                logger.info(f"[LanceDB] Using fallback text column: {str_cols[0]}")
        
        if not text_cols:
            logger.warning("[LanceDB] No textual columns found in results.")
            logger.info(f"[LanceDB] Available columns: {results.columns.tolist()}")
            return []
        
        logger.info(f"[LanceDB] Using text columns: {text_cols}")
        
        # ====================================================================
        # Step 6: Gather snippets
        # ====================================================================
        for _, row in results.iterrows():
            snippet_parts = []
            for c in text_cols:
                v = row.get(c)
                if isinstance(v, str) and v.strip():
                    snippet_parts.append(v.strip())
            if snippet_parts:
                context.append(" ".join(snippet_parts))
        
        logger.info(f"[LanceDB] Retrieved {len(context)} context snippets.")
        return context[:k]
    
    except Exception as e:
        logger.warning(f"[LanceDB] Context retrieval failed: {e}")
        traceback.print_exc()
        return []


# ============================================================
# CLAUDE / STRANDS SUMMARIZATION (Bedrock)
# ============================================================
def claude_summarize(prompt: str) -> str:
    """
    Summarize text using a simple Strands Agent with the default model.
    """
    try:
        model = get_default_model()   # automatically picks Claude or the configured Bedrock model
        agent = Agent(model=model)
        response = agent(prompt)
        summary_text = (response.text if hasattr(response, "text") else str(response))
        logger.debug(f"[Claude-Agent] Summary generated ({len(summary_text)} chars)")
        return summary_text
    except Exception as e:
        logger.exception(f"[Claude-Agent] Summarization failed: {e}")
        return f"[AUTO-SUMMARY FAILSAFE]\n{prompt[:1000]}"


# ============================================================
# TEXT HELPERS
# ============================================================
def format_list(pairs, limit=5):
    # convert URIs to friendly names where possible
    def short(n):
        if isinstance(n, tuple):
            n = n[0]
        return n.split('/')[-1] if "/" in n else n
    return "\n".join([f"- {short(n)} ({round(v,3)})" for n, v in pairs[:limit]])


def join_snippets(snips, limit=5):
    return "\n".join([f'"{s.strip()}"' for s in snips[:limit]])


def translate_centrality_list(pairs):
    """Translate centrality pairs into journalist-friendly descriptions."""
    descs = []
    for n, v in pairs:
        name = n.split('/')[-1] if "/" in n else n
        descs.append(f"{name} — mentioned frequently (score {round(v,3)})")
    return descs

def extract_query_subgraph(G: nx.DiGraph, query_text: str, node_meta: Dict, hop_distance: int = 2) -> nx.DiGraph:
    """Extract subgraph centered on nodes relevant to query."""
    if not query_text:
        return G
    
    try:
        # Find nodes matching query
        query_lower = query_text.lower()
        relevant_nodes = []
        
        for node in G.nodes():
            node_str = str(node).lower()
            label = node_meta.get(node, {}).get("label", "").lower()
            
            if query_lower in node_str or query_lower in label:
                relevant_nodes.append(node)
        
        if not relevant_nodes:
            logger.warning(f"[Subgraph] No nodes match query '{query_text}', returning full graph")
            return G
        
        # Get nodes within hop_distance
        subgraph_nodes = set(relevant_nodes)
        for node in relevant_nodes:
            # Add neighbors within hop distance
            for target in nx.single_source_shortest_path_length(G, node, cutoff=hop_distance):
                subgraph_nodes.add(target)
        
        subgraph = G.subgraph(subgraph_nodes).copy()
        logger.info(f"[Subgraph] Extracted subgraph: {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
        return subgraph
        
    except Exception as e:
        logger.warning(f"[Subgraph] Extraction failed: {e}, returning full graph")
        return G
        
# ============================================================
# MAIN SUMMARIZATION PIPELINE
# ============================================================
def summarize_graph(
    ttl_path: str,
    query_text: Optional[str] = None,
    ldb_path: Optional[str] = None,
    limit: int = 200,
    top_k: int = 5,
    filter_by_query: bool = True  # NEW parameter
) -> dict:
    """
    Summarize a knowledge graph with optional query-focused analysis.
    """
    logger.info(f"[Summarize] Starting summarization for {ttl_path}")
    if query_text:
        logger.info(f"[Summarize] Focus query: '{query_text}'")

    # --- Step 1: Extract triples + node meta ---
    triples, node_meta = extract_triples_oxigraph(ttl_path, limit)
    if not triples:
        return {"error": "No triples extracted from graph."}

    # --- NEW: Filter triples by query if provided ---
    original_count = len(triples)
    if query_text and filter_by_query:
        triples = filter_triples_by_query(triples, query_text, threshold=0.25)
        logger.info(f"[Summarize] Query filtering: {len(triples)}/{original_count} triples retained")

    # --- Step 2: Build graph ---
    G = build_graph(triples)
    
    # --- NEW: Extract query-focused subgraph ---
    if query_text:
        G = extract_query_subgraph(G, query_text, node_meta, hop_distance=2)
    
    top_deg, top_btw = top_centralities(G, k=top_k)
    top_deg_desc = translate_centrality_list(top_deg)
    top_btw_desc = [f"{n.split('/')[-1] if '/' in n else n} — acts as a bridge connecting different conversations (score {round(v,3)})" for n, v in top_btw]

    # --- Step 3: Semantic context ---
    context = []
    if ldb_path:
        # Use actual query, not fallback
        context = retrieve_semantic_context(query_text if query_text else "knowledge graph summary", ldb_path, k=top_k)
    
    # --- Step 4: Verification analysis ---
    verification = analyze_verification_triples(triples, node_meta)
    
    # --- NEW: Additional analyses ---
    confidence_dist = analyze_confidence_distribution(node_meta)
    temporal_info = extract_temporal_patterns(triples, node_meta)
    source_analysis = analyze_sources(triples)
    
    # Build verification snippet
    verified_count = len(verification.get("verified_claims", []))
    disputed_count = len(verification.get("disputed_claims", []))
    unverified_count = len(verification.get("unverified_claims", []))
    total_claims = verification.get("total_claims", 0)

    verification_snippet = (
        f"Out of approximately {total_claims} claim-related nodes, "
        f"{verified_count} appear verified, {disputed_count} disputed, "
        f"and {unverified_count} remain unverified. "
        f"Verification ratio: {verification.get('verification_ratio')}"
    )
    
    # Add confidence distribution if available
    if confidence_dist.get("available"):
        verification_snippet += f"\nConfidence scores: mean={confidence_dist['mean']}, median={confidence_dist['median']}, range=[{confidence_dist['min']}, {confidence_dist['max']}]"

    # --- Step 5: Build enhanced prompt ---
    prompt_intro = f"""
You are an expert assistant writing a concise briefing for a journalist. Use plain language; avoid technical graph jargon.
"""
    
    if query_text:
        prompt_intro += f"""
FOCUS QUERY: "{query_text}"
The journalist specifically wants to understand what this knowledge graph reveals about: {query_text}

Structure your response around this query. Highlight:
- How "{query_text}" appears in the graph and related discussions
- Key accounts/sources discussing this specific topic
- Verification status of claims related to "{query_text}"
- How this topic connects to broader conversations
"""
    else:
        prompt_intro += """
Focus on what this knowledge graph reveals about information spread and verification.
"""
    
    prompt = prompt_intro + f"""

Graph snapshot ({"query-filtered" if query_text and filter_by_query else "full"}):
- Nodes: {len(G.nodes)}, Edges: {len(G.edges)}
{f"- Original graph had {original_count} triples before query filtering" if query_text and filter_by_query and original_count != len(triples) else ""}

Source analysis:
- Total sources: {source_analysis['total_sources']}, Unique domains: {source_analysis['unique_domains']}
- Top domains: {', '.join([f"{d['domain']} ({d['count']})" for d in source_analysis['top_domains'][:5]])}
- Reddit focus: {round(source_analysis.get('reddit_focus', 0) * 100, 1)}%

Top accounts/sources (by mentions):
{chr(10).join(['- ' + s for s in top_deg_desc])}

Bridge accounts (connect different conversations):
{chr(10).join(['- ' + s for s in top_btw_desc])}

Verification summary:
{verification_snippet}

{"Temporal pattern: Claims span " + str(round(temporal_info['span_hours'], 1)) + " hours from " + temporal_info['earliest'] + " to " + temporal_info['latest'] if temporal_info.get('available') else "No temporal information available"}

Contextual evidence snippets (up to {top_k}):
{join_snippets(context)}

Verification reasoning examples:
"""
    
    # Add reasoning examples
    reasoning_examples = []
    for ex in verification.get("verification_examples", [])[:5]:
        node_short = ex["node"].split('/')[-1] if "/" in ex["node"] else ex["node"]
        reason_text = " | ".join(ex.get("reasoning", [])) if ex.get("reasoning") else "No reasoning available"
        reasoning_examples.append(f"- {node_short}: {reason_text}")
    
    prompt += "\n".join(reasoning_examples) if reasoning_examples else "- None found in metadata"
    
    prompt += """

Write the briefing now. Use simple, direct sentences. Structure as:
1. TOP-LINE SUMMARY (3-5 sentences)
2. KEY FINDINGS (bullet points)
3. VERIFICATION STATUS (what's confirmed, what's disputed)
4. SOURCES & PROPAGATION (who's amplifying this)
5. WHAT TO WATCH (next steps for reporting)
"""

    # --- Step 6: Generate summary ---
    summary = claude_summarize(prompt)

    # --- Step 7: Return enhanced structured output ---
    result = {
        "summary": summary,
        "query": query_text,
        "query_filtered": filter_by_query and query_text is not None,
        "graph_stats": {
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "original_triples": original_count,
            "filtered_triples": len(triples) if query_text and filter_by_query else original_count
        },
        "top_degree": top_deg,
        "top_betweenness": top_btw,
        "top_degree_descriptions": top_deg_desc,
        "top_betweenness_descriptions": top_btw_desc,
        "verification_stats": verification,
        "confidence_distribution": confidence_dist,
        "temporal_analysis": temporal_info,
        "source_analysis": source_analysis,
        "context_samples": context,
        "timestamp": datetime.now().isoformat()
    }

    logger.info(f"[Summarize] Completed: {result['graph_stats']}")
    return result


# ============================================================
# LOCAL TEST HARNESS (CLI)
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
