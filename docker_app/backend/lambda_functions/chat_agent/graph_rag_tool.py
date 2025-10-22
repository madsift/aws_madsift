#!/usr/bin/env python3
"""
Graph RAG Tool - Hybrid Graph-RAG functionality with Dynamic Few-Shot Prompting

This module contains the hybrid Graph-RAG tool that:
  - Pre-fetches real, relevant examples from the KG.
  - Asks the LLM to produce a SPARQL query and keywords, primed with the dynamic examples.
  - Executes the SPARQL against Oxigraph and captures all variables.
  - Runs a semantic vector search in LanceDB.
  - Fuses and re-ranks results.
  - Returns a compact, provenance-rich context for the LLM to answer.
"""

import json
import re
import time
import traceback
import asyncio
from typing import Dict, Any, Optional, List, Tuple

from common.llm_models import get_default_model
from common.embed import generate_embeddings


class GraphRAGTool:
    """Hybrid Graph-RAG tool for combining KG and vector search"""
    
    def __init__(self, kg_store, vector_db, vector_table):
        self.kg_store = kg_store
        self.vector_db = vector_db
        self.vector_table = vector_table

    def _extract_simple_keywords(self, query: str, num_keywords: int = 3) -> List[str]:
        """A simple keyword extractor to find terms for the pre-fetch query."""
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'on', 'for', 'with', 'about', 'and', 'or', 'list', 'show', 'get', 'find'}
        words = re.findall(r'\b\w{3,}\b', query.lower())
        return [word for word in words if word not in stop_words][:num_keywords]

    def _prefetch_examples(self, keywords: List[str], limit: int = 3) -> str:
        """
        Runs a simple, broad SPARQL query to find a few relevant example triples
        from the Knowledge Graph to be used in the planner prompt.
        """
        if not self.kg_store or not keywords:
            return "No relevant examples found."

        # Create a FILTER clause that checks for any of the keywords
        filter_conditions = [f'CONTAINS(LCASE(STR(?o)), "{kw}")' for kw in keywords]
        sparql_filter = f"FILTER ( {' || '.join(filter_conditions)} )"

        prefetch_query = f"""
            SELECT ?s ?p ?o WHERE {{
              ?s ?p ?o .
              {sparql_filter}
            }} LIMIT {limit}
        """
        
        try:
            results_iter = self.kg_store.query(prefetch_query)
            
            formatted_examples = []
            for solution in results_iter:
                s_term = solution.get('s')
                p_term = solution.get('p')
                o_term = solution.get('o')

                if not all([s_term, p_term, o_term]):
                    continue

                s = s_term.n3()
                p = p_term.n3()
                o = o_term.n3()
                
                # Clean up the n3 representation for better readability in the prompt
                s = s.split('/')[-1].replace('>', '')
                p = p.split('/')[-1].replace('>', '')
                
                formatted_examples.append(f"ex:{s} {p} {o} .")

            if not formatted_examples:
                return "No specific examples found in the KG for these keywords."
                
            # Return a nicely formatted block of text for the prompt
            return "Here are some RELEVANT EXAMPLES from the actual graph:\n" + "\n".join(formatted_examples)

        except Exception as e:
            print(f"[Prefetch] Could not fetch examples: {e}")
            return "Could not retrieve examples from the Knowledge Graph."
    
    def query_combined_graph_rag(
        self,
        user_query: str,
        top_k_graph: int = 5,
        top_k_vector: int = 10,
        re_rank_with_cross_encoder: bool = False
    ) -> str:
        """
        Full-power hybrid Graph-RAG with Dynamic Few-Shot Prompting.
        """
        start_time = time.time()
        print(f"[Hybrid] Running combined Graph-RAG for: {user_query[:200]}")
        
        # The PLANNER_PROMPT now includes a placeholder for our dynamic examples.
        PLANNER_PROMPT = """
            You are a concise retrieval planner. Your job is to return JSON only (no extra text) with "sparql" and "keywords" keys.

            Always include these prefixes in your SPARQL query:
            PREFIX ex: <http://example.org/>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX sioc: <http://rdfs.org/sioc/ns#>
            PREFIX schema1: <http://schema.org/>
            PREFIX ns1: <http://example.org/pred/>

            ### Schema Guide:
            - Claims (`ex:Claim`) have:
                - Text: `ex:canonicalForm`
                - Verification Status: `ns1:label` (e.g., "SUPPORTED")
            - Posts (`sioc:Post`) have:
                - Headline: `schema1:headline`

            {relevant_examples}

            ### Your Task:
            Based on the user's query and the relevant examples (if any), perform the following:
            1.  **Analyze user intent.** If the user asks to "list" or "show" items and provides no specific criteria, create a broad SPARQL query for that type.
            2.  For search queries, write a specific SPARQL query using a case-insensitive `FILTER`.
            3.  Generate relevant keywords for vector search.

            ### User Query:
            "{user_query}"

            ### Output Example (for a general request):
            {{
              "sparql": "PREFIX ex: <http://example.org/> PREFIX ns1: <http://example.org/pred/> SELECT ?claim ?text ?status WHERE {{ ?claim a ex:Claim ; ex:canonicalForm ?text ; ns1:label ?status . }} LIMIT 10",
              "keywords": ["list all claims", "show claims"]
            }}
            """

        try:
            # --- Dynamic Few-Shot Prompting Logic ---
            
            # 1. Extract keywords for pre-fetching
            prefetch_keywords = self._extract_simple_keywords(user_query)
            
            # 2. Pre-fetch real examples from the KG
            formatted_examples = self._prefetch_examples(prefetch_keywords)
            
            # 3. Inject the dynamic examples and user query into the final prompt
            final_planner_prompt = PLANNER_PROMPT.format(
                relevant_examples=formatted_examples,
                user_query=user_query
            )

            # 4. Generate the plan using the new, context-rich prompt
            plan_text = self._generate_plan_text(final_planner_prompt)
            
            # --- End of Dynamic Logic ---

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
            if sparql_query and self.kg_store:
                try:
                    print(f"[Hybrid] Executing SPARQL:\n{sparql_query}")
                    results_iter = self.kg_store.query(sparql_query)
                    vars_list = [v.value for v in results_iter.variables]
                    results_list = list(results_iter)

                    for sol in results_list:
                        # --- IMPROVED: Capture all variables from the query ---
                        result_dict = {var: str(sol[var].value) for var in vars_list if sol.get(var)}

                        # Create a nicely formatted text string from the dictionary
                        text_parts = [f"{key}: {value}" for key, value in result_dict.items()]
                        full_text = " | ".join(text_parts)

                        candidate = {
                            "uri": result_dict.get(vars_list[0], full_text[:128]), # Fallback URI
                            "text": full_text
                        }
                        
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
                            search_builder = self.vector_table.search(q_emb).limit(top_k_vector)
                            res = search_builder.to_list()
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
                if not uri: continue
                merged[uri] = { "uri": uri, "text": g.get("text"), "in_graph": True, "distance": None, "graph_rank": 1 }

            for v in vector_candidates:
                uri = v.get("uri") or (v.get("text")[:128] if v.get("text") else None)
                if not uri: continue
                if uri in merged:
                    merged[uri]["distance"] = v.get("distance")
                else:
                    merged[uri] = { "uri": uri, "text": v.get("text"), "in_graph": False, "distance": v.get("distance"), "graph_rank": 0 }

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
                formatted_lines.append(f"{i}. [{prov}{dist_str}] {text_snip}")

            context = "Combined retrieval results (hybrid Graph-RAG):\n\n" + "\n\n".join(formatted_lines)

            if len(context) > 5000:
                context = context[:5000] + "\n...[truncated for brevity]..."

            elapsed = round(time.time() - start_time, 2)
            print(f"[Hybrid] Completed hybrid retrieval in {elapsed}s with {len(top_items)} results.")

            return context

        except Exception as e:
            print(f"[Hybrid] Fatal error: {e}\n{traceback.format_exc()}")
            return json.dumps({ "success": False, "error": str(e) }, ensure_ascii=False)

    def _generate_plan_text(self, plan_prompt: str):
        print("[Planner] Starting planner call...")
        try:
            local_model = get_default_model()
        except Exception as e:
            print(f"[Planner][ERROR] get_default_model() failed: {e}")
            return ""

        async def _collect_stream():
            chunks = []
            try:
                # Note: System prompt is now baked into the main prompt for simplicity with this technique
                messages = [{"role": "user", "content": [{"type": "text", "text": plan_prompt}]}]
                async for event in local_model.stream(messages):
                    content = None
                    if hasattr(event, "content"): content = event.content
                    elif isinstance(event, dict):
                        if "contentBlockDelta" in event: content = event["contentBlockDelta"].get("delta", {}).get("text")
                    
                    if isinstance(content, str) and content.strip():
                        chunks.append(content)
                return "".join(chunks)
            except Exception as ex:
                print(f"[Planner][ERROR] Stream exception: {ex}")
                return ""

        try:
            plan_text = asyncio.run(_collect_stream())
            print(f"[Planner] plan_text length: {len(plan_text)}")
            return plan_text or ""
        except Exception as e:
            print(f"[Planner][FATAL] asyncio.run failed: {e}")
            return ""

    def _parse_llm_json(self, text):
        if not text: return {}
        s = str(text)
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
        start = s.find('{')
        end = s.rfind('}')
        if start == -1 or end == -1:
            print(f"[Planner] Could not find JSON braces. Raw plan: {s[:1000]}")
            return {}
        try:
            return json.loads(s[start:end+1])
        except Exception as e:
            print(f"[Planner] JSON parse failed: {e}, raw: {s[:1000]}")
            return {}

    def _embed_text(self, text: str) -> Optional[List[float]]:
        try:
            emb = generate_embeddings([text], batch_size=1)
            return emb[0] if emb else None
        except Exception as e:
            print(f"Embedding error: {e}")
        return None

    def _hybrid_score(self, vector_distance: Optional[float], in_graph: bool, alpha: float = 0.75, beta: float = 0.25) -> float:
        if vector_distance is None: vec_score = 0.0
        else: vec_score = 1.0 / (1.0 + max(1e-6, float(vector_distance)))
        graph_boost = 1.0 if in_graph else 0.0
        return alpha * vec_score + beta * graph_boost
