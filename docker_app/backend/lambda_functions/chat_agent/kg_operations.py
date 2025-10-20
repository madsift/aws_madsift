#!/usr/bin/env python3
"""
Knowledge Graph Operations - KG loading and querying functionality

This module handles all Knowledge Graph related operations including:
- Loading KG from S3 or local paths
- Oxigraph store management
- SPARQL query execution
- Vector store loading and management
"""

import os
import re
import boto3
from pyoxigraph import Store
from typing import Optional
from common.lancedb_helper import connect_lancedb


class KGOperations:
    """Handles Knowledge Graph and Vector Store operations"""
    
    def __init__(self):
        self.kg_store = None
        self.kg_path = None
        self.vector_db = None
        self.vector_table = None
        self.ldb_path = None
        self.region = None

    def load_knowledge_graph(self, kg_path: str, region: Optional[str] = None) -> bool:
        """Load the selected knowledge graph (TTL) into Oxigraph."""
        # Reuse cached store if path is the same
        if self.kg_store and self.kg_path == kg_path:
            print("Reusing cached Oxigraph store.")
            return True

        print(f"🧠 Loading KG from: {kg_path}")
        self.kg_path = kg_path
        self.region = region
        local_path = kg_path

        # If path is on S3, download to /tmp/
        if kg_path.startswith("s3://"):
            try:
                s3 = boto3.client("s3", region_name=region)
                bucket, key = kg_path.replace("s3://", "").split("/", 1)
                local_path = f"/tmp/{os.path.basename(key)}"
                print(f"Downloading {kg_path} → {local_path}")
                s3.download_file(bucket, key, local_path)
            except Exception as e:
                print(f"❌ Failed to download KG from S3: {e}")
                raise

        # Initialize Oxigraph store
        self.kg_store = Store()
        try:
            with open(local_path, "rb") as f:
                data = f.read()
            self.kg_store.load(data, format="text/turtle")
            print(f"✅ Loaded KG ({os.path.getsize(local_path)} bytes) successfully")
        except Exception as e:
            print(f"❌ Failed to load KG into Oxigraph: {e}")
            raise

        return True

    def load_vector_store(self, ldb_path: Optional[str], username: str = "default", table_name: Optional[str] = None) -> bool:
        """Connect to LanceDB (S3 or local) and open a default table."""
        self.vector_db = None
        self.vector_table = None
        self.ldb_path = ldb_path

        if not ldb_path:
            print("No ldb_path provided; skipping vector store load.")
            return False

        print(f"📦 Loading vector store from: {ldb_path}")
        try:
            if ldb_path.startswith("s3://"):
                _, path_no_scheme = ldb_path.split("s3://", 1)
                bucket, prefix = path_no_scheme.split("/", 1)
                s3_prefix_user = prefix
                self.vector_db = connect_lancedb(bucket, s3_prefix_user, local_path=None)
            else:
                # if your helper supports local path, pass as prefix with None bucket
                self.vector_db = connect_lancedb(None, ldb_path, local_path=None)
        except Exception as e:
            print(f"❌ Failed connecting to LanceDB: {e}")
            self.vector_db = None
            return False

        try:
            tables = self.vector_db.table_names()
            print(f"Available tables in vector DB: {list(tables)}")
            
            if table_name:
                # Try exact match first
                if table_name in tables:
                    self.vector_table = self.vector_db.open_table(table_name)
                    print(f"✅ Using specified vector table: {table_name}")
                else:
                    # Try sanitized version
                    sanitized = re.sub(r'[^a-zA-Z0-9-_]', '_', table_name)
                    if sanitized in tables:
                        self.vector_table = self.vector_db.open_table(sanitized)
                        print(f"✅ Using sanitized vector table: {sanitized}")
                    else:
                        print(f"❌ Specified vector table '{table_name}' (or '{sanitized}') not found in DB.")
                        print(f"Available tables: {list(tables)}")
                        # DO NOT fall back to wrong dataset - this causes incorrect results
                        print(f"⚠️ Cannot proceed without correct vector table. Vector search will be disabled.")
                        self.vector_table = None
            else:
                if tables:
                    chosen = list(tables)[0]
                    print(f"Using first available vector table: {chosen}")
                    self.vector_table = self.vector_db.open_table(chosen)
                else:
                    print("⚠️ No tables found in vector DB.")
                    self.vector_table = None
        except Exception as e:
            print(f"❌ Error opening vector table: {e}")
            self.vector_db = None
            self.vector_table = None
            return False

        print("✅ Vector store loaded.")
        return True

    def query_knowledge_graph_oxigraph(self, sparql_query: str) -> str:
        """
        Executes a SPARQL query against the loaded in-memory Oxigraph knowledge graph.
        """
        print("[DEBUG] Attempting to query the Knowledge Graph.")

        if not self.kg_store:
            error_message = "Error: Knowledge graph is not loaded. Cannot execute query."
            print(f"[DEBUG] {error_message}")
            raise ValueError(error_message)

        print(f"[DEBUG] Executing SPARQL Query:\n---\n{sparql_query}\n---")

        try:
            results_iterator = self.kg_store.query(sparql_query)
            variables = [var.value for var in results_iterator.variables]
            query_results = list(results_iterator)
            num_results = len(query_results)
            print(f"[DEBUG] Query executed successfully. Found {num_results} result(s).")

            if num_results == 0:
                return "The query executed successfully but returned no results. 🤷‍♂️"

            formatted_results = []
            for i, solution in enumerate(query_results, 1):
                row = {}
                for var_name in variables:
                    term = solution[var_name]
                    if term:
                        row[var_name] = str(term.value)
                row_str = " | ".join(f"{k}: {v}" for k, v in row.items())
                formatted_results.append(f"{i}. {row_str}")

            return "\n".join(formatted_results)

        except Exception as e:
            error_message = f"An error occurred while querying the knowledge graph: {e}"
            print(f"❌ {error_message}")
            return error_message

    def query_knowledge_graph_fuseki(self, query: str) -> str:
        """Query external Fuseki endpoint"""
        GRAPH_DB_ENDPOINT = os.getenv("GRAPH_DB_QUERY_ENDPOINT")
        if not GRAPH_DB_ENDPOINT:
            return "Error: Graph database endpoint is not configured."

        try:
            import requests
            import json
            print(f"SPARQL Query Generated by Agent:\n---\n{query}\n---")
            headers = {'Accept': 'application/sparql-results+json'}
            params = {'query': query}
            print(f"Sending SPARQL query to {GRAPH_DB_ENDPOINT}")
            response = requests.post(GRAPH_DB_ENDPOINT, data=params, headers=headers, timeout=20)
            response.raise_for_status()
            results = response.json()
            bindings = results.get("results", {}).get("bindings", [])
            if not bindings:
                return "The query executed successfully but returned no results."
            formatted_results = json.dumps(bindings, indent=2)
            print(formatted_results)
            return formatted_results

        except Exception as e:
            print(f"❌ Graph DB query failed: {e}")
            return f"Error querying the graph database: {str(e)}. The query might be malformed or the database unavailable."

    def _fuse_results(self, kg_results: list, vector_results: list, top_k: int = 10) -> str:
        """
        Combine and summarize KG and vector search results into a single context.
        """
        context_parts = []
        seen_texts = set()

        # Format KG results
        if kg_results:
            for row in kg_results[:top_k]:
                line = " | ".join([f"{k}: {v}" for k, v in row.items()])
                if line not in seen_texts:
                    context_parts.append(f"[KG] {line}")
                    seen_texts.add(line)

        # Format vector results
        if vector_results:
            for res in vector_results[:top_k]:
                text = res.get("text") or res.get("content") or str(res)
                score = res.get("score", "")
                key = f"[VS] ({score}) {text}"
                if text not in seen_texts:
                    context_parts.append(key)
                    seen_texts.add(text)

        # Merge both
        fused_context = "\n".join(context_parts)
        return fused_context

    def _add_kg_context(self, user_message: str) -> str:
        """
        Augments the user message with Knowledge Graph and vector store context.
        Adds high-level KG statistics (posts, claims, users) if available.
        """
        try:
            # Basic metadata if KG loaded
            if getattr(self, "kg_store", None):
                platform = "Example.org"
                node_count = 0
                try:
                    # quick count if supported
                    node_count = len(set(self.kg_store.subjects()))
                except Exception:
                    pass
                kg_context = f"A Knowledge Graph with triples {len(list(self.kg_store))} nodes from {platform} is currently loaded."
            else:
                kg_context = "No Knowledge Graph is currently loaded."

            # Vector store metadata if loaded
            if getattr(self, "vector_db", None):
                try:
                    tbls = self.vector_db.table_names()
                    table_info = ", ".join(tbls) if tbls else "no vector tables"
                    vs_context = f"A LanceDB vector store is connected with {table_info}."
                except Exception:
                    vs_context = "A LanceDB vector store connection exists."
            else:
                vs_context = "No LanceDB vector store is connected."

            return (
                f"[SYSTEM CONTEXT]\n"
                f"{kg_context}\n{vs_context}\n\n"
                f"User question:\n{user_message}"
            )

        except Exception as e:
            print(f"[KGOperations] _add_kg_context failed: {e}")
            return user_message