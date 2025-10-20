import os
import json
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF

import numpy as np
import re
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from common.aws_clients import bedrock_client
from common.lancedb_helper import connect_lancedb

MODEL_ID = "cohere.embed-v4:0"
S3_BUCKET = os.getenv("KG_BUCKET" ,"madsift")
S3_PREFIX = "knowledge_graph"  # os.getenv('LANCE_S3_PREFIX', 'lancedb/graph_embeddings')
LOCAL_LANCE_PATH = os.getenv('LOCAL_LANCE_PATH', '/tmp/lancedb')

EMBEDDING_DIM = 256
BATCH_SIZE = 96

def _sanitize_table_name(name: str) -> str:
    """Sanitizes a string to be a valid LanceDB table name."""
    name = re.sub(r'\.ttl$', '', name, flags=re.IGNORECASE)
    sanitized_name = re.sub(r'[^a-zA-Z0-9-]', '_', name)
    return sanitized_name

def generate_embeddings(texts: list, batch_size: int = BATCH_SIZE, max_workers: int = None) -> list:
    """
    Embedding generator for cohere.embed-v4:0.
    Returns embeddings as lists of int8 (dimension = EMBEDDING_DIM).
    """
    cpu_count = multiprocessing.cpu_count()
    max_workers = max_workers or max(1, min(10, math.ceil(cpu_count / 2)))
    client = bedrock_client()

    print(f"[Embedding] Starting embedding generation for {len(texts)} texts.")
    print(f"[Config] Model={MODEL_ID}, Dim={EMBEDDING_DIM}, Batch={batch_size}, Threads={max_workers}")

    all_embeddings = [None] * len(texts)
    batches = [(i, texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
    total_batches = len(batches)

    def embed_batch(start_idx, batch):
        """Makes a single Bedrock API call for a batch of texts."""
        # Build payload according to the model spec (Embed v4) :contentReference[oaicite:0]{index=0}
        payload = {
            "input_type": "search_document",               # embedding documents
            "texts": batch,
            "embedding_types": ["int8"],                  # request float embeddings
            "output_dimension": EMBEDDING_DIM,             # singular key
            "truncate": "RIGHT"                             # valid options: "LEFT", "RIGHT", "NONE"
        }
        try:
            resp = client.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                body=json.dumps(payload)
            )
            body_bytes = resp['body'].read()
            obj = json.loads(body_bytes)
            # The API returns embeddings: either a flat list (if single type) or a dict mapping types
            emb = obj.get("embeddings")
            if emb is None:
                print(f"[Warn] Batch {start_idx}-{start_idx + len(batch) - 1}: no embeddings field in response: {obj}")
                batch_embeddings = [[0.0] * EMBEDDING_DIM for _ in batch]
            else:
                if isinstance(emb, dict):
                    # if embeddings_by_type form
                    batch_embeddings = emb.get("int8")
                else:
                    # embeddings_floats form
                    batch_embeddings = emb
                # validate shape
                if not batch_embeddings or len(batch_embeddings) != len(batch):
                    print(f"[Warn] Batch {start_idx}-{start_idx + len(batch) - 1}: unexpected embedding shape: {batch_embeddings}")
                    batch_embeddings = [[0.0] * EMBEDDING_DIM for _ in batch]
        except Exception as e:
            print(f"[Error] Batch {start_idx}-{start_idx + len(batch) - 1} failed: {e}")
            # Optionally log more detailed error
            batch_embeddings = [[0.0] * EMBEDDING_DIM for _ in batch]

        return start_idx, batch_embeddings

    start_time = time.time()
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_batch, start, batch) for start, batch in batches]
        for f in as_completed(futures):
            start_idx, batch_embeddings = f.result()
            all_embeddings[start_idx:start_idx + len(batch_embeddings)] = batch_embeddings
            completed_batches += 1
            if completed_batches % 5 == 0 or completed_batches == total_batches:
                printed = min(completed_batches * batch_size, len(texts))
                print(f"[Progress] {completed_batches}/{total_batches} batches complete ({printed}/{len(texts)} texts).")

    elapsed = round(time.time() - start_time, 2)
    print(f"[Done] Embedded {len(all_embeddings)} texts in {elapsed}s using {max_workers} workers.")
    return all_embeddings

def generate_node_documents(ttl_path: str) -> dict:
    """Smarter Embedding Strategy: Node-Centric Document Generation."""
    g = Graph()
    g.parse(ttl_path, format='turtle')
    node_documents = {}
    subjects = set(g.subjects())
    for s in subjects:
        if not isinstance(s, URIRef):
            continue
        doc_lines = []
        node_uri = str(s)
        node_type = g.value(subject=s, predicate=RDF.type)
        if node_type:
            type_name = str(node_type).split('/')[-1].split('#')[-1]
            doc_lines.append(f"This is a {type_name} identified by '{node_uri.split('/')[-1]}'.")
        for p, o in g.predicate_objects(subject=s):
            predicate_str = str(p).split('/')[-1].split('#')[-1]
            if isinstance(o, Literal):
                if predicate_str in ['canonicalForm', 'content', 'text']:
                    doc_lines.append(f"The content is: \"{str(o)}\".")
                else:
                    doc_lines.append(f"Its '{predicate_str}' is '{str(o)}'.")
            elif isinstance(o, URIRef):
                object_str = str(o).split('/')[-1].split('#')[-1]
                doc_lines.append(f"It is connected to '{object_str}' through the '{predicate_str}' property.")
        if doc_lines:
            node_documents[node_uri] = " ".join(doc_lines)
    print(f"[Embedding] Generated {len(node_documents)} node documents from the graph.")
    return node_documents

def store_to_lancedb(node_docs: dict, embeddings: list, username: str, table_name: str, subreddit: str = None, timestamp: str = None, job_id: str = None):
    """Stores the node URI, text document, and its vector into a dynamically named LanceDB table."""
    safe_username = "".join(c if c.isalnum() or c in "-_" else "_" for c in username)
    bucket_to_use = S3_BUCKET 
    
    if "putin" in table_name:
        S3_PREFIX_USER = "vector_store"
        bucket_to_use = "madsiftpublic"
        sanitized_table_name = _sanitize_table_name(table_name)
    else:
        S3_PREFIX_USER = f"knowledge_graph/{safe_username}"
        # Use new naming pattern with job_id: reddit_{subreddit}_{job_id}_{timestamp}_vectorstore
        if subreddit and timestamp and job_id:
            sanitized_table_name = f"reddit_{subreddit}_{job_id}_{timestamp}_vectorstore"
        elif subreddit and timestamp:
            sanitized_table_name = f"reddit_{subreddit}_{timestamp}_vectorstore"
        else:
            sanitized_table_name = _sanitize_table_name(table_name)
    
    db = connect_lancedb(bucket_to_use, S3_PREFIX_USER, local_path=LOCAL_LANCE_PATH)

    uris = list(node_docs.keys())
    records = [{'uri': uris[i], 'text': node_docs[uris[i]], 'vector': embeddings[i]} for i in range(len(embeddings))]

    if sanitized_table_name in db.table_names():
        print(f"Table '{sanitized_table_name}' already exists. Overwriting.")
        db.drop_table(sanitized_table_name)

    if not records:
        print("No records to store in LanceDB.")
        return None

    tbl = db.create_table(sanitized_table_name, data=records)
    print(f"Stored {len(records)} embeddings in LanceDB table '{sanitized_table_name}'.")
    return tbl

def run_kg_embedding(ttl_path: str, username: str, source_name: str, subreddit: str = None, timestamp: str = None, job_id: str = None):
    """Main orchestrator function for the embedding process."""
    if not username:
        return {'success': False, 'error': 'Username is required for embedding storage.'}
    if not source_name:
        return {'success': False, 'error': 'A source_name is required to name the embedding table.'}

    node_documents = generate_node_documents(ttl_path)
    if not node_documents:
        return {'success': False, 'error': 'No documents could be generated from the TTL file.'}

    texts_to_embed = list(node_documents.values())
    embeddings = generate_embeddings(texts_to_embed)

    if not embeddings or len(embeddings) != len(texts_to_embed):
        return {'success': False, 'error': 'Failed to generate embeddings via Bedrock.'}

    tbl = store_to_lancedb(node_documents, embeddings, username, table_name=source_name, subreddit=subreddit, timestamp=timestamp, job_id=job_id)
    if not tbl:
        return {'success': False, 'error': 'Failed to store embeddings in LanceDB.'}

    # Determine final table name
    if subreddit and timestamp and job_id and "putin" not in source_name:
        final_table_name = f"reddit_{subreddit}_{job_id}_{timestamp}_vectorstore"
    elif subreddit and timestamp and "putin" not in source_name:
        final_table_name = f"reddit_{subreddit}_{timestamp}_vectorstore"
    else:
        final_table_name = _sanitize_table_name(source_name)

    return {
        'success': True,
        'stored_count': len(embeddings),
        'table_name': final_table_name,
        'table_uri': getattr(tbl, 'uri', str(tbl))
    }
