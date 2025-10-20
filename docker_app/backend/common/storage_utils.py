# ============================================================
# File: lambda_functions/common/storage_utils.py
# Purpose: Unified I/O and storage management for Knowledge Graphs and Vector Stores
# Author: Saket K.
# ============================================================

import os
import re
import boto3
import tempfile
import traceback
from typing import Optional, Tuple

try:
    from common.lancedb_helper import connect_lancedb
except: 
    from lancedb_helper import connect_lancedb
# Optional dependencies
try:
    from pyoxigraph import Store
except ImportError:
    Store = None

try:
    import lancedb
except ImportError:
    lancedb = None

# Config defaults
LOCAL_LANCE_PATH = "/tmp/lancedb"
REGION = os.getenv("AWS_REGION", "us-east-1")


# ----------------------------
# GENERIC HELPERS
# ----------------------------

def download_from_s3(s3_uri: str, dest_dir: str = "/tmp") -> str:
    """
    Download a file from S3 to a local path.
    Returns the local path of the downloaded file.
    """
    assert s3_uri.startswith("s3://"), "Invalid S3 URI"
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    local_path = os.path.join(dest_dir, os.path.basename(key))
    print(f"⬇️ Downloading {s3_uri} → {local_path}")
    try:
        s3 = boto3.client("s3", region_name=REGION)
        os.makedirs(dest_dir, exist_ok=True)
        s3.download_file(bucket, key, local_path)
        print(f"✅ Downloaded {os.path.getsize(local_path)} bytes")
        return local_path
    except Exception as e:
        print(f"❌ Failed to download {s3_uri}: {e}")
        traceback.print_exc()
        raise


# ----------------------------
# KNOWLEDGE GRAPH LOADER
# ----------------------------

def load_knowledge_graph(kg_path: str) -> Tuple[Optional[object], Optional[str]]:
    """
    Download (if S3) and load the selected knowledge graph into Oxigraph.
    Returns (Store object, local_path) tuple.
    """
    if not Store:
        raise ImportError("Oxigraph is not installed in this environment.")

    print(f"🧠 Loading Knowledge Graph: {kg_path}")
    local_path = kg_path

    # Handle S3 download
    if kg_path.startswith("s3://"):
        local_path = download_from_s3(kg_path)

    # Initialize Oxigraph store
    store = Store()
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        store.load(data, format="text/turtle")
        print(f"✅ KG loaded successfully ({os.path.getsize(local_path)} bytes, {kg_path})")
    except Exception as e:
        print(f"❌ Failed to load KG into Oxigraph: {e}")
        traceback.print_exc()
        raise

    return store, local_path



def load_vector_store(ldb_path: Optional[str], username: str = "default", table_name: Optional[str] = None):
    """
    Load/connect to the LanceDB vector store.
    Returns (vector_db, vector_table, local_path)
    """
    vector_db, vector_table = None, None

    if not ldb_path:
        print("ℹ️ No LanceDB path provided; skipping vector store load.")
        return None, None, None

    print(f"📦 Loading vector store from: {ldb_path}")

    try:
        if ldb_path.startswith("s3://"):
            # Parse S3 URI
            _, path_no_scheme = ldb_path.split("s3://", 1)
            bucket, prefix = path_no_scheme.split("/", 1)
            safe_username = "".join(c if c.isalnum() or c in "-_" else "_" for c in username)
            vector_db = connect_lancedb(bucket, prefix, local_path=LOCAL_LANCE_PATH)
        else:
            vector_db = connect_lancedb(None, ldb_path, local_path=LOCAL_LANCE_PATH)

        # Choose table
        if table_name:
            sanitized = re.sub(r"[^a-zA-Z0-9-_]", "_", table_name)
            if sanitized in vector_db.table_names():
                vector_table = vector_db.open_table(sanitized)
                print(f"✅ Loaded specific table: {sanitized}")
            else:
                print(f"⚠️ Table '{sanitized}' not found in vector DB.")
        else:
            tables = vector_db.table_names()
            if tables:
                chosen = list(tables)[0]
                print(f"✅ Using default table: {chosen}")
                vector_table = vector_db.open_table(chosen)
            else:
                print("⚠️ No tables found in LanceDB.")

        return vector_db, vector_table, LOCAL_LANCE_PATH

    except Exception as e:
        print(f"❌ Error connecting to LanceDB: {e}")
        traceback.print_exc()
        return None, None, None

