import os
import lancedb
from pathlib import Path

def connect_lancedb(s3_bucket=None, s3_prefix=None, local_path='/tmp/lancedb'):
    if s3_bucket and s3_prefix:
        uri = f"s3://{s3_bucket}/{s3_prefix}"
    else:
        Path(local_path).mkdir(parents=True, exist_ok=True)
        uri = local_path
    db = lancedb.connect(uri)
    return db
