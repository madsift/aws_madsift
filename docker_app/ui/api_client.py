# ui/api_client.py

import os
import requests
import json
import streamlit as st
from typing import Optional, List, Dict

def get_corresponding_ldb_info(kg_path: str) -> dict:
    """Convert KG path to corresponding LanceDB database path and table name"""
    if not kg_path:
        return {"ldb_path": None, "table_name": None}
        
    # Handle public files
    if kg_path == "s3://madsiftpublic/putinmissing-all-rnr-threads_kg.ttl":
        return {
            "ldb_path": "s3://madsiftpublic/vector_store",
            "table_name": None  # Use default table selection for public
        }
    
    # For user files: extract database path and table name
    if kg_path.endswith('.ttl'):
        # Extract filename without extension
        filename = kg_path.split('/')[-1][:-4]  # Remove .ttl
        
        # Database path is the directory containing the KG file
        db_path = '/'.join(kg_path.split('/')[:-1])
       
        if filename.startswith('reddit_'):
            table_name = f"{filename}_vectorstore"
        else:
            table_name = f"{filename}_vectorstore"
        
        return {
            "ldb_path": db_path,
            "table_name": table_name
        }
    
    return {"ldb_path": None, "table_name": None}


def parse_kg_filename(filename: str) -> dict:
    """
    Parse KG filename to extract components.
    Supports both old and new formats.
    
    Args:
        filename: KG filename (e.g., 'reddit_Nepal_kg-123_20250117_143022.ttl')
    
    Returns:
        dict: {
            'subreddit': str,
            'job_id': str or None,
            'timestamp': str,
            'format': 'new' or 'old'
        }
    """
    import re
    
    # Remove .ttl extension if present
    if filename.endswith('.ttl'):
        filename = filename[:-4]
    
    # Try new format: reddit_{subreddit}_{job_id}_{timestamp}
    pattern = r'reddit_(.+?)_(kg-\d+-\w+)_(\d{8}_\d{6})'
    match = re.match(pattern, filename)
    if match:
        return {
            'subreddit': match.group(1),
            'job_id': match.group(2),
            'timestamp': match.group(3),
            'format': 'new'
        }
    
    # Try old format: reddit_{subreddit}_{timestamp}
    pattern = r'reddit_(.+?)_(\d{8}_\d{6})'
    match = re.match(pattern, filename)
    if match:
        return {
            'subreddit': match.group(1),
            'job_id': None,
            'timestamp': match.group(2),
            'format': 'old'
        }
    
    # Unknown format
    return {
        'subreddit': None,
        'job_id': None,
        'timestamp': None,
        'format': 'unknown'
    }


def group_kgs_by_job(kg_files: list) -> dict:
    """
    Group KG files by job_id.
    
    Args:
        kg_files: List of KG file dicts with 'path' and 'display_name'
    
    Returns:
        dict: {
            'job_id': [list of files],
            'ungrouped': [list of old format files]
        }
    """
    groups = {}
    
    for file in kg_files:
        filename = file['path'].split('/')[-1]
        parsed = parse_kg_filename(filename)
        
        if parsed['job_id']:
            # New format - group by job_id
            job_id = parsed['job_id']
            if job_id not in groups:
                groups[job_id] = []
            groups[job_id].append(file)
        else:
            # Old format or unknown - put in ungrouped
            if 'ungrouped' not in groups:
                groups['ungrouped'] = []
            groups['ungrouped'].append(file)
    
    return groups


def load_summary(job_id: str, username: str) -> dict:
    """Load summary for a job from S3"""
    import boto3
    
    s3 = boto3.client('s3')
    prefix = f"knowledge_graph/{username}/summaries/{job_id}_summary_"
    
    try:
        # List summaries for this job
        response = s3.list_objects_v2(
            Bucket='madsift',
            Prefix=prefix
        )
        
        if 'Contents' not in response or len(response['Contents']) == 0:
            return None
        
        # Get most recent summary
        latest = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
        
        # Download and parse
        obj = s3.get_object(Bucket='madsift', Key=latest['Key'])
        summary = json.loads(obj['Body'].read())
        
        return summary
    except Exception as e:
        st.error(f"Failed to load summary: {str(e)}")
        return None


def display_summary(summary: dict):
    """Display production-ready summary in UI"""
    if not summary:
        st.info("📊 No summary available yet")
        return
    
    # Header with job ID
    job_id = summary.get('job_id', 'Unknown')
    st.markdown(f"### 📊 Knowledge Graph Summary")
    st.caption(f"Job ID: `{job_id}` • Generated: {summary.get('timestamp', 'Unknown')[:19]}")
    
    # Metrics in a clean row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📈 Triples", f"{summary.get('total_triples', 0):,}")
    with col2:
        st.metric("🎯 Claims", f"{summary.get('total_claims', 0):,}")
    with col3:
        st.metric("👥 Entities", f"{summary.get('total_entities', 0):,}")
    
    st.markdown("")  # Spacing
    
    # Narrative summary - full width with nice styling
    summary_text = summary.get('summary', 'No summary available')
    
    # Handle both string and dict summaries
    if isinstance(summary_text, dict):
        if 'error' in summary_text:
            st.error(f"⚠️ Summary generation encountered an issue")
        else:
            st.info("📝 **Analysis**")
            st.write(summary_text)
    else:
        st.info("📝 **Analysis**")
        st.write(summary_text)


def display_detailed_summary(summary: dict):
    """
    Display detailed summary from graph_summary.py output.
    
    Expected structure:
    {
        "summary": str,
        "top_degree": [(node, score), ...],
        "top_betweenness": [(node, score), ...],
        "context_samples": [str, ...],
        "graph_stats": {"nodes": int, "edges": int},
        "query": str,
        "timestamp": str
    }
    """
    if not summary:
        st.info("No summary available yet")
        return
    
    st.subheader("📊 Detailed Knowledge Graph Analysis")
    
    # Graph Statistics
    if 'graph_stats' in summary:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Nodes", summary['graph_stats'].get('nodes', 0))
        with col2:
            st.metric("Total Edges", summary['graph_stats'].get('edges', 0))
    
    # Query Context
    if summary.get('query'):
        st.info(f"**Query Focus:** {summary['query']}")
    
    # Main Summary
    st.write("### 📝 Summary")
    st.write(summary.get('summary', 'No summary available'))
    
    # Top Entities by Degree Centrality
    if summary.get('top_degree'):
        st.write("### 🔗 Most Connected Entities (Degree Centrality)")
        for node, score in summary['top_degree'][:5]:
            # Extract readable name from URI
            node_name = node.split('/')[-1] if '/' in node else node
            st.write(f"- **{node_name}** (score: {score:.3f})")
    
    # Key Bridge Nodes
    if summary.get('top_betweenness'):
        st.write("### 🌉 Key Bridge Nodes (Betweenness Centrality)")
        for node, score in summary['top_betweenness'][:5]:
            node_name = node.split('/')[-1] if '/' in node else node
            st.write(f"- **{node_name}** (score: {score:.3f})")
    
    # Semantic Context Samples
    if summary.get('context_samples'):
        with st.expander("📄 Contextual Evidence (Semantic Retrieval)", expanded=False):
            for i, context in enumerate(summary['context_samples'][:5], 1):
                st.write(f"{i}. \"{context.strip()}\"")
    
    # Timestamp
    st.caption(f"Generated: {summary.get('timestamp', 'Unknown')}")


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:3000")


def _get_auth_headers():
    """Get headers with Cognito access token for API authentication."""
    headers = {"Content-Type": "application/json"}
    
    # Get tokens from session state
    tokens = st.session_state.get("tokens", {})
    access_token = tokens.get("access_token")
    
    # Debug logging
    if access_token:
        #st.write(f"[DEBUG] Access token found: {access_token[:50]}...")
        headers["Authorization"] = f"Bearer {access_token}"
    else:
        st.warning("[DEBUG] ⚠️ No access token found in session state!")
        st.write(f"[DEBUG] Session state keys: {list(st.session_state.keys())}")
        st.write(f"[DEBUG] Tokens dict: {tokens}")
    
    return headers


def _parse_response(response):
    """Parse API response handling Lambda proxy format."""
    try:
        data = response.json()
        # Handle Lambda proxy integration format
        if isinstance(data, dict) and "body" in data:
            try:
                return json.loads(data["body"])
            except (json.JSONDecodeError, TypeError):
                return {"content": data["body"]}
        return data
    except json.JSONDecodeError:
        return {"content": response.text, "error": "Invalid JSON response"}


def _handle_request(method: str, url: str, **kwargs):
    """Common request handler with error handling."""
    try:
        kwargs.setdefault("headers", _get_auth_headers())
        kwargs.setdefault("timeout", 60)
        
        # Debug: Show request details
        #st.write(f"[DEBUG] Making {method} request to: {url}")
        #st.write(f"[DEBUG] Headers: {kwargs['headers']}")
        
        resp = requests.request(method, url, **kwargs)
        
        # Debug: Show response
        #st.write(f"[DEBUG] Response status: {resp.status_code}")
        if resp.status_code >= 400:
            st.write(f"[DEBUG] Response body: {resp.text}")
        
        resp.raise_for_status()
        return _parse_response(resp)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return {"error": "Unauthorized - please log in again", "status_code": 401}
        elif e.response.status_code == 403:
            return {"error": "Forbidden - insufficient permissions", "status_code": 403}
        else:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}", "status_code": e.response.status_code}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error: {str(e)}"}


def chat(message: str, session_id: str, user_id: str = None, kg_path: str = None, chat_options: dict = None):
    """Send chat message to backend (supports KG path and extra options)."""
    url = f"{API_BASE_URL}/chat"
    
    payload = {
        "message": message,
        "session_id": session_id,
        "type": "chat",
    }
    
    if user_id:
        payload["user_id"] = user_id
    if kg_path:
        payload["kg_path"] = kg_path
        # Auto-resolve corresponding LanceDB info
        ldb_info = get_corresponding_ldb_info(kg_path)
        payload["ldb_path"] = ldb_info["ldb_path"]
        payload["ldb_table_name"] = ldb_info["table_name"]
    if chat_options:
        payload["chat_options"] = chat_options

    return _handle_request("POST", url, json=payload)


def summarize_graph_api(message: str, session_id: str, user_id: str = None, kg_path: str = None, chat_options: dict = None):
    """Send summarization request to backend API."""
    url = f"{API_BASE_URL}/chat"
    
    payload = {
        "message": message,
        "session_id": session_id,
        "type": "summarize",
    }
    
    if user_id:
        payload["user_id"] = user_id
    if kg_path:
        payload["kg_path"] = kg_path
        # Auto-resolve corresponding LanceDB info
        ldb_info = get_corresponding_ldb_info(kg_path)
        payload["ldb_path"] = ldb_info["ldb_path"]
        payload["ldb_table_name"] = ldb_info["table_name"]
    if chat_options:
        payload["chat_options"] = chat_options

    return _handle_request("POST", url, json=payload)
    
    
def health():
    """Check API health status."""
    url = f"{API_BASE_URL}/health"
    return _handle_request("GET", url, timeout=10)


def kg_process(action: str, payload: dict):
    """Process knowledge graph operations."""
    url = f"{API_BASE_URL}/kg-processor"
    body = {"action": action, "payload": payload}
    return _handle_request("POST", url, json=body, timeout=600)


def autokg_process(action: str, payload: dict):
    """Process knowledge graph operations."""
    url = f"{API_BASE_URL}/autokg-processor"
    body = payload  # Send the payload directly
    return _handle_request("POST", url, json=body, timeout=600)

    
def start_monitoring_job(payload: dict, processor):
    """Start an auto-monitoring job."""
    url = f"{API_BASE_URL}/{processor}"
    body = {"action": "start_monitoring", "payload": payload} 
    return _handle_request("POST", url, json=body, timeout=600)


def stop_monitoring_job(job_id: str, processor):
    """Stop a running monitoring job."""
    url = f"{API_BASE_URL}/{processor}"
    body = {"action": "stop_monitoring", "payload": {"job_id": job_id}} 
    return _handle_request("POST", url, json=body, timeout=60)



def get_user_files(prefix: str = ""):
    """Get list of user's files from S3."""
    url = f"{API_BASE_URL}/files"
    params = {"prefix": prefix} if prefix else {}
    return _handle_request("GET", url, params=params, timeout=30)


def upload_file(file_content: bytes, filename: str, content_type: str = "application/octet-stream"):
    """Upload a file to user's S3 directory."""
    url = f"{API_BASE_URL}/files/upload"
    body = {
        "filename": filename,
        "content": file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content,
        "content_type": content_type
    }
    return _handle_request("POST", url, json=body, timeout=120)
