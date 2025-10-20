# In ui/streamlit_pages/interactive_chat.py
import os
import streamlit as st
from datetime import datetime
import boto3
from api_client import chat, get_corresponding_ldb_info, summarize_graph_api, parse_kg_filename, group_kgs_by_job
from utils.auth import get_id_token
import jwt
import json

PUBLIC_GRAPH = "s3://madsiftpublic/putinmissing-all-rnr-threads_kg.ttl"
PUBLIC_DB = "s3://madsiftpublic/vector_store"

PRIVATE_BUCKET = os.getenv("PRIVATE_BUCKET")
PRIVATE_PREFIX = os.getenv("PRIVATE_PREFIX")

# --- (get_user_sub and list_user_ttls functions remain exactly the same) ---
def get_user_sub():
    """Extract Cognito user sub (unique id) from the ID token."""
    try:
        id_token = get_id_token()
        decoded = jwt.decode(id_token, options={"verify_signature": False})
        return decoded.get("sub")
    except Exception as e:
        st.error(f"Failed to decode Cognito token: {e}")
        return None

def list_user_ttls(user_sub):
    """List all TTL files for the user in S3 with rich metadata."""
    s3 = boto3.client("s3")
    prefix = f"{PRIVATE_PREFIX}/{user_sub}/"
    
    try:
        response = s3.list_objects_v2(Bucket=PRIVATE_BUCKET, Prefix=prefix)
        files = []
        for obj in response.get("Contents", []):
            if obj["Key"].endswith(".ttl"):
                s3_path = f"s3://{PRIVATE_BUCKET}/{obj['Key']}"
                
                # Get object metadata
                try:
                    head_response = s3.head_object(Bucket=PRIVATE_BUCKET, Key=obj["Key"])
                    metadata = head_response.get('Metadata', {})
                    
                    # Create display name from metadata
                    if metadata:
                        subreddit = metadata.get('subreddit', 'unknown')
                        query = metadata.get('query', 'No query')
                        created_at = metadata.get('createdat', 'Unknown date')
                        post_count = metadata.get('postcount', '0')
                        
                        # Format timestamp for display
                        try:
                            from datetime import datetime
                            dt = datetime.strptime(created_at, "%Y%m%d_%H%M%S")
                            formatted_date = dt.strftime("%b %d, %Y %H:%M")
                        except:
                            formatted_date = created_at
                        
                        display_name = f"Reddit: {subreddit} - \"{query}\" ({formatted_date}) - {post_count} posts"
                    else:
                        # Fallback to filename-based display
                        filename = obj["Key"].split('/')[-1]
                        display_name = filename
                    
                    files.append({
                        'path': s3_path,
                        'display_name': display_name,
                        'metadata': metadata
                    })
                except Exception as meta_e:
                    # Fallback if metadata retrieval fails
                    filename = obj["Key"].split('/')[-1]
                    files.append({
                        'path': s3_path,
                        'display_name': filename,
                        'metadata': {}
                    })
        
        return files
    except Exception as e:
        st.warning(f"Could not list TTLs: {e}")
        return []


def load_summary(job_id: str, username: str) -> dict:
    """Load summary for a job from S3"""
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
    """Display summary in UI"""
    if not summary:
        st.info("No summary available yet (summaries are generated after 4 iterations)")
        return
    
    st.subheader("📊 KG Summary")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Triples", summary.get('total_triples', 0))
    with col2:
        st.metric("Total Claims", summary.get('total_claims', 0))
    with col3:
        st.metric("Total Entities", summary.get('total_entities', 0))
    
    # Narrative summary
    st.write("### Summary")
    st.write(summary.get('summary', 'No summary available'))
    
    # Timestamp
    st.caption(f"Generated: {summary.get('timestamp', 'Unknown')}")

def initialize_chat_session():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = f"st_session_{int(datetime.now().timestamp())}"
    if "selected_kg" not in st.session_state:
        st.session_state.selected_kg = PUBLIC_GRAPH

def display_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools_used"):
            with st.expander("Tools Used by Backend"):
                st.json(message["tools_used"])

def show():
    st.title("💬 Interactive KG Chat")
    st.markdown("*Chat with and summarize your personalized or shared Knowledge Graphs*")

    initialize_chat_session()

    user_sub = get_user_sub()
    user_files = list_user_ttls(user_sub) if user_sub else []
    
    # Group user files by job_id
    grouped_files = group_kgs_by_job(user_files) if user_files else {}
    
    # Build options list with public graph first
    options = [{"path": PUBLIC_GRAPH, "display_name": "Public: Putin Missing - All Threads (Static)", "group": "public"}]
    
    # Add grouped KGs
    for group_key, files in grouped_files.items():
        if group_key == 'ungrouped':
            # Add ungrouped files individually
            for file in files:
                options.append({**file, "group": "ungrouped"})
        else:
            # Add files from job groups
            for file in files:
                options.append({**file, "group": group_key})
    
    # Create display mapping
    display_names = [opt["display_name"] for opt in options]
    path_mapping = {opt["display_name"]: opt["path"] for opt in options}
    current_path = st.session_state.get("selected_kg", PUBLIC_GRAPH)
    current_display = next((opt["display_name"] for opt in options if opt["path"] == current_path), options[0]["display_name"])
    
    # Display grouped KGs in sidebar or expander
    with st.expander("📚 Your Knowledge Graphs", expanded=True):
        # Show job groups
        for group_key, files in grouped_files.items():
            if group_key == 'ungrouped':
                st.write("**Individual KGs (old format)**")
            else:
                # Display job group header
                col_header, col_summary = st.columns([3, 1])
                with col_header:
                    st.write(f"**Job: {group_key}** ({len(files)} KGs)")
                    
                    # Show date range
                    if len(files) > 1:
                        timestamps = []
                        for f in files:
                            parsed = parse_kg_filename(f['path'].split('/')[-1])
                            if parsed['timestamp']:
                                timestamps.append(parsed['timestamp'])
                        if timestamps:
                            timestamps.sort()
                            try:
                                start_dt = datetime.strptime(timestamps[0], "%Y%m%d_%H%M%S")
                                end_dt = datetime.strptime(timestamps[-1], "%Y%m%d_%H%M%S")
                                st.caption(f"📅 {start_dt.strftime('%b %d, %H:%M')} → {end_dt.strftime('%b %d, %H:%M')}")
                            except:
                                st.caption(f"📅 {timestamps[0]} → {timestamps[-1]}")
                
                with col_summary:
                    # Show summary button for job groups
                    if st.button("📈 Summary", key=f"summary_{group_key}", use_container_width=True):
                        summary = load_summary(group_key, user_sub)
                        if summary:
                            display_summary(summary)
                        else:
                            st.info("No summary available yet (generated after 4 iterations)")
            
            # List files in this group
            for file in files:
                filename = file['path'].split('/')[-1]
                parsed = parse_kg_filename(filename)
                
                # Create a compact display
                if parsed['timestamp']:
                    try:
                        dt = datetime.strptime(parsed['timestamp'], "%Y%m%d_%H%M%S")
                        time_str = dt.strftime("%b %d, %H:%M")
                    except:
                        time_str = parsed['timestamp']
                    
                    button_label = f"  📊 {time_str}"
                else:
                    button_label = f"  📊 {filename}"
                
                if st.button(button_label, key=file['path'], use_container_width=True):
                    st.session_state['selected_kg'] = file['path']
                    st.rerun()
    
    # --- Simplified selection display ---
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_display = st.selectbox(
            "Select Knowledge Graph (.ttl)",
            display_names,
            index=display_names.index(current_display) if current_display in display_names else 0,
            label_visibility="collapsed"
        )
        selected_kg = path_mapping[selected_display]
        st.session_state.selected_kg = selected_kg

    with col2:
        if st.button("🪄 Summarize", use_container_width=True):
            if not selected_kg:
                st.warning("Please select a Knowledge Graph to summarize.")
            else:
                ldb_info = get_corresponding_ldb_info(selected_kg)
                with st.spinner("🧠 Generating summary..."):
                    result = summarize_graph_api(
                        "summarize",
                        st.session_state.chat_session_id,
                        kg_path=st.session_state.selected_kg
                    )

                if result and not result.get("error"):
                    summary_text = result.get('summary', 'No summary available.')
                    top_degree_nodes = result.get('top_degree', [])
                    top_betweenness_nodes = result.get('top_betweenness', [])

                    formatted_content = f"### 📊 Summary of `{selected_kg.split('/')[-1]}`\n\n"
                    formatted_content += f"**Overall Summary:**\n{summary_text}\n\n"
                    
                    if top_degree_nodes:
                        formatted_content += "**Most Connected Nodes (Highest Degree):**\n"
                        for node in top_degree_nodes: formatted_content += f"- `{node}`\n"
                    
                    if top_betweenness_nodes:
                        formatted_content += "\n**Most Influential Nodes (Highest Betweenness):**\n"
                        for node in top_betweenness_nodes: formatted_content += f"- `{node}`\n"

                    summary_msg = {"role": "assistant", "content": formatted_content}
                    st.session_state.chat_messages.append(summary_msg)
                    st.rerun()
                else:
                    st.error(f"❌ Summarization failed: {result.get('error', 'Unknown error')}")

    ldb_info = get_corresponding_ldb_info(selected_kg)
    st.info(f"**Selected KG:** {selected_kg}\n\n**Vector DB:** {ldb_info['ldb_path']}\n\n**Table:** {ldb_info['table_name'] or 'Default'}")

    # --- Chat history display and input (NO CHANGES HERE) ---
    for msg in st.session_state.chat_messages:
        display_message(msg)

    if prompt := st.chat_input("Ask a question about this KG..."):
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat_messages.append(user_msg)
        display_message(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Querying your Knowledge Graph..."):
                response = chat(
                    prompt,
                    st.session_state.chat_session_id,
                    kg_path=st.session_state.selected_kg
                )

        assistant_msg = {
            "role": "assistant",
            "content": response.get("content", "Error: No content received."),
            "tools_used": response.get("tools_used", [])
        }
        st.session_state.chat_messages.append(assistant_msg)
        st.rerun()
