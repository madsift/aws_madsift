# In ui/streamlit_pages/reddit_kg_builder.py

import streamlit as st
import jwt
from api_client import autokg_process
from utils.auth import get_id_token

def get_user_sub():
    """
    Extracts the Cognito user's unique ID (sub) from the ID token.
    This is essential for the backend to identify the user.
    """
    try:
        id_token = get_id_token()
        if not id_token:
            st.error("Authentication error: Could not retrieve user token. Please log in again.")
            return None
        decoded = jwt.decode(id_token, options={"verify_signature": False})
        return decoded.get("sub")
    except Exception as e:
        st.error(f"Authentication error: Could not decode user token. Please log in again. Error: {e}")
        return None

def show():
    """
    Shows a simple UI to configure and dispatch a one-time Reddit KG building job.
    This version is correctly authenticated.
    """
    st.title("🔗 One-Time Knowledge Graph Builder")
    st.markdown("*Fetch live Reddit posts and build a Knowledge Graph via the backend.*")
    
    # --- Configuration Form ---
    st.header("📋 Configuration")
    
    with st.form(key="one_time_kg_form"):
        col1, col2 = st.columns(2)
        with col1:
            subreddit = st.text_input("Subreddit:", value="worldnews", placeholder="e.g., worldnews")
        with col2:
            limit = st.slider("Number of Posts to Fetch:", 5, 50, 10, 5)

        with st.expander("🔧 Advanced Options"):
            extract_claims = st.checkbox("Extract Claims (uses AI on backend)", value=True)
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 Build Knowledge Graph", type="primary", use_container_width=True)

    # --- After Form Submission ---
    if submitted:
        # --- ESSENTIAL AUTHENTICATION STEP ---
        user_sub = get_user_sub()
        if not user_sub:
            # Error is already shown by get_user_sub(), so we just stop.
            return
            
        if not subreddit:
            st.error("Please enter a subreddit name.")
        else:
            payload = {
                "action": "build_reddit_kg",
                "query": subreddit,
                "subreddit": subreddit,
                "limit": limit,
                "extract_claims": extract_claims
            }

            with st.spinner(f"Sending request to build KG for 'r/{subreddit}'..."):
                result = autokg_process(action="build_reddit_kg", payload=payload)
            
            # --- Simple Success/Failure Messaging ---
            st.markdown("---")
            st.header("🎉 Dispatch Result")

            if result and result.get("success"):
                job_id = result.get("job_id")
                st.success("✅ Job was successfully dispatched to the backend!")
                st.info(f"**Job ID:** `{job_id}`")
                st.markdown(
                    """
                    The backend will now process this request. This may take a few minutes.
                    
                    **Once available, your new Knowledge Graph will appear in the "Your Knowledge Graphs" section on the 💬 Interactive Chat page.**
                    """
                )
            else:
                st.error("❌ The job could not be dispatched to the backend.")
                st.write("The backend returned the following error:")
                st.json(result)
