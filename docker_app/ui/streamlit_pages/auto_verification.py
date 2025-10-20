# In: ui/streamlit_pages/auto_verification.py
import streamlit as st
from datetime import datetime, timedelta, time, timezone 
from api_client import start_monitoring_job, stop_monitoring_job, load_summary
from api_client import display_detailed_summary as display_summary
from utils.auth import get_id_token
import jwt

def show():
    """
    Displays the UI for configuring automated monitoring jobs.
    This version includes 'Minutes' as a duration unit for easier testing.
    """
    st.title("🤖 Bedrock agentcore Automated Content Monitoring")
    st.markdown(
        "Define a topic, select sources, and set a schedule to continuously monitor for new content. "
        "This configuration will be handed off to a backend agent to execute the job."
    )
    st.markdown("---")

    # --- TOP-LEVEL CONTROLS (OUTSIDE THE FORM) ---
    st.header("1. Define Monitoring Topic")
    
    query = st.text_input(
        label="**Query or Keywords**",
        placeholder="e.g., election security, climate change misinformation",
        help="The topic or keywords for the agent to monitor.",
        key="auto_query"  # ← Unique key
    )
    
    col1, col2 = st.columns(2)
    with col1:
        subreddit = st.text_input(
            label="**Subreddit Name** (for Reddit monitoring)",
            placeholder="e.g., worldnews, politics, news",
            help="The subreddit to monitor (required for Reddit source)",
            key="auto_subreddit"  # ← Unique key
        )
    with col2:
        limit = st.slider(
            label="**Posts per Check**",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help="Number of posts to fetch each time the agent checks for new content",
            key="auto_limit"  # ← Unique key
        )

    sources = st.multiselect(
        label="**Social Media Feeds to Monitor**",
        options=["Reddit", "Twitter"],
        default=["Reddit"],
        help="Select one or more sources for the agent to monitor.",
        key="auto_sources"  # ← Unique key
    )

    interval_options = {
        "Every 5 Minutes": 300, "Every 15 Minutes": 900, "Every 30 Minutes": 1800,
        "Every Hour": 3600, "Every 6 Hours": 21600, "Once a Day": 86400
    }
    selected_interval_label = st.selectbox(
        label="**Check for new content...**",
        options=list(interval_options.keys()),
        index=0,
        help="How often the backend agent should query for new posts."
    )

    st.header("2. Set Monitoring Schedule")

    schedule_type = st.radio(
        "**Scheduling Method:**",
        ["Simple Duration", "Date Range (Calendar)"],
        horizontal=True,
        key="schedule_type_radio",
        help="Choose to select a specific date range or a simple duration from now."
    )

    # --- Job Configuration Form ---
    with st.form(key="monitoring_job_form"):
        
        if schedule_type == "Simple Duration":
            st.subheader("⏱️ Set Job Duration")
            col1, col2 = st.columns(2)
            with col1:
                # --- THIS IS THE CHANGE: Default value is now 30 ---
                duration_value = st.number_input("Run for the next...", min_value=1, value=20)
            with col2:
                # --- THIS IS THE CHANGE: "Minutes" is now an option and the default ---
                duration_unit = st.selectbox("Unit", ["Minutes", "Hours", "Days"])
            
        else: 
            st.subheader("🗓️ Select Start and End Dates")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now())
                start_time = st.time_input("Start Time", value=time(9, 0))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now() + timedelta(days=7), min_value=start_date)
                end_time = st.time_input("End Time", value=time(17, 0))
        

        st.markdown("---")
        submitted = st.form_submit_button(
            label="🚀 Prepare Monitoring Job",
            type="primary",
            use_container_width=True
        )

    # --- After Form Submission ---
    if submitted:
        start_time_dt, end_time_dt = None, None
        if schedule_type == "Date Range (Calendar)":
            # For calendar, we need to treat them as UTC
            start_time_dt = datetime.combine(start_date, start_time)
            end_time_dt = datetime.combine(end_date, end_time)
        else:
            # FIX: Use datetime.utcnow() for timezone-naive UTC calculation
            start_time_dt = datetime.utcnow()
            
            if duration_unit == "Minutes":
                end_time_dt = start_time_dt + timedelta(minutes=duration_value)
            elif duration_unit == "Hours":
                end_time_dt = start_time_dt + timedelta(hours=duration_value)
            else: # Days
                end_time_dt = start_time_dt + timedelta(days=duration_value)

        # Validation
        if not query or not sources or start_time_dt >= end_time_dt:
            if not query: st.error("❌ Please enter a query.")
            if not sources: st.error("❌ Please select at least one source.")
            if start_time_dt >= end_time_dt: st.error("❌ End Date/Time must be after Start Date/Time.")
        
        # Additional validation for Reddit
        if "Reddit" in sources and not subreddit:
            st.error("❌ Please enter a subreddit name for Reddit monitoring.")
        
        if not (not query or not sources or start_time_dt >= end_time_dt or ("Reddit" in sources and not subreddit)):
            job_payload = {
                "query": query,
                "subreddit": subreddit,  # ← Added subreddit field
                "limit": limit,  # ← Added limit field
                "sources": ",".join(sources),
                "schedule": {
                    "start_time_iso": start_time_dt.isoformat(),
                    "end_time_iso": end_time_dt.isoformat(),
                },
                "interval": {
                    "label": selected_interval_label,
                    "seconds": interval_options[selected_interval_label]
                }
            }
            
            if "Reddit" in sources:
                job_payload["action"] = "build_reddit_kg"
            elif "Twitter" in sources:
                job_payload["action"] = "build_twitter_kg"  # (future support)
            else:
                job_payload["action"] = "build_reddit_kg_dummy"   # default fallback
            job_payload["summary_action"] = "summarize_graph"
            
            with st.spinner("🚀 Starting monitoring job..."):
                result = start_monitoring_job(job_payload, "autokg-processor")
            
            if result and result.get("success"):
                job_id = result.get("job_id")
                st.success(f"✅ Monitoring job started successfully!")
                st.info(f"Job ID: `{job_id}`")
                
                if "auto_active_jobs" not in st.session_state:
                    st.session_state.auto_active_jobs = []
                
                job_payload["status"] = "Running"
                job_payload["job_id"] = job_id
                st.session_state.auto_active_jobs.insert(0, job_payload)
                st.rerun()
            else:
                st.error("❌ Failed to start monitoring job. Please try again.")

    # --- Display Active Jobs Section ---
    st.markdown("---")
    st.header("Active Monitoring Jobs")

    if "auto_active_jobs" in st.session_state and st.session_state.auto_active_jobs:
        for idx, job in enumerate(st.session_state.auto_active_jobs):
            # Track button states outside container
            stop_clicked = False
            summary_clicked = False
            
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**Query:** {job['query']}")
                    st.caption(f"Subreddit: {job.get('subreddit', 'N/A')} | Sources: {job.get('sources', 'N/A')} | Limit: {job.get('limit', 'N/A')}")
                with col2:
                    st.markdown(f"**End Time:** {datetime.fromisoformat(job['schedule']['end_time_iso']).strftime('%b %d, %Y @ %H:%M')}")
                    st.caption(f"Interval: {job['interval']['label']}")
                with col3:
                    # Show status with appropriate state
                    status = job.get("status", "Running")
                    if status == "Stopped":
                        st.status(status, state="complete")
                    elif status == "Failed":
                        st.status(status, state="error")
                    else:
                        st.status(status, state="running")
                    
                    # Action buttons
                    col_stop, col_summary = st.columns(2)
                    
                    with col_stop:
                        # Disable stop button if job is already stopped
                        if status == "Stopped":
                            st.button("⏹️ Stopped", key=f"stop_{job['job_id']}", disabled=True, use_container_width=True)
                        else:
                            if st.button("⏹️ Stop", key=f"stop_{job['job_id']}", use_container_width=True):
                                stop_clicked = True
                    
                    with col_summary:
                        if st.button("📈 Summary", key=f"summary_{job['job_id']}", use_container_width=True):
                            summary_clicked = True
            
            # Handle actions OUTSIDE the container for full-width display
            if stop_clicked and job.get("status") != "Stopped":
                with st.spinner(f"Stopping job {job['job_id'][:8]}..."):
                    stop_result = stop_monitoring_job(job['job_id'], 'autokg-processor')
                
                if stop_result and stop_result.get("success"):
                    st.success("✅ Job stopped successfully!")
                    # Update job status in session state
                    for i, session_job in enumerate(st.session_state.auto_active_jobs):
                        if session_job['job_id'] == job['job_id']:
                            st.session_state.auto_active_jobs[i]['status'] = 'Stopped'
                            break
                    # Force UI refresh to clear spinner and update status
                    st.rerun()
                else:
                    st.error("❌ Failed to stop job.")
                    # Still refresh to clear the spinner state
                    st.rerun()
            
            # Display summary OUTSIDE container for full width
            if summary_clicked:
                try:
                    id_token = get_id_token()
                    decoded = jwt.decode(id_token, options={"verify_signature": False})
                    user_sub = decoded.get("sub")
                    
                    with st.spinner("Loading summary..."):
                        summary = load_summary(job['job_id'], user_sub)
                    
                    if summary:
                        st.markdown("---")
                        display_summary(summary)
                        st.markdown("---")
                    else:
                        st.info("📊 No summary available yet .")
                except Exception as e:
                    st.error(f"❌ Failed to load summary")
    else:
        st.info("No active monitoring jobs. Configure and start one above.")
