# In: ui/streamlit_pages/auto_verification.py
import streamlit as st
from datetime import datetime, timedelta, time
import jwt
import math  

from api_client import start_monitoring_job, stop_monitoring_job, load_summary
from api_client import display_detailed_summary as display_summary
from utils.auth import get_id_token

def show():
    """
    Displays the UI for configuring automated monitoring jobs with call constraints.
    """
    st.title("🤖 Bedrock agentcore Automated Content Monitoring")
    st.markdown(
        "Define a topic, select sources, and set a schedule to continuously monitor for new content. "
        "This configuration will be handed off to a backend agent to execute the job."
    )
    st.markdown("---")

    # --- DEFINE CONSTRAINTS ---
    MAX_AUTOMATED_CALLS = 20

    # --- TOP-LEVEL CONTROLS (OUTSIDE THE FORM) ---
    st.header("1. Define Monitoring Topic")
    
    query = st.text_input(
        label="**Query or Keywords**",
        placeholder="e.g., election security, climate change misinformation",
        help="The topic or keywords for the agent to monitor.",
        key="auto_query"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        subreddit = st.text_input(
            label="**Subreddit Name** (for Reddit monitoring)",
            placeholder="e.g., worldnews, politics, news",
            help="The subreddit to monitor (required for Reddit source)",
            key="auto_subreddit"
        )
    with col2:
        limit = st.slider(
            label="**Posts per Check**",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help="Number of posts to fetch each time the agent checks for new content",
            key="auto_limit"
        )

    sources = st.multiselect(
        label="**Social Media Feeds to Monitor**",
        options=["Reddit", "Twitter"],
        default=["Reddit"],
        help="Select one or more sources for the agent to monitor.",
        key="auto_sources"
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
    interval_seconds = interval_options[selected_interval_label]

    st.header("2. Set Monitoring Schedule")

    schedule_type = st.radio(
        "**Scheduling Method:**",
        ["Simple Duration", "Date Range (Calendar)"],
        horizontal=True,
        key="schedule_type_radio",
        help="Choose to select a specific date range or a simple duration from now."
    )
    
    # --- DYNAMIC VALIDATION LOGIC (RUNS ON EVERY INTERACTION) ---
    is_job_valid = True
    
    if schedule_type == "Simple Duration":
        st.subheader("⏱️ Set Job Duration")
        col_dur1, col_dur2 = st.columns(2)
        with col_dur1:
            duration_value = st.number_input("Run for the next...", min_value=1, value=20)
        with col_dur2:
            duration_unit = st.selectbox("Unit", ["Minutes", "Hours", "Days"])

        # Calculate total duration in seconds
        if duration_unit == "Minutes":
            total_duration_seconds = duration_value * 60
        elif duration_unit == "Hours":
            total_duration_seconds = duration_value * 3600
        else: # Days
            total_duration_seconds = duration_value * 86400
        
        # Calculate the number of automated calls
        # We use math.ceil to round up, as even a partial interval counts as a call.
        num_calls = math.ceil(total_duration_seconds / interval_seconds) if interval_seconds > 0 else 0

        # Display the warning/info message and set the validation flag
        if num_calls > MAX_AUTOMATED_CALLS:
            st.warning(
                f"⚠️ Your configuration would result in **{num_calls}** automated checks, "
                f"which exceeds the demo limit of **{MAX_AUTOMATED_CALLS}**. Please reduce the "
                "duration or increase the check interval."
            )
            is_job_valid = False
        else:
            st.info(
                f"✔️ This configuration will run approximately **{num_calls}** times. "
                f"This is within the demo limit of {MAX_AUTOMATED_CALLS}."
            )
            is_job_valid = True

    else: # Date Range (Calendar)
        st.subheader("🗓️ Select Start and End Dates (Disabled)")
        st.warning("🗓️ Date Range scheduling is currently disabled for this demo.")
        is_job_valid = False # Disable submission if calendar is selected
        col_cal1, col_cal2 = st.columns(2)
        with col_cal1:
            st.date_input("Start Date", value=datetime.now(), disabled=True)
            st.time_input("Start Time", value=time(9, 0), disabled=True)
        with col_cal2:
            st.date_input("End Date", value=datetime.now() + timedelta(days=7), disabled=True)
            st.time_input("End Time", value=time(17, 0), disabled=True)


    # --- Job Configuration Form ---
    with st.form(key="monitoring_job_form"):
        st.markdown("---")
        # The button's state is controlled by our validation flag
        submitted = st.form_submit_button(
            label="🚀 Prepare Monitoring Job",
            type="primary",
            use_container_width=True,
            disabled=not is_job_valid
        )

    # --- After Form Submission ---
    if submitted:
        # NOTE: The duration calculation is already done above. We just need to
        # recalculate the end time for the API payload.
        start_time_dt = datetime.utcnow()
        if duration_unit == "Minutes":
            end_time_dt = start_time_dt + timedelta(minutes=duration_value)
        elif duration_unit == "Hours":
            end_time_dt = start_time_dt + timedelta(hours=duration_value)
        else: # Days
            end_time_dt = start_time_dt + timedelta(days=duration_value)

        # Basic validation for other fields
        validation_error = False
        if not query:
            st.error("❌ Please enter a query.")
            validation_error = True
        if not sources:
            st.error("❌ Please select at least one source.")
            validation_error = True
        if "Reddit" in sources and not subreddit:
            st.error("❌ Please enter a subreddit name for Reddit monitoring.")
            validation_error = True
        
        if not validation_error:
            job_payload = {
                "query": query,
                "subreddit": subreddit,
                "limit": limit,
                "sources": ",".join(sources),
                "schedule": {
                    "start_time_iso": start_time_dt.isoformat(),
                    "end_time_iso": end_time_dt.isoformat(),
                },
                "interval": {
                    "label": selected_interval_label,
                    "seconds": interval_seconds
                }
            }
            
            if "Reddit" in sources:
                job_payload["action"] = "build_reddit_kg"
            else:
                job_payload["action"] = "build_reddit_kg_dummy"
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

    # --- Display Active Jobs Section (No changes needed here) ---
    st.markdown("---")
    st.header("Active Monitoring Jobs")

    if "auto_active_jobs" in st.session_state and st.session_state.auto_active_jobs:
        for idx, job in enumerate(st.session_state.auto_active_jobs):
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
                    status = job.get("status", "Running")
                    if status == "Stopped": st.status(status, state="complete")
                    elif status == "Failed": st.status(status, state="error")
                    else: st.status(status, state="running")
                    
                    col_stop, col_summary = st.columns(2)
                    with col_stop:
                        if status == "Stopped":
                            st.button("⏹️ Stopped", key=f"stop_{job['job_id']}", disabled=True, use_container_width=True)
                        else:
                            if st.button("⏹️ Stop", key=f"stop_{job['job_id']}", use_container_width=True):
                                stop_clicked = True
                    with col_summary:
                        if st.button("📈 Summary", key=f"summary_{job['job_id']}", use_container_width=True):
                            summary_clicked = True
            
            if stop_clicked and job.get("status") != "Stopped":
                with st.spinner(f"Stopping job {job['job_id'][:8]}..."):
                    stop_result = stop_monitoring_job(job['job_id'], 'autokg-processor')
                if stop_result and stop_result.get("success"):
                    st.success("✅ Job stopped successfully!")
                    st.session_state.auto_active_jobs[idx]['status'] = 'Stopped'
                    st.rerun()
                else:
                    st.error("❌ Failed to stop job.")
                    st.rerun()
            
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
                        st.info("📊 No summary available yet (Only after 2nd auto call).")
                except Exception as e:
                    st.error(f"❌ Failed to load summary")
    else:
        st.info("No active monitoring jobs. Configure and start one above.")
