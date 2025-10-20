import streamlit as st
from utils.auth import login, logout, get_id_token, decode_jwt_no_verify
from api_client import health
import logging

logging.basicConfig(level=logging.DEBUG)
st.write("Debug: loaded initial UI")  # simple marker

def show_navigation_sidebar(user_info):
    """
    Draws the main navigation sidebar with login info and returns the selected page.
    This is the primary router for the UI.
    """
    # Login status at top of sidebar
    with st.sidebar:
        st.success("✅ Logged in")
        
        # User info
        with st.expander("👤 User Profile", expanded=False):
            username = user_info.get("cognito:username", "N/A")
            email = user_info.get("email", "N/A")
            st.write(f"**Username:** {username}")
            st.write(f"**Email:** {email}")
        
        # Logout button
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            logout()
            st.stop()
        
        st.markdown("---")
    
    st.sidebar.title("🔍 Navigation")
    
    # This radio button controls which page is displayed
    page = st.sidebar.radio(
        "Select Mode",
        [
            "🏠 Dashboard",
            "🔗 Reddit KG Builder",
            "💬 Interactive Chat",
            "🔍 Auto Verification", 
            "📊 Monitoring & Alerts",
        ],
        key="main_navigation"
    )
    
    st.sidebar.markdown("---")
    
    # You can add other global sidebar elements here if needed
    
    return page


def show_dashboard():
    """
    Displays the main dashboard/landing page.
    Includes a health check to verify the backend connection.
    """
    st.title("🔍 Rumour Verification Agentic Framework")
    st.markdown("*Proactive Intelligence Platform (API-Driven UI)*")
    st.markdown("---")
    
    st.subheader("🏥 Backend System Health")
    st.info("This status is fetched in real-time from the backend's `/health` endpoint.")
    
    with st.spinner("Checking backend health..."):
        health_status = health()
    
    if health_status and health_status.get("status") == "healthy":
        st.success("✅ Backend API is healthy and connected.")
    else:
        st.error("❌ Could not connect to the backend API. Please ensure the SAM local API is running.")
    
    # Display the full health check response for debugging
    st.json(health_status)


def show_placeholder_page(title: str):
    """A generic placeholder for pages that are not yet refactored."""
    st.title(f"🚧 {title}")
    st.warning("This page has not been fully refactored to use the API backend yet.")
    st.info("The navigation link exists, but the logic needs to be migrated.")


def main():
    """
    Main Streamlit application entry point.
    This function orchestrates the page routing.
    """
    st.set_page_config(
        page_title="MadSift",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check authentication first
    is_authenticated = login()
    
    if not is_authenticated:
        st.stop()
        return
    
    # Get user info from ID token
    id_token = get_id_token()
    user_info = decode_jwt_no_verify(id_token) if id_token else {}
    
    # Get the user's page selection from the sidebar
    page = show_navigation_sidebar(user_info)
    
    # --- Page Routing Logic ---
    # Based on the selection, import and run the 'show()' function
    # from the corresponding module in the 'streamlit_pages' directory.
    
    if page == "🏠 Dashboard":
        show_dashboard()
    
    elif page == "🔗 Reddit KG Builder":
        from streamlit_pages import reddit_kg_builder
        reddit_kg_builder.show()
        
    elif page == "💬 Interactive Chat":
        from streamlit_pages import interactive_chat
        interactive_chat.show()
    
    #elif page == "🔍 Reactive Verification":
    #    from streamlit_pages import reactive_verification
    #    reactive_verification.show() 
    elif page == "🔍 Auto Verification":
        from streamlit_pages import auto_verification
        auto_verification.show() 
    elif page == "📊 Monitoring & Alerts":
        show_placeholder_page("Monitoring & Alerts")
        
    else:
        # Fallback for any unknown page
        show_dashboard()


if __name__ == "__main__":
    main()
