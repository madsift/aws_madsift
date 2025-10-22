import base64
import hashlib
import json
import secrets
import urllib.parse
import time
import requests
import streamlit as st
import os

# ---------- CONFIG ----------
COGNITO_DOMAIN = os.getenv("COGNITO_DOMAIN")
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID")
COGNITO_REDIRECT_URI = os.getenv("COGNITO_REDIRECT_URI")

PRIVATE_BUCKET = os.getenv("PRIVATE_BUCKET")
PRIVATE_PREFIX = os.getenv("PRIVATE_PREFIX")
COGNITO_SCOPES = "openid profile email"

AUTHORIZE_URL = f"{COGNITO_DOMAIN}/oauth2/authorize"
TOKEN_URL = f"{COGNITO_DOMAIN}/oauth2/token"
LOGOUT_URL = f"{COGNITO_DOMAIN}/logout"


# ---------- PKCE HELPERS ----------
def _generate_code_verifier() -> str:
    raw = secrets.token_bytes(32)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def decode_jwt_no_verify(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode()))
    except Exception:
        return {}


# ---------- LOGIN ----------
def login() -> bool:
    params_raw = st.query_params or {}
    params = {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in params_raw.items()}

    # Handle post-logout redirect - clear everything and restart
    if "post_logout" in params:
        st.query_params.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        time.sleep(0.2)
        st.rerun()
        return False

    # Already authenticated?
    if "tokens" in st.session_state:
        # Double check we don't have stale query params
        if "code" in params or "error" in params:
            st.query_params.clear()
            st.rerun()
        return True

    # If there's an error from Cognito
    if "error" in params:
        st.error(f"❌ Authentication error: {params.get('error_description', params['error'])}")
        st.query_params.clear()
        time.sleep(1)
        st.rerun()
        return False

    # If returning from Cognito with ?code=
    if "code" in params:
        return _handle_callback(params)

    # ---------- NOT AUTHENTICATED - START NEW FLOW ----------
    _start_pkce_flow()
    st.stop()  # Stop execution here
    return False


def _handle_callback(params: dict) -> bool:
    """Handle the OAuth callback with authorization code"""
    code = params["code"]
    state = params.get("state", "")
    
    # Get verifier from session
    verifier = st.session_state.get("code_verifier")
    
    # If no verifier in session, try to extract from state
    if not verifier and state:
        try:
            padded = state + "=" * (-len(state) % 4)
            decoded = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
            verifier = decoded.get("v")
        except Exception:
            pass
    
    # If still no verifier, this is a stale/reused code
    if not verifier:
        st.warning("⚠️ Session expired or invalid. Starting fresh login...")
        st.query_params.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        time.sleep(0.5)
        st.rerun()
        return False
    
    # Exchange code for tokens
    payload = {
        "grant_type": "authorization_code",
        "client_id": COGNITO_CLIENT_ID,
        "code": code,
        "redirect_uri": COGNITO_REDIRECT_URI,
        "code_verifier": verifier,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    
    try:
        resp = requests.post(TOKEN_URL, data=payload, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            # Success! Store tokens and clear auth state
            st.session_state["tokens"] = resp.json()
            st.session_state.pop("code_verifier", None)
            st.session_state.pop("state_nonce", None)
            st.query_params.clear()
            time.sleep(0.2)
            st.rerun()
            return True
        else:
            # Token exchange failed
            st.error(f"❌ Login failed: {resp.text}")
            st.query_params.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            time.sleep(1)
            st.rerun()
            return False
            
    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        st.query_params.clear()
        time.sleep(1)
        st.rerun()
        return False


def _start_pkce_flow():
    """Start a fresh PKCE authentication flow"""
    # Clear ALL query params and session state
    st.query_params.clear()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    
    # Generate fresh PKCE parameters
    verifier = _generate_code_verifier()
    challenge = _code_challenge(verifier)
    nonce = secrets.token_urlsafe(16)
    
    # Store in state parameter
    state_payload = {"n": nonce, "v": verifier, "t": int(time.time())}
    encoded_state = base64.urlsafe_b64encode(json.dumps(state_payload).encode()).decode().rstrip("=")
    
    # Store verifier in session
    st.session_state["code_verifier"] = verifier
    st.session_state["state_nonce"] = nonce
    
    # Build authorization URL
    auth_url = (
        f"{AUTHORIZE_URL}"
        f"?response_type=code"
        f"&client_id={urllib.parse.quote(COGNITO_CLIENT_ID)}"
        f"&redirect_uri={urllib.parse.quote(COGNITO_REDIRECT_URI)}"
        f"&scope={urllib.parse.quote(COGNITO_SCOPES)}"
        f"&state={urllib.parse.quote(encoded_state)}"
        f"&code_challenge_method=S256"
        f"&code_challenge={urllib.parse.quote(challenge)}"
    )
    
    # Redirect to Cognito
    st.markdown(
        f"""
        <meta http-equiv="refresh" content="0;url={auth_url}">
        <script>
            window.location.replace("{auth_url}");
        </script>
        <p>Redirecting to login...</p>
        """,
        unsafe_allow_html=True,
    )


# ---------- LOGOUT ----------
def logout():
    """Logout and redirect to Cognito logout endpoint"""
    # Build logout URL
    post_logout_uri = urllib.parse.quote(COGNITO_REDIRECT_URI.rstrip("/") + "/?post_logout=1")
    logout_url = f"{LOGOUT_URL}?client_id={COGNITO_CLIENT_ID}&logout_uri={post_logout_uri}"
    
    # Clear session state
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    
    # Redirect to Cognito logout
    st.markdown(
        f"""
        <meta http-equiv="refresh" content="0;url={logout_url}">
        <script>
            // Clear browser storage
            try {{
                localStorage.clear();
                sessionStorage.clear();
            }} catch(e) {{}}
            // Redirect
            window.location.replace("{logout_url}");
        </script>
        <p>Logging out...</p>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ---------- UTILS ----------
def get_access_token() -> str | None:
    return st.session_state.get("tokens", {}).get("access_token")


def get_id_token() -> str | None:
    return st.session_state.get("tokens", {}).get("id_token")
