# backend/common/cognito_auth.py
"""
Cognito JWT token validation and user extraction for Lambda functions.
Install dependencies: pip install python-jose requests
"""

import json
import os
import time
from typing import Dict, Optional, Tuple
from jose import jwt, JWTError
import requests

# Cognito Configuration
COGNITO_REGION =  os.getenv("COGNITO_REGION", "us-east-1")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USERPOOL_ID", "us-east-1_XXXXXXXXX")
COGNITO_APP_CLIENT_ID = os.getenv("COGNITO_APP_CLIENT_ID", "2qnmhh7vjc15baqiio0p3t95tn")

# Cache for JWKs (JSON Web Keys)
_jwks_cache = None
_jwks_cache_time = 0
JWKS_CACHE_TTL = 3600  # 1 hour


def _get_jwks() -> Dict:
    """Get JSON Web Keys from Cognito (with caching)."""
    global _jwks_cache, _jwks_cache_time

    current_time = time.time()
    if _jwks_cache and (current_time - _jwks_cache_time) < JWKS_CACHE_TTL:
        return _jwks_cache

    # Fetch fresh JWKs
    jwks_url = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/jwks.json"
    response = requests.get(jwks_url, timeout=10)
    response.raise_for_status()

    _jwks_cache = response.json()
    _jwks_cache_time = current_time

    return _jwks_cache


def validate_token(token: str, token_use: str = "access") -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Validate Cognito JWT token.

    Args:
        token: JWT token string
        token_use: Expected token use ("access" or "id")

    Returns:
        Tuple of (is_valid, claims_dict, error_message)
    """
    try:
        # Get the key ID from the token header
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            return False, None, "Token missing 'kid' in header"

        # Get JWKs and find the matching key
        jwks = _get_jwks()
        key = None
        for jwk in jwks.get("keys", []):
            if jwk.get("kid") == kid:
                key = jwk
                break

        if not key:
            return False, None, f"Public key not found for kid: {kid}"

        # Verify and decode the token
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=COGNITO_APP_CLIENT_ID,
            issuer=f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}",
        )

        # Validate token_use claim
        if claims.get("token_use") != token_use:
            return False, None, f"Token use mismatch. Expected '{token_use}', got '{claims.get('token_use')}'"

        # Check expiration
        if claims.get("exp", 0) < time.time():
            return False, None, "Token has expired"

        return True, claims, None

    except JWTError as e:
        return False, None, f"JWT validation error: {str(e)}"
    except Exception as e:
        return False, None, f"Token validation error: {str(e)}"


def extract_user_info(claims: Dict) -> Dict[str, str]:
    """
    Extract user information from validated token claims.

    Args:
        claims: Validated JWT claims

    Returns:
        Dictionary with user information
    """
    return {
        "username": claims.get("username") or claims.get("cognito:username", "unknown"),
        "user_id": claims.get("sub", "unknown"),
        "email": claims.get("email", ""),
        "groups": claims.get("cognito:groups", []),
    }


def get_s3_user_prefix(username: str) -> str:
    """
    Get S3 prefix for user's files.

    Args:
        username: Cognito username

    Returns:
        S3 prefix path for the user
    """
    # Sanitize username for S3 path
    safe_username = "".join(c if c.isalnum() or c in "-_" else "_" for c in username)
    return f"users/{safe_username}/"


def authorize_request(event: Dict) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
    """
    Authorize Lambda API Gateway request using Cognito token.

    Args:
        event: Lambda event from API Gateway

    Returns:
        Tuple of (is_authorized, user_info, error_response)
        If authorized: (True, user_info_dict, None)
        If not authorized: (False, None, error_response_dict)
    """
    # Extract token from Authorization header
    headers = event.get("headers", {})

    # API Gateway might lowercase headers
    auth_header = headers.get("Authorization") or headers.get("authorization", "")

    if not auth_header:
        return False, None, {
            "statusCode": 401,
            "body": json.dumps({"error": "Missing Authorization header"})
        }

    # Extract Bearer token
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False, None, {
            "statusCode": 401,
            "body": json.dumps({"error": "Invalid Authorization header format. Use 'Bearer <token>'"})
        }

    token = parts[1]

    # Validate token
    is_valid, claims, error = validate_token(token, token_use="access")

    if not is_valid:
        return False, None, {
            "statusCode": 401,
            "body": json.dumps({"error": f"Invalid token: {error}"})
        }

    # Extract user info
    user_info = extract_user_info(claims)

    return True, user_info, None


def lambda_handler_with_auth(handler_func):
    """
    Decorator to add Cognito authentication to Lambda handlers.

    Usage:
        @lambda_handler_with_auth
        def lambda_handler(event, context, user_info):
            # user_info contains validated user data
            username = user_info["username"]
            # ... your handler code
    """
    def wrapper(event, context):
        # Check if path is a health check endpoint (no auth required)
        path = event.get("path", "")
        if path == "/health" or path.endswith("/health"):
            return handler_func(event, context, user_info=None)

        # Authorize request
        is_authorized, user_info, error_response = authorize_request(event)

        if not is_authorized:
            return error_response

        # Call the actual handler with user_info
        return handler_func(event, context, user_info=user_info)

    return wrapper
