"""Authentication utilities for JWT, password hashing, and token generation."""

import os
import secrets
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, Request
from starlette.status import HTTP_401_UNAUTHORIZED

logger = logging.getLogger(__name__)

# Configuration constants (environment variables override defaults)
# SECURITY: In production, JWT_SECRET_KEY MUST be set via environment variable
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.environ.get("JWT_EXPIRE_HOURS", "168"))  # Default: 7 days
ACCESS_TOKEN_COOKIE_NAME = "access_token"

# Password hashing configuration
BCRYPT_ROUNDS = 12

# Security logger for auth events
security_logger = logging.getLogger("security")


class AuthenticationError(HTTPException):
    """Custom exception for authentication failures."""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=HTTP_401_UNAUTHORIZED, detail=detail)


class PasswordUtils:
    """Utilities for password hashing and verification."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt with 12 rounds."""
        salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, list[str]]:
        """
        Validate password meets security requirements.

        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []

        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")

        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        return len(errors) == 0, errors


class TokenUtils:
    """Utilities for secure token generation and validation."""

    @staticmethod
    def generate_secure_token() -> str:
        """Generate a cryptographically secure token for email verification/password reset."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def hash_token(token: str) -> str:
        """Hash a token for secure database storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def verify_token_hash(token: str, token_hash: str) -> bool:
        """Verify a token against its stored hash."""
        return hashlib.sha256(token.encode()).hexdigest() == token_hash


class JWTUtils:
    """Utilities for JWT token creation and validation."""

    @staticmethod
    def create_access_token(
        user_id: int, email: str, username: str | None = None
    ) -> str:
        """
        Create a JWT access token for a user.

        Args:
            user_id: User's database ID
            email: User's email address
            username: User's username (optional)

        Returns:
            JWT token string
        """
        now = datetime.now(timezone.utc)
        expire = now + timedelta(hours=JWT_EXPIRE_HOURS)

        payload = {
            "sub": str(user_id),  # Subject (user ID)
            "email": email,
            "username": username,
            "exp": expire,
            "iat": now,
        }

        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    @staticmethod
    def decode_access_token(token: str) -> dict[str, Any]:
        """
        Decode and validate a JWT access token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload dictionary

        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

            # Validate required fields
            if "sub" not in payload or "email" not in payload:
                raise AuthenticationError("Invalid token payload")

            return payload

        except JWTError as e:
            security_logger.warning(f"JWT decode error: {e}")
            raise AuthenticationError("Invalid or expired token")


class CookieUtils:
    """Utilities for secure cookie handling."""

    @staticmethod
    def get_cookie_settings() -> dict[str, Any]:
        """Get secure cookie settings for JWT storage."""
        return {
            "httponly": True,
            "secure": True,  # Should be True in production (HTTPS)
            "samesite": "strict",
            "max_age": JWT_EXPIRE_HOURS * 3600,  # Convert hours to seconds
        }


def get_token_from_cookie(request: Request) -> str:
    """
    Extract JWT token from httpOnly cookie.

    Args:
        request: FastAPI request object

    Returns:
        JWT token string

    Raises:
        AuthenticationError: If no token found in cookies
    """
    token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if not token:
        raise AuthenticationError("No authentication token found")
    return token


def get_current_user_from_token(request: Request) -> dict[str, Any]:
    """
    Get current user information from JWT token in cookie.

    This function is designed to be used as a FastAPI dependency.

    Args:
        request: FastAPI request object

    Returns:
        Dictionary containing user information from JWT payload

    Raises:
        AuthenticationError: If authentication fails
    """
    try:
        # Extract token from cookie
        token = get_token_from_cookie(request)

        # Decode and validate token
        payload = JWTUtils.decode_access_token(token)

        # Log successful authentication
        security_logger.info(f"Successful authentication for user: {payload['email']}")

        return {
            "id": int(payload["sub"]),
            "email": payload["email"],
            "username": payload.get("username"),
        }

    except AuthenticationError:
        # Log failed authentication attempt
        security_logger.warning(
            f"Failed authentication attempt from IP: {request.client.host if request.client else 'unknown'}"
        )
        raise


# FastAPI dependency for protecting routes
async def get_current_user(request: Request) -> dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user.

    Usage in route:
        @app.get("/protected")
        async def protected_route(current_user: dict = Depends(get_current_user)):
            return {"user": current_user}
    """
    return get_current_user_from_token(request)


async def get_current_user_optional(request: Request) -> dict[str, Any] | None:
    """
    FastAPI dependency to get current authenticated user, returning None if not authenticated.

    Usage in route:
        @app.post("/debates")
        async def create_debate(setup: DebateSetupRequest, current_user: dict | None = Depends(get_current_user_optional)):
            user_id = current_user["id"] if current_user else None
    """
    try:
        # Log cookie info for debugging
        cookies = dict(request.cookies)
        logger.info(f"get_current_user_optional: Request cookies: {list(cookies.keys())}")
        if 'access_token' in cookies:
            logger.info(f"get_current_user_optional: Found access_token cookie (length: {len(cookies['access_token'])})")

        user = get_current_user_from_token(request)
        logger.info(f"get_current_user_optional: Successfully authenticated user: {user}")
        return user
    except AuthenticationError as e:
        logger.info(f"get_current_user_optional: Authentication failed: {e}")
        return None


def log_security_event(
    event_type: str, details: dict[str, Any], request: Request | None = None
):
    """
    Log security-related events for monitoring and auditing.

    Args:
        event_type: Type of security event (e.g., "login_attempt", "registration")
        details: Dictionary of event details (avoid sensitive data)
        request: Optional FastAPI request for IP logging
    """
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **details,
    }

    if request and request.client:
        log_data["client_ip"] = request.client.host

    security_logger.info(f"Security event: {log_data}")
