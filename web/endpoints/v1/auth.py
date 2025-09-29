"""Authentication endpoints."""

import logging

from fastapi import HTTPException, Request, Response, Depends, APIRouter
from starlette.status import HTTP_409_CONFLICT, HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED

from config.settings import get_default_config
from web.auth_schemas import (
    UserRegistrationSchema, RegistrationResponse, LoginSchema, LoginResponse,
    RegistrationCompletionSchema, MessageResponse, CurrentUserResponse,
    ForgotPasswordSchema, PasswordResetSchema, EmailVerificationSchema,
    EmailVerificationResponse, ErrorResponse, AuthErrorResponse, UserInfoSchema
)
from web.auth_utils import (
    PasswordUtils, JWTUtils, TokenUtils, CookieUtils, get_current_user,
    log_security_event, AuthenticationError
)
from web.auth_database import AuthDatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth")

# Initialize auth database manager
auth_db = AuthDatabaseManager()


@router.post("/register", response_model=RegistrationResponse,
          responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def register(request: Request, user_data: UserRegistrationSchema):
    """Register a new user account."""
    try:
        # Check if email already exists
        if auth_db.email_exists(user_data.email):
            log_security_event("registration_failed", {"email": user_data.email, "reason": "email_exists"}, request)
            raise HTTPException(status_code=HTTP_409_CONFLICT, detail="Email already registered")

        # Hash password
        password_hash = PasswordUtils.hash_password(user_data.password)

        # Get config to check development mode
        config = get_default_config()
        auth_config = getattr(config, 'auth', None)
        is_development = getattr(auth_config, 'development_mode', False) if auth_config else False

        # Create user account
        user_id = auth_db.create_user(user_data.email, password_hash, is_verified=is_development)

        if is_development:
            # Development mode: skip email verification, auto-verify user
            logger.info(f"Development mode: Auto-verified user {user_data.email}")
            log_security_event("registration_success", {"email": user_data.email, "user_id": user_id, "auto_verified": True}, request)
            return RegistrationResponse(message=f"Account created and verified for {user_data.email}")
        else:
            # Production mode: require email verification
            # Generate email verification token
            token = TokenUtils.generate_secure_token()
            token_hash = TokenUtils.hash_token(token)
            auth_db.create_email_verification(user_id, token_hash)

            # TODO: Send verification email with token
            # For now, log the token for development
            logger.info(f"Email verification token for {user_data.email}: {token}")

            log_security_event("registration_success", {"email": user_data.email, "user_id": user_id}, request)
            return RegistrationResponse(message=f"Verification email sent to {user_data.email}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        log_security_event("registration_error", {"email": user_data.email, "error": str(e)}, request)
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/verify", response_model=EmailVerificationResponse,
          responses={400: {"model": ErrorResponse}})
async def verify_email(verification_data: EmailVerificationSchema):
    """Verify user email address."""
    try:
        # Hash the provided token
        token_hash = TokenUtils.hash_token(verification_data.token)

        # Verify token and get user ID
        user_id = auth_db.verify_email_token(token_hash)
        if not user_id:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Invalid or expired verification token")

        # Mark user as verified
        auth_db.update_user_verification(user_id, True)

        log_security_event("email_verification_success", {"user_id": user_id})
        return EmailVerificationResponse(message="Email verified successfully", user_id=user_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email verification error: {e}")
        raise HTTPException(status_code=500, detail="Email verification failed")


@router.post("/complete-registration", response_model=MessageResponse,
          responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
async def complete_registration(completion_data: RegistrationCompletionSchema):
    """Complete user registration by setting username."""
    try:
        # Verify user exists and is verified
        user = auth_db.get_user_by_id(completion_data.user_id)
        if not user:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Invalid user ID")

        if not user["is_verified"]:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Email not verified")

        # Check if username already exists
        if auth_db.username_exists(completion_data.username):
            raise HTTPException(status_code=HTTP_409_CONFLICT, detail="Username already taken")

        # Update username
        auth_db.update_user_username(completion_data.user_id, completion_data.username)

        log_security_event("registration_completed", {"user_id": completion_data.user_id, "username": completion_data.username})
        return MessageResponse(message="Registration completed successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration completion error: {e}")
        raise HTTPException(status_code=500, detail="Registration completion failed")


@router.post("/login", response_model=LoginResponse,
          responses={401: {"model": AuthErrorResponse}})
async def login(request: Request, response: Response, login_data: LoginSchema):
    """Login user and set authentication cookie."""
    try:
        # Get user by email
        user = auth_db.get_user_by_email(login_data.email)
        if not user:
            log_security_event("login_failed", {"email": login_data.email, "reason": "user_not_found"}, request)
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        # Verify password
        if not PasswordUtils.verify_password(login_data.password, user["password_hash"]):
            log_security_event("login_failed", {"email": login_data.email, "reason": "invalid_password"}, request)
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        # Check if user is verified
        if not user["is_verified"]:
            log_security_event("login_failed", {"email": login_data.email, "reason": "unverified"}, request)
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Email not verified")

        # Generate JWT token
        jwt_token = JWTUtils.create_access_token(user["id"], user["email"], user["username"])

        # Set secure cookie
        cookie_settings = CookieUtils.get_cookie_settings()
        response.set_cookie("access_token", jwt_token, **cookie_settings)

        log_security_event("login_success", {"email": login_data.email, "user_id": user["id"]}, request)
        return LoginResponse(
            message="Login successful",
            user=UserInfoSchema(
                id=user["id"],
                email=user["email"],
                username=user["username"]
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        log_security_event("login_error", {"email": login_data.email, "error": str(e)}, request)
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/logout", response_model=MessageResponse)
async def logout(response: Response):
    """Logout user by clearing authentication cookie."""
    try:
        # Clear the access token cookie
        response.delete_cookie("access_token")
        return MessageResponse(message="Logged out successfully")

    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@router.get("/me", response_model=CurrentUserResponse,
         responses={401: {"model": AuthErrorResponse}})
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user information."""
    try:
        # Get full user data from database
        user = auth_db.get_user_by_id(current_user["id"])
        if not user:
            raise AuthenticationError("User not found")

        return CurrentUserResponse(
            id=user["id"],
            email=user["email"],
            username=user["username"],
            is_verified=user["is_verified"]
        )

    except AuthenticationError:
        raise
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user information")


@router.post("/forgot-password", response_model=MessageResponse,
          responses={400: {"model": ErrorResponse}})
async def forgot_password(request: Request, forgot_data: ForgotPasswordSchema):
    """Request password reset for user account."""
    try:
        # Check if user exists (but always return success message for security)
        user = auth_db.get_user_by_email(forgot_data.email)

        if user:
            # Generate secure reset token
            token = TokenUtils.generate_secure_token()
            token_hash = TokenUtils.hash_token(token)

            # Create password reset record (expires in 1 hour)
            auth_db.create_password_reset(user["id"], token_hash, expires_hours=1)

            # TODO: Send password reset email with token
            # For now, log the token for development
            logger.info(f"Password reset token for {forgot_data.email}: {token}")

            log_security_event("password_reset_requested", {"email": forgot_data.email, "user_id": user["id"]}, request)

        # Always return success message to prevent email enumeration
        log_security_event("password_reset_request", {"email": forgot_data.email}, request)
        return MessageResponse(message="If email exists, reset instructions have been sent")

    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        log_security_event("password_reset_error", {"email": forgot_data.email, "error": str(e)}, request)
        raise HTTPException(status_code=500, detail="Password reset request failed")


@router.post("/reset-password", response_model=MessageResponse,
          responses={400: {"model": ErrorResponse}})
async def reset_password(request: Request, reset_data: PasswordResetSchema):
    """Reset user password using reset token."""
    try:
        # Hash the provided token
        token_hash = TokenUtils.hash_token(reset_data.token)

        # Verify token and get user ID
        user_id = auth_db.verify_password_reset_token(token_hash)
        if not user_id:
            log_security_event("password_reset_failed", {"reason": "invalid_token"}, request)
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token")

        # Hash new password
        new_password_hash = PasswordUtils.hash_password(reset_data.new_password)

        # Update user password
        if not auth_db.update_user_password(user_id, new_password_hash):
            raise HTTPException(status_code=500, detail="Failed to update password")

        # Clean up any remaining password reset tokens for this user
        # Note: The verify_password_reset_token already deleted the used token

        log_security_event("password_reset_success", {"user_id": user_id}, request)
        return MessageResponse(message="Password reset successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        log_security_event("password_reset_error", {"error": str(e)}, request)
        raise HTTPException(status_code=500, detail="Password reset failed")