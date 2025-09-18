"""Pydantic schemas for authentication endpoints."""

import re
from typing import Any
from pydantic import BaseModel, Field, field_validator, EmailStr


class UserRegistrationSchema(BaseModel):
    """Schema for user registration request."""

    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, max_length=128, description="User's password")

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, password: str) -> str:
        """Validate password meets security requirements."""
        errors = []

        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")

        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        if errors:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")

        return password

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123"
            }
        }


class EmailVerificationSchema(BaseModel):
    """Schema for email verification request."""

    token: str = Field(..., min_length=1, description="Email verification token")

    class Config:
        json_schema_extra = {
            "example": {
                "token": "abc123def456ghi789jkl012mno345pqr678"
            }
        }


class RegistrationCompletionSchema(BaseModel):
    """Schema for completing user registration with username."""

    user_id: int = Field(..., gt=0, description="User's database ID")
    username: str = Field(..., min_length=3, max_length=20, description="Desired username")

    @field_validator("username")
    @classmethod
    def validate_username(cls, username: str) -> str:
        """Validate username format and content."""
        # Username must be alphanumeric with optional underscores and hyphens
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")

        # Must start with a letter or number
        if not username[0].isalnum():
            raise ValueError("Username must start with a letter or number")

        # Must end with a letter or number
        if not username[-1].isalnum():
            raise ValueError("Username must end with a letter or number")

        # No consecutive special characters
        if '--' in username or '__' in username or '-_' in username or '_-' in username:
            raise ValueError("Username cannot contain consecutive special characters")

        return username.lower()  # Store usernames in lowercase

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "username": "awesome_user"
            }
        }


class LoginSchema(BaseModel):
    """Schema for user login request."""

    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=1, description="User's password")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123"
            }
        }


class ForgotPasswordSchema(BaseModel):
    """Schema for forgot password request."""

    email: EmailStr = Field(..., description="User's email address")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }


class PasswordResetSchema(BaseModel):
    """Schema for password reset request."""

    token: str = Field(..., min_length=1, description="Password reset token")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, password: str) -> str:
        """Validate password meets security requirements."""
        errors = []

        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")

        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        if errors:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")

        return password

    class Config:
        json_schema_extra = {
            "example": {
                "token": "abc123def456ghi789jkl012mno345pqr678",
                "new_password": "NewSecurePass456"
            }
        }


# Response schemas

class MessageResponse(BaseModel):
    """Generic message response schema."""

    message: str = Field(..., description="Response message")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Operation completed successfully"
            }
        }


class RegistrationResponse(MessageResponse):
    """Response schema for successful registration."""
    pass


class EmailVerificationResponse(BaseModel):
    """Response schema for successful email verification."""

    message: str = Field(..., description="Success message")
    user_id: int = Field(..., description="Verified user's ID")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Email verified successfully",
                "user_id": 123
            }
        }


class UserInfoSchema(BaseModel):
    """Schema for user information in responses."""

    id: int = Field(..., description="User's database ID")
    email: str = Field(..., description="User's email address")
    username: str | None = Field(None, description="User's username")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 123,
                "email": "user@example.com",
                "username": "awesome_user"
            }
        }


class LoginResponse(BaseModel):
    """Response schema for successful login."""

    message: str = Field(..., description="Success message")
    user: UserInfoSchema = Field(..., description="User information")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Login successful",
                "user": {
                    "id": 123,
                    "email": "user@example.com",
                    "username": "awesome_user"
                }
            }
        }


class CurrentUserResponse(BaseModel):
    """Response schema for current user information."""

    id: int = Field(..., description="User's database ID")
    email: str = Field(..., description="User's email address")
    username: str | None = Field(None, description="User's username")
    is_verified: bool = Field(..., description="Email verification status")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 123,
                "email": "user@example.com",
                "username": "awesome_user",
                "is_verified": True
            }
        }


# Error response schemas

class ErrorResponse(BaseModel):
    """Generic error response schema."""

    detail: str = Field(..., description="Error description")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "An error occurred"
            }
        }


class ValidationErrorResponse(BaseModel):
    """Validation error response schema."""

    detail: list[dict[str, Any]] = Field(..., description="Validation error details")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": [
                    {
                        "loc": ["body", "email"],
                        "msg": "field required",
                        "type": "missing"
                    }
                ]
            }
        }


class AuthErrorResponse(BaseModel):
    """Authentication error response schema."""

    detail: str = Field(..., description="Authentication error description")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid credentials"
            }
        }