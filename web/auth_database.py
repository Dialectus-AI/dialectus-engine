"""Database operations for authentication system."""

import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import TypedDict
from contextlib import contextmanager
from pathlib import Path
from debate_engine.database.database import get_database_path

logger = logging.getLogger(__name__)


class UserData(TypedDict):
    """Type definition for user data."""
    id: int
    email: str
    username: str | None
    password_hash: str
    is_verified: bool
    created_at: str
    updated_at: str


class AuthDatabaseManager:
    """Database manager specifically for authentication operations."""

    def __init__(self, db_path: str | None = None):
        """Initialize with database path."""
        if db_path is None:
            self.db_path = get_database_path()
        else:
            self.db_path = Path(db_path)

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def create_user(self, email: str, password_hash: str, is_verified: bool = False) -> int:
        """
        Create a new user account.

        Args:
            email: User's email address
            password_hash: Bcrypt hashed password
            is_verified: Whether the user is already verified (for development mode)

        Returns:
            User ID of created user

        Raises:
            sqlite3.IntegrityError: If email already exists
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO users (email, password_hash, is_verified, created_at, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
                (email, password_hash, is_verified),
            )

            user_id = cursor.lastrowid
            if user_id is None:
                raise RuntimeError("Failed to get user ID from database")

            conn.commit()
            logger.info(f"Created user account for email: {email} (verified: {is_verified})")
            return user_id

    def get_user_by_email(self, email: str) -> UserData | None:
        """
        Get user by email address.

        Args:
            email: User's email address

        Returns:
            UserData dict or None if user not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "id": row["id"],
                "email": row["email"],
                "username": row["username"],
                "password_hash": row["password_hash"],
                "is_verified": bool(row["is_verified"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def get_user_by_id(self, user_id: int) -> UserData | None:
        """
        Get user by ID.

        Args:
            user_id: User's database ID

        Returns:
            UserData dict or None if user not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "id": row["id"],
                "email": row["email"],
                "username": row["username"],
                "password_hash": row["password_hash"],
                "is_verified": bool(row["is_verified"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def update_user_verification(self, user_id: int, is_verified: bool = True) -> bool:
        """
        Update user verification status.

        Args:
            user_id: User's database ID
            is_verified: Verification status

        Returns:
            True if user was updated, False if user not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE users
                SET is_verified = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (is_verified, user_id),
            )

            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.info(f"Updated verification status for user {user_id}: {is_verified}")

            return updated

    def update_user_username(self, user_id: int, username: str) -> bool:
        """
        Update user's username.

        Args:
            user_id: User's database ID
            username: New username

        Returns:
            True if user was updated, False if user not found

        Raises:
            sqlite3.IntegrityError: If username already exists
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE users
                SET username = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (username, user_id),
            )

            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.info(f"Updated username for user {user_id}: {username}")

            return updated

    def update_user_password(self, user_id: int, password_hash: str) -> bool:
        """
        Update user's password hash.

        Args:
            user_id: User's database ID
            password_hash: New bcrypt hashed password

        Returns:
            True if user was updated, False if user not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE users
                SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (password_hash, user_id),
            )

            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.info(f"Updated password for user {user_id}")

            return updated

    def create_email_verification(self, user_id: int, token_hash: str, expires_hours: int = 24) -> int:
        """
        Create an email verification token.

        Args:
            user_id: User's database ID
            token_hash: SHA-256 hash of the verification token
            expires_hours: Token expiration time in hours

        Returns:
            Verification record ID
        """
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO email_verifications (user_id, token, expires_at)
                VALUES (?, ?, ?)
            """,
                (user_id, token_hash, expires_at),
            )

            verification_id = cursor.lastrowid
            if verification_id is None:
                raise RuntimeError("Failed to get verification ID from database")

            conn.commit()
            logger.info(f"Created email verification for user {user_id}")
            return verification_id

    def verify_email_token(self, token_hash: str) -> int | None:
        """
        Verify email verification token and return user ID if valid.

        Args:
            token_hash: SHA-256 hash of the verification token

        Returns:
            User ID if token is valid and not expired, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if token exists and is not expired
            cursor.execute(
                """
                SELECT user_id FROM email_verifications
                WHERE token = ? AND expires_at > CURRENT_TIMESTAMP
            """,
                (token_hash,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            user_id = row["user_id"]

            # Delete the used token
            cursor.execute(
                "DELETE FROM email_verifications WHERE token = ?",
                (token_hash,)
            )

            conn.commit()
            logger.info(f"Email verification successful for user {user_id}")
            return user_id

    def create_password_reset(self, user_id: int, token_hash: str, expires_hours: int = 1) -> int:
        """
        Create a password reset token.

        Args:
            user_id: User's database ID
            token_hash: SHA-256 hash of the reset token
            expires_hours: Token expiration time in hours

        Returns:
            Reset record ID
        """
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO password_resets (user_id, token, expires_at)
                VALUES (?, ?, ?)
            """,
                (user_id, token_hash, expires_at),
            )

            reset_id = cursor.lastrowid
            if reset_id is None:
                raise RuntimeError("Failed to get reset ID from database")

            conn.commit()
            logger.info(f"Created password reset for user {user_id}")
            return reset_id

    def verify_password_reset_token(self, token_hash: str) -> int | None:
        """
        Verify password reset token and return user ID if valid.

        Args:
            token_hash: SHA-256 hash of the reset token

        Returns:
            User ID if token is valid and not expired, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if token exists and is not expired
            cursor.execute(
                """
                SELECT user_id FROM password_resets
                WHERE token = ? AND expires_at > CURRENT_TIMESTAMP
            """,
                (token_hash,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            user_id = row["user_id"]

            # Delete the used token
            cursor.execute(
                "DELETE FROM password_resets WHERE token = ?",
                (token_hash,)
            )

            conn.commit()
            logger.info(f"Password reset token verified for user {user_id}")
            return user_id

    def cleanup_expired_tokens(self):
        """Clean up expired verification and reset tokens."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Clean up expired email verifications
            cursor.execute(
                "DELETE FROM email_verifications WHERE expires_at <= CURRENT_TIMESTAMP"
            )
            email_deleted = cursor.rowcount

            # Clean up expired password resets
            cursor.execute(
                "DELETE FROM password_resets WHERE expires_at <= CURRENT_TIMESTAMP"
            )
            reset_deleted = cursor.rowcount

            conn.commit()

            if email_deleted > 0 or reset_deleted > 0:
                logger.info(f"Cleaned up {email_deleted} expired email verifications and {reset_deleted} expired password resets")

    def username_exists(self, username: str) -> bool:
        """
        Check if username already exists.

        Args:
            username: Username to check

        Returns:
            True if username exists, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (username,)
            )
            return cursor.fetchone() is not None

    def email_exists(self, email: str) -> bool:
        """
        Check if email already exists.

        Args:
            email: Email to check

        Returns:
            True if email exists, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT 1 FROM users WHERE email = ?",
                (email,)
            )
            return cursor.fetchone() is not None