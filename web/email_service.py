"""Email service for sending verification and password reset emails."""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(template_name: str) -> str:
    """Load email template from file.

    Args:
        template_name: Name of template file (e.g., 'email_verification.html')

    Returns:
        Template content as string
    """
    template_path = TEMPLATE_DIR / template_name
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Email template not found: {template_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading template {template_name}: {e}")
        raise


def render_template(template_content: str, variables: dict[str, str]) -> str:
    """Render template by replacing {{variable}} placeholders.

    Args:
        template_content: Template string with {{variable}} placeholders
        variables: Dictionary mapping variable names to values

    Returns:
        Rendered template string
    """
    result = template_content
    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, value)
    return result


class EmailService:
    """Service for sending emails via SMTP."""

    def __init__(self, smtp_server: str, smtp_port: int, smtp_user: str,
                 smtp_password: str, from_email: str, from_name: str,
                 frontend_url: str, enabled: bool = True, use_tls: bool = True):
        """Initialize email service with SMTP configuration."""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.from_name = from_name
        self.frontend_url = frontend_url
        self.enabled = enabled
        self.use_tls = use_tls

    def send_email(self, to_email: str, subject: str, html_body: str,
                   text_body: Optional[str] = None) -> bool:
        """Send an email via SMTP.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text fallback (optional)

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning(f"Email service disabled. Would send to {to_email}: {subject}")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email

            # Add plain text and HTML parts
            if text_body:
                part1 = MIMEText(text_body, 'plain')
                msg.attach(part1)

            part2 = MIMEText(html_body, 'html')
            msg.attach(part2)

            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending email to {to_email}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email to {to_email}: {e}")
            return False

    def send_verification_email(self, to_email: str, token: str) -> bool:
        """Send email verification link.

        Args:
            to_email: Recipient email address
            token: Verification token

        Returns:
            True if email sent successfully, False otherwise
        """
        verification_url = f"{self.frontend_url}/#/verify?token={token}"

        variables = {"verification_url": verification_url}

        html_template = load_template("email_verification.html")
        text_template = load_template("email_verification.txt")

        html_body = render_template(html_template, variables)
        text_body = render_template(text_template, variables)

        subject = "Verify Your Dialectus Account"
        return self.send_email(to_email, subject, html_body, text_body)

    def send_password_reset_email(self, to_email: str, token: str) -> bool:
        """Send password reset link.

        Args:
            to_email: Recipient email address
            token: Password reset token

        Returns:
            True if email sent successfully, False otherwise
        """
        reset_url = f"{self.frontend_url}/#/reset-password?token={token}"

        variables = {"reset_url": reset_url}

        html_template = load_template("password_reset.html")
        text_template = load_template("password_reset.txt")

        html_body = render_template(html_template, variables)
        text_body = render_template(text_template, variables)

        subject = "Reset Your Dialectus Password"
        return self.send_email(to_email, subject, html_body, text_body)


# Global email service instance (initialized by API startup)
_email_service: Optional[EmailService] = None


def initialize_email_service(config) -> None:
    """Initialize the global email service from configuration."""
    global _email_service

    # Get email config
    auth_config = getattr(config, 'auth', None)
    if not auth_config:
        logger.warning("No auth configuration found, email service disabled")
        _email_service = None
        return

    email_config = getattr(auth_config, 'email', None)
    if not email_config:
        logger.warning("No email configuration found, email service disabled")
        _email_service = None
        return

    # Check if email is enabled
    enabled = getattr(email_config, 'enabled', False)
    if not enabled:
        logger.info("Email service explicitly disabled in configuration")
        _email_service = None
        return

    # Extract SMTP settings
    smtp_server = getattr(email_config, 'smtp_server', '')
    smtp_port = getattr(email_config, 'smtp_port', 587)
    smtp_user = getattr(email_config, 'smtp_user', '')
    smtp_password = getattr(email_config, 'smtp_password', '')
    from_email = getattr(email_config, 'from_email', '')
    from_name = getattr(email_config, 'from_name', 'Dialectus AI')
    frontend_url = getattr(email_config, 'frontend_url', 'http://localhost:5173')
    use_tls = getattr(email_config, 'use_tls', True)

    # Validate required fields
    if not all([smtp_server, smtp_user, smtp_password, from_email, frontend_url]):
        logger.warning("Incomplete email configuration, email service disabled")
        _email_service = None
        return

    _email_service = EmailService(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_email=from_email,
        from_name=from_name,
        frontend_url=frontend_url,
        enabled=enabled,
        use_tls=use_tls
    )

    logger.info(f"Email service initialized: {smtp_server}:{smtp_port} (frontend: {frontend_url})")


def get_email_service() -> Optional[EmailService]:
    """Get the global email service instance."""
    return _email_service