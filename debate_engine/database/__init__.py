"""Database management module."""

def get_database_manager():
    """Lazy import of DatabaseManager to avoid dependency issues during testing."""
    from .database import DatabaseManager
    return DatabaseManager

# For backward compatibility, provide direct import when dependencies are available
try:
    from .database import DatabaseManager, get_database_path
    __all__ = ["DatabaseManager", "get_database_path"]
except ImportError:
    # During testing or when dependencies aren't available
    __all__ = ["get_database_manager"]