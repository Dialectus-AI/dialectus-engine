"""Schema manager for organizing and executing database schema files."""

import logging
from pathlib import Path
from typing import Any
import sqlite3

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages database schema creation from organized SQL files."""

    def __init__(self, schema_dir: Path | None = None):
        """Initialize schema manager with schema directory path."""
        if schema_dir is None:
            schema_dir = Path(__file__).parent

        self.schema_dir = schema_dir
        self.tables_dir = schema_dir / "tables"

        # Define the order of table creation to handle dependencies
        self.table_creation_order = [
            "debates.sql",
            "messages.sql",
            "judge_decisions.sql",
            "ensemble_summary.sql",
            "criterion_scores.sql",
            "tournaments.sql",
            "tournament_participants.sql",
            "tournament_matches.sql",
            "tournament_judges.sql",
            "indexes.sql"  # Create indexes last
        ]

    def load_schema_file(self, filename: str) -> str:
        """Load SQL content from a schema file."""
        file_path = self.tables_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")

        return file_path.read_text(encoding="utf-8")

    def execute_schema_file(self, cursor: sqlite3.Cursor, filename: str) -> None:
        """Execute SQL from a single schema file."""
        try:
            sql_content = self.load_schema_file(filename)

            # Handle files that may contain multiple statements
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

            for statement in statements:
                if statement:
                    cursor.execute(statement)

            logger.debug(f"Executed schema file: {filename}")

        except Exception as e:
            logger.error(f"Failed to execute schema file {filename}: {e}")
            raise

    def initialize_database_schema(self, cursor: sqlite3.Cursor) -> None:
        """Initialize complete database schema by executing all schema files in order."""
        logger.info("Initializing database schema from files")

        try:
            for filename in self.table_creation_order:
                self.execute_schema_file(cursor, filename)

            logger.info("Database schema initialization completed successfully")

        except Exception as e:
            logger.error(f"Database schema initialization failed: {e}")
            raise

    def get_available_schema_files(self) -> list[str]:
        """Get list of available schema files in the tables directory."""
        if not self.tables_dir.exists():
            return []

        return [f.name for f in self.tables_dir.glob("*.sql")]

    def validate_schema_files(self) -> bool:
        """Validate that all expected schema files exist."""
        missing_files = []

        for filename in self.table_creation_order:
            file_path = self.tables_dir / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            logger.error(f"Missing schema files: {missing_files}")
            return False

        logger.info("All schema files validated successfully")
        return True