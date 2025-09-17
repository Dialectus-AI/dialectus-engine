"""SQLite database manager for debate transcripts."""

import sqlite3
import logging
from pathlib import Path
from typing import Any, TypedDict
from contextlib import contextmanager
from judges.base import JudgeDecision
from .schema import SchemaManager

logger = logging.getLogger(__name__)


class ParticipantInfo(TypedDict):
    """Information about a debate participant."""

    name: str
    personality: str


class DebateMetadata(TypedDict):
    """Metadata for a debate transcript."""

    id: int
    topic: str
    format: str
    participants: dict[str, ParticipantInfo]
    final_phase: str
    total_rounds: int
    saved_at: str
    message_count: int
    word_count: int
    total_debate_time_ms: int
    created_at: str


class MessageData(TypedDict):
    """Data structure for a single debate message."""

    speaker_id: str
    position: str
    phase: str
    round_number: int
    content: str
    timestamp: str
    word_count: int
    metadata: dict[str, Any]


class FullTranscriptData(TypedDict):
    """Complete transcript data including metadata and messages."""

    metadata: dict[str, Any]  # Contains the same fields as DebateMetadata but nested
    messages: list[MessageData]
    scores: dict[str, float]
    context_metadata: dict[str, Any]


class DatabaseManager:
    """Manages SQLite database connections and schema for debate transcripts."""

    def __init__(self, db_path: str = "debates.db"):
        self.db_path = Path(db_path)
        self.schema_manager = SchemaManager()
        self._init_database()

    def _init_database(self):
        """Initialize the database with required tables using schema manager."""
        # Validate that all required schema files exist
        if not self.schema_manager.validate_schema_files():
            raise RuntimeError("Database schema validation failed - missing schema files")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Use schema manager to initialize all tables and indexes
            self.schema_manager.initialize_database_schema(cursor)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

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

    def save_debate(self, debate_data: dict[str, Any]) -> int:
        """Save a complete debate to the database and return the debate ID."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert debate metadata
            metadata = debate_data["metadata"]
            cursor.execute(
                """
                INSERT INTO debates (
                    topic, format, participants, final_phase, total_rounds,
                    saved_at, message_count, word_count, total_debate_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata["topic"],
                    metadata["format"],
                    json.dumps(metadata["participants"]),
                    metadata["final_phase"],
                    metadata["total_rounds"],
                    metadata["saved_at"],
                    metadata["message_count"],
                    metadata["word_count"],
                    metadata["total_debate_time_ms"],
                ),
            )

            debate_id = cursor.lastrowid
            if debate_id is None:
                raise RuntimeError("Failed to get debate ID from database")

            # Insert all messages
            for message in debate_data["messages"]:
                cursor.execute(
                    """
                    INSERT INTO messages (
                        debate_id, speaker_id, position, phase, round_number,
                        content, timestamp, word_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        debate_id,
                        message["speaker_id"],
                        message["position"],
                        message["phase"],
                        message["round_number"],
                        message["content"],
                        message["timestamp"],
                        message["word_count"],
                        json.dumps(message["metadata"]),
                    ),
                )

            conn.commit()
            logger.info(
                f"Saved debate {debate_id} with {len(debate_data['messages'])} messages"
            )
            return debate_id

    def load_debate(self, debate_id: int) -> FullTranscriptData | None:
        """Load a complete debate by ID."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Load debate metadata
            cursor.execute("SELECT * FROM debates WHERE id = ?", (debate_id,))
            debate_row = cursor.fetchone()

            if not debate_row:
                return None

            # Load messages
            cursor.execute(
                """
                SELECT * FROM messages WHERE debate_id = ? 
                ORDER BY round_number, id
            """,
                (debate_id,),
            )
            message_rows = cursor.fetchall()

            # Load judge decision if available
            judge_decision = self.load_judge_decision(debate_id)

            # Reconstruct debate data
            debate_data: FullTranscriptData = {
                "metadata": {
                    "topic": debate_row["topic"],
                    "format": debate_row["format"],
                    "participants": json.loads(debate_row["participants"]),
                    "final_phase": debate_row["final_phase"],
                    "total_rounds": debate_row["total_rounds"],
                    "saved_at": debate_row["saved_at"],
                    "message_count": debate_row["message_count"],
                    "word_count": debate_row["word_count"],
                    "total_debate_time_ms": debate_row["total_debate_time_ms"],
                },
                "messages": [
                    {
                        "speaker_id": row["speaker_id"],
                        "position": row["position"],
                        "phase": row["phase"],
                        "round_number": row["round_number"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "word_count": row["word_count"],
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                    }
                    for row in message_rows
                ],
                "scores": {},  # Legacy field, now empty
                "context_metadata": (
                    {"judge_decision": judge_decision} if judge_decision else {}
                ),
            }

            return debate_data

    def list_debates(
        self, limit: int | None = None, offset: int = 0
    ) -> list[DebateMetadata]:
        """List debates with metadata only (no messages)."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT id, topic, format, participants, final_phase, total_rounds,
                       saved_at, message_count, word_count, total_debate_time_ms, created_at
                FROM debates 
                ORDER BY created_at DESC
            """

            params = []
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "topic": row["topic"],
                    "format": row["format"],
                    "participants": json.loads(row["participants"]),
                    "final_phase": row["final_phase"],
                    "total_rounds": row["total_rounds"],
                    "saved_at": row["saved_at"],
                    "message_count": row["message_count"],
                    "word_count": row["word_count"],
                    "total_debate_time_ms": row["total_debate_time_ms"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def list_debates_with_metadata(
        self, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List debates with metadata including judge model info via SQL joins."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    d.id, d.topic, d.format, d.participants, d.final_phase, d.total_rounds,
                    d.saved_at, d.message_count, d.word_count, d.total_debate_time_ms,
                    d.created_at,
                    jd.judge_model
                FROM debates d
                LEFT JOIN judge_decisions jd ON d.id = jd.debate_id
                ORDER BY d.created_at DESC
            """

            params = []
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "topic": row["topic"],
                    "format": row["format"],
                    "participants": json.loads(row["participants"]),
                    "final_phase": row["final_phase"],
                    "total_rounds": row["total_rounds"],
                    "saved_at": row["saved_at"],
                    "message_count": row["message_count"],
                    "word_count": row["word_count"],
                    "total_debate_time_ms": row["total_debate_time_ms"],
                    "created_at": row["created_at"],
                    "judge_model": row["judge_model"] or "none",
                }
                for row in rows
            ]

    def delete_debate(self, debate_id: int) -> bool:
        """Delete a debate and all its messages."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM debates WHERE id = ?", (debate_id,))
            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Deleted debate {debate_id}")

            return deleted

    def save_judge_decision(self, debate_id: int, judge_decision: JudgeDecision) -> int:
        """Save an individual judge decision and return the judge_decision_id."""
        if debate_id is None:
            raise ValueError("debate_id cannot be None when saving judge decision")
        if not isinstance(debate_id, int) or debate_id <= 0:
            raise ValueError(f"debate_id must be a positive integer, got: {debate_id}")

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert judge decision using strongly-typed properties
            cursor.execute(
                """
                INSERT INTO judge_decisions (
                    debate_id, judge_model, judge_provider, winner_id, winner_margin,
                    overall_feedback, reasoning, generation_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    debate_id,
                    judge_decision.judge_model,
                    judge_decision.judge_provider,
                    judge_decision.winner_id,
                    judge_decision.winner_margin,
                    judge_decision.overall_feedback,
                    judge_decision.reasoning,
                    judge_decision.generation_time_ms,
                ),
            )

            judge_decision_id = cursor.lastrowid
            if judge_decision_id is None:
                raise RuntimeError("Failed to get judge_decision_id from database")

            # Insert criterion scores using strongly-typed CriterionScore objects
            for score in judge_decision.criterion_scores:
                cursor.execute(
                    """
                    INSERT INTO criterion_scores (
                        judge_decision_id, criterion, participant_id, score, feedback
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        judge_decision_id,
                        score.criterion.value,  # Extract enum value
                        score.participant_id,
                        score.score,
                        score.feedback,
                    ),
                )

            conn.commit()
            logger.info(
                f"Saved individual judge decision {judge_decision_id} for debate {debate_id}"
            )
            return judge_decision_id

    def save_ensemble_summary(
        self, debate_id: int, ensemble_summary: dict[str, Any]
    ) -> int:
        """Save ensemble summary to the ensemble_summary table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO ensemble_summary (
                    debate_id, final_winner_id, final_margin, ensemble_method, num_judges,
                    consensus_level, summary_reasoning, summary_feedback, participating_judge_decision_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    debate_id,
                    ensemble_summary["final_winner_id"],
                    ensemble_summary["final_margin"],
                    ensemble_summary.get("ensemble_method", "majority"),
                    ensemble_summary["num_judges"],
                    ensemble_summary.get("consensus_level"),
                    ensemble_summary.get("summary_reasoning"),
                    ensemble_summary.get("summary_feedback"),
                    ensemble_summary.get(
                        "participating_judge_decision_ids"
                    ),  # JSON string
                ),
            )

            ensemble_id = cursor.lastrowid
            if ensemble_id is None:
                raise RuntimeError("Failed to get ensemble_id from database")

            conn.commit()
            logger.info(f"Saved ensemble summary {ensemble_id} for debate {debate_id}")
            return ensemble_id

    def load_judge_decisions(self, debate_id: int) -> list[dict[str, Any]]:
        """Load all individual judge decisions for a debate with their criterion scores."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM judge_decisions
                WHERE debate_id = ?
                ORDER BY created_at
            """,
                (debate_id,),
            )
            decision_rows = cursor.fetchall()

            decisions = []
            for decision_row in decision_rows:
                # Load criterion scores for this judge decision
                cursor.execute(
                    """
                    SELECT * FROM criterion_scores WHERE judge_decision_id = ?
                    ORDER BY criterion, participant_id
                """,
                    (decision_row["id"],),
                )
                score_rows = cursor.fetchall()

                decision = {
                    "winner_id": decision_row["winner_id"],
                    "winner_margin": decision_row["winner_margin"],
                    "overall_feedback": decision_row["overall_feedback"],
                    "reasoning": decision_row["reasoning"],
                    "criterion_scores": [
                        {
                            "criterion": row["criterion"],
                            "participant_id": row["participant_id"],
                            "score": row["score"],
                            "feedback": row["feedback"],
                        }
                        for row in score_rows
                    ],
                    "metadata": {
                        "judge_model": decision_row["judge_model"],
                        "judge_provider": decision_row["judge_provider"],
                        "generation_time_ms": decision_row["generation_time_ms"],
                    },
                }
                decisions.append(decision)

            return decisions

    def load_ensemble_summary(self, debate_id: int) -> dict[str, Any] | None:
        """Load ensemble summary for a debate."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM ensemble_summary WHERE debate_id = ?
            """,
                (debate_id,),
            )
            ensemble_row = cursor.fetchone()

            if not ensemble_row:
                return None

            return {
                "final_winner_id": ensemble_row["final_winner_id"],
                "final_margin": ensemble_row["final_margin"],
                "ensemble_method": ensemble_row["ensemble_method"],
                "num_judges": ensemble_row["num_judges"],
                "consensus_level": ensemble_row["consensus_level"],
                "summary_reasoning": ensemble_row["summary_reasoning"],
                "summary_feedback": ensemble_row["summary_feedback"],
                "participating_judge_decision_ids": ensemble_row[
                    "participating_judge_decision_ids"
                ],
            }

    def load_judge_decision(self, debate_id: int) -> dict[str, Any] | None:
        """Load the first/primary judge decision for a debate."""
        decisions = self.load_judge_decisions(debate_id)
        return decisions[0] if decisions else None

    def get_debate_count(self) -> int:
        """Get total number of debates."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM debates")
            return cursor.fetchone()[0]
