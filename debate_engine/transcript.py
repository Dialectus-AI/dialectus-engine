"""Transcript management for debates using SQLite."""

from datetime import datetime
from typing import Any, TypedDict
import logging

from .models import DebateContext
from .database import DatabaseManager
from .database.database import FullTranscriptData

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





class TranscriptManager:
    """Manages saving and loading of debate transcripts in SQLite database."""

    def __init__(self, db_path: str = "debates.db"):
        self.db_manager = DatabaseManager(db_path)

    def save_transcript(self, context: DebateContext, total_debate_time_ms: int, user_id: int | None = None) -> int:
        """Save a debate transcript to the database and return the debate ID."""
        transcript_data = self._context_to_dict(context, total_debate_time_ms)
        try:
            debate_id = self.db_manager.save_debate(transcript_data, user_id=user_id)
            logger.info(f"Saved transcript to database with ID {debate_id}" + (f" for user {user_id}" if user_id else ""))
            return debate_id
        except Exception as e:
            logger.error(f"Failed to save transcript to database: {e}")
            raise

    def _context_to_dict(
        self, context: DebateContext, total_debate_time_ms: int
    ) -> dict[str, Any]:
        """Convert DebateContext to dictionary for JSON serialization."""
        return {
            "metadata": {
                "topic": context.topic,
                "format": context.metadata.get("format", "unknown"),
                "participants": {
                    pid: {"name": config.name, "personality": config.personality}
                    for pid, config in context.participants.items()
                },
                "final_phase": context.current_phase.value,
                "total_rounds": context.current_round,
                "saved_at": datetime.now().isoformat(),
                "message_count": len(context.messages),
                "word_count": sum(len(msg.content.split()) for msg in context.messages),
                "total_debate_time_ms": total_debate_time_ms,
            },
            "messages": [
                {
                    "speaker_id": msg.speaker_id,
                    "position": msg.position.value,
                    "phase": msg.phase.value,
                    "round_number": msg.round_number,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "word_count": len(msg.content.split()),
                    "metadata": msg.metadata,
                    "cost": msg.cost,
                    "generation_id": msg.generation_id,
                    "cost_queried_at": msg.cost_queried_at.isoformat() if msg.cost_queried_at else None,
                }
                for msg in context.messages
            ],
            "scores": context.scores,
            "context_metadata": context.metadata,
        }

    def load_transcript(self, debate_id: int) -> FullTranscriptData | None:
        """Load a debate transcript by ID."""
        try:
            result = self.db_manager.load_debate(debate_id)
            return result  # DatabaseManager already returns properly typed data
        except Exception as e:
            logger.error(f"Failed to load transcript from database: {e}")
            return None

    def list_transcripts(
        self, limit: int | None = None, offset: int = 0
    ) -> list[DebateMetadata]:
        """List transcripts with pagination support."""
        try:
            result = self.db_manager.list_debates(limit=limit, offset=offset)
            return result  # DatabaseManager already returns properly typed data
        except Exception as e:
            logger.error(f"Failed to list debates from database: {e}")
            return []

    def get_transcript_summary(self, debate_id: int) -> DebateMetadata | None:
        """Get summary information about a transcript without loading all messages."""
        debates = self.list_transcripts()
        for debate in debates:
            if debate.get("id") == debate_id:
                return debate
        return None

    def format_transcript_for_judging(self, context: DebateContext) -> str:
        """Format transcript as plain text for AI judging."""
        lines = [
            f"DEBATE TRANSCRIPT",
            f"Topic: {context.topic}",
            f"Format: {context.metadata.get('format', 'unknown')}",
            f"Participants: {', '.join(f'{pid} ({config.name})' for pid, config in context.participants.items())}",
            f"Total Messages: {len(context.messages)}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=" * 80,
            "",
        ]

        current_round = 0
        current_phase = None

        for msg in context.messages:
            # Add round/phase headers when they change
            if msg.round_number != current_round or msg.phase != current_phase:
                if current_round > 0:  # Add separator between rounds
                    lines.append("")
                lines.append(f"ROUND {msg.round_number} - {msg.phase.value.upper()}")
                lines.append("-" * 40)
                current_round = msg.round_number
                current_phase = msg.phase

            # Format message
            speaker_header = f"{msg.speaker_id.upper()} ({msg.position.value.upper()})"
            lines.extend(
                [
                    f"[{speaker_header}]",
                    msg.content.strip(),
                    f"    Words: {len(msg.content.split())} | Time: {msg.timestamp.strftime('%H:%M:%S')}",
                    "",
                ]
            )

        lines.extend(
            ["=" * 80, f"END OF TRANSCRIPT - {len(context.messages)} total messages"]
        )

        return "\n".join(lines)

    def delete_transcript(self, debate_id: int) -> bool:
        """Delete a debate transcript by ID."""
        try:
            return self.db_manager.delete_debate(debate_id)
        except Exception as e:
            logger.error(f"Failed to delete transcript {debate_id}: {e}")
            return False

    def get_debate_count(self, user_id: int | None = None) -> int:
        """Get the total number of debates stored, optionally filtered by user."""
        try:
            return self.db_manager.get_debate_count(user_id=user_id)
        except Exception as e:
            logger.error(f"Failed to get debate count: {e}")
            return 0

    def search_transcripts_by_topic(self, topic_search: str) -> list[DebateMetadata]:
        """Search for transcripts containing the topic search term."""
        debates = self.list_transcripts()
        return [
            debate
            for debate in debates
            if topic_search.lower() in debate.get("topic", "").lower()
        ]

    def get_transcripts_by_format(self, format_name: str) -> list[DebateMetadata]:
        """Get all transcripts for a specific debate format."""
        debates = self.list_transcripts()
        return [
            debate
            for debate in debates
            if debate.get("format", "").lower() == format_name.lower()
        ]
