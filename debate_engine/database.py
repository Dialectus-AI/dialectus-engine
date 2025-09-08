"""SQLite database manager for debate transcripts."""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, TypedDict
from contextlib import contextmanager

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
    participants: Dict[str, ParticipantInfo]
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
    metadata: Dict[str, Any]


class FullTranscriptData(TypedDict):
    """Complete transcript data including metadata and messages."""
    metadata: Dict[str, Any]  # Contains the same fields as DebateMetadata but nested
    messages: List[MessageData]
    scores: Dict[str, float]
    context_metadata: Dict[str, Any]


class DatabaseManager:
    """Manages SQLite database connections and schema for debate transcripts."""
    
    def __init__(self, db_path: str = "debates.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create debates table for metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS debates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    format TEXT NOT NULL,
                    participants TEXT NOT NULL,  -- JSON string
                    final_phase TEXT NOT NULL,
                    total_rounds INTEGER NOT NULL,
                    saved_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    total_debate_time_ms INTEGER NOT NULL,  -- Total debate duration in milliseconds
                    scores TEXT,  -- JSON string
                    context_metadata TEXT,  -- JSON string
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id INTEGER NOT NULL,
                    speaker_id TEXT NOT NULL,
                    position TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    round_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    metadata TEXT,  -- JSON string
                    FOREIGN KEY (debate_id) REFERENCES debates (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_debate_id 
                ON messages (debate_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_round_phase 
                ON messages (debate_id, round_number, phase)
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
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
    
    def save_debate(self, debate_data: Dict[str, Any]) -> int:
        """Save a complete debate to the database and return the debate ID."""
        import json
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert debate metadata
            metadata = debate_data["metadata"]
            cursor.execute("""
                INSERT INTO debates (
                    topic, format, participants, final_phase, total_rounds,
                    saved_at, message_count, word_count, total_debate_time_ms, scores, context_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata["topic"],
                metadata["format"],
                json.dumps(metadata["participants"]),
                metadata["final_phase"],
                metadata["total_rounds"],
                metadata["saved_at"],
                metadata["message_count"],
                metadata["word_count"],
                metadata["total_debate_time_ms"],
                json.dumps(debate_data["scores"]),
                json.dumps(debate_data["context_metadata"])
            ))
            
            debate_id = cursor.lastrowid
            if debate_id is None:
                raise RuntimeError("Failed to get debate ID from database")
            
            # Insert all messages
            for message in debate_data["messages"]:
                cursor.execute("""
                    INSERT INTO messages (
                        debate_id, speaker_id, position, phase, round_number,
                        content, timestamp, word_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    debate_id,
                    message["speaker_id"],
                    message["position"],
                    message["phase"],
                    message["round_number"],
                    message["content"],
                    message["timestamp"],
                    message["word_count"],
                    json.dumps(message["metadata"])
                ))
            
            conn.commit()
            logger.info(f"Saved debate {debate_id} with {len(debate_data['messages'])} messages")
            return debate_id
    
    def load_debate(self, debate_id: int) -> Optional[FullTranscriptData]:
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
            cursor.execute("""
                SELECT * FROM messages WHERE debate_id = ? 
                ORDER BY round_number, id
            """, (debate_id,))
            message_rows = cursor.fetchall()
            
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
                    "total_debate_time_ms": debate_row["total_debate_time_ms"]
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
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    }
                    for row in message_rows
                ],
                "scores": json.loads(debate_row["scores"]) if debate_row["scores"] else {},
                "context_metadata": json.loads(debate_row["context_metadata"]) if debate_row["context_metadata"] else {}
            }
            
            return debate_data
    
    def list_debates(self, limit: Optional[int] = None, offset: int = 0) -> List[DebateMetadata]:
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
                    "created_at": row["created_at"]
                }
                for row in rows
            ]

    def list_debates_with_metadata(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """List debates with metadata including context_metadata for judge model info."""
        import json
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT id, topic, format, participants, final_phase, total_rounds,
                       saved_at, message_count, word_count, total_debate_time_ms, 
                       created_at, context_metadata
                FROM debates 
                ORDER BY created_at DESC
            """
            
            params = []
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                debate_data = {
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
                    "created_at": row["created_at"]
                }
                
                # Extract judge model from context_metadata
                if row["context_metadata"]:
                    try:
                        context_metadata = json.loads(row["context_metadata"])
                        
                        # Try to get judge model from various possible locations
                        judge_model = None
                        if "judge_decision" in context_metadata and "metadata" in context_metadata["judge_decision"]:
                            judge_model = context_metadata["judge_decision"]["metadata"].get("judge_model")
                        elif "judging_config" in context_metadata:
                            judge_model = context_metadata["judging_config"].get("judge_model")
                        elif "judge_model" in context_metadata:
                            judge_model = context_metadata["judge_model"]
                        
                        debate_data["judge_model"] = judge_model or "none"
                    except (json.JSONDecodeError, KeyError):
                        debate_data["judge_model"] = "unknown"
                else:
                    debate_data["judge_model"] = "none"
                
                result.append(debate_data)
            
            return result
    
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
    
    def update_debate_metadata(self, debate_id: int, context_metadata: Dict[str, Any]) -> None:
        """Update the context_metadata for an existing debate."""
        import json
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE debates SET context_metadata = ? WHERE id = ?",
                (json.dumps(context_metadata), debate_id)
            )
            conn.commit()
            
            if cursor.rowcount == 0:
                raise RuntimeError(f"No debate found with ID {debate_id}")
    
    def get_debate_count(self) -> int:
        """Get total number of debates."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM debates")
            return cursor.fetchone()[0]