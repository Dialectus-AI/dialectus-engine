"""Tournament database operations."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional

from .models import (
    BracketData,
    MatchStatus,
    Tournament,
    TournamentJudge,
    TournamentMatch,
    TournamentParticipant,
    TournamentStatus,
    TournamentSummary,
    WeightClass,
)

logger = logging.getLogger(__name__)


class TournamentDatabaseManager:
    """Manages SQLite database operations for tournaments."""

    def __init__(self, db_path: str = "tournaments.db"):
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
            logger.error(f"Tournament database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def create_tournament(self, tournament: Tournament) -> int:
        """Create a new tournament and return its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO tournaments (
                    name, weight_class, format, word_limit, status, bracket_size,
                    current_round, total_rounds, tournament_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tournament.name,
                    tournament.weight_class.value,
                    tournament.format,
                    tournament.word_limit,
                    tournament.status.value,
                    tournament.bracket_size,
                    tournament.current_round,
                    tournament.total_rounds,
                    json.dumps(tournament.tournament_metadata or {}),
                ),
            )

            tournament_id = cursor.lastrowid
            if tournament_id is None:
                raise RuntimeError("Failed to get tournament ID from database")

            conn.commit()
            logger.info(f"Created tournament {tournament_id}: {tournament.name}")
            return tournament_id

    def get_tournament(self, tournament_id: int) -> Tournament | None:
        """Get tournament by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM tournaments WHERE id = ?", (tournament_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return Tournament(
                id=row["id"],
                name=row["name"],
                weight_class=WeightClass(row["weight_class"]),
                format=row["format"],
                word_limit=row["word_limit"],
                status=TournamentStatus(row["status"]),
                bracket_size=row["bracket_size"],
                current_round=row["current_round"],
                total_rounds=row["total_rounds"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                winner_model_id=row["winner_model_id"],
                tournament_metadata=json.loads(row["tournament_metadata"] or "{}"),
            )

    def list_tournaments(
        self, limit: int | None = None, offset: int = 0
    ) -> List[TournamentSummary]:
        """List tournaments with summary information."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT id, name, weight_class, status, bracket_size, current_round,
                       total_rounds, created_at, winner_model_id
                FROM tournaments
                ORDER BY created_at DESC
            """

            params = []
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                TournamentSummary(
                    id=row["id"],
                    name=row["name"],
                    weight_class=WeightClass(row["weight_class"]),
                    status=TournamentStatus(row["status"]),
                    bracket_size=row["bracket_size"],
                    current_round=row["current_round"],
                    total_rounds=row["total_rounds"],
                    created_at=row["created_at"],
                    winner_model_id=row["winner_model_id"],
                )
                for row in rows
            ]

    def update_tournament_status(
        self, tournament_id: int, status: TournamentStatus, **kwargs
    ) -> bool:
        """Update tournament status and optional fields."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build dynamic update query
            set_clauses = ["status = ?"]
            params: List[Any] = [status.value]

            if "started_at" in kwargs:
                set_clauses.append("started_at = ?")
                params.append(kwargs["started_at"])

            if "completed_at" in kwargs:
                set_clauses.append("completed_at = ?")
                params.append(kwargs["completed_at"])

            if "winner_model_id" in kwargs:
                set_clauses.append("winner_model_id = ?")
                params.append(kwargs["winner_model_id"])

            if "current_round" in kwargs:
                set_clauses.append("current_round = ?")
                params.append(kwargs["current_round"])

            params.append(tournament_id)

            query = f"""
                UPDATE tournaments
                SET {', '.join(set_clauses)}
                WHERE id = ?
            """

            cursor.execute(query, params)
            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.info(
                    f"Updated tournament {tournament_id} status to {status.value}"
                )

            return updated

    def delete_tournament(self, tournament_id: int) -> bool:
        """Delete tournament and all related data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM tournaments WHERE id = ?", (tournament_id,))
            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Deleted tournament {tournament_id}")

            return deleted

    def add_participant(self, participant: TournamentParticipant) -> int:
        """Add participant to tournament."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO tournament_participants (
                    tournament_id, model_id, model_name, seed_number
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    participant.tournament_id,
                    participant.model_id,
                    participant.model_name,
                    participant.seed_number,
                ),
            )

            participant_id = cursor.lastrowid
            if participant_id is None:
                raise RuntimeError("Failed to get participant ID from database")

            conn.commit()
            return participant_id

    def get_participants(self, tournament_id: int) -> List[TournamentParticipant]:
        """Get all participants for a tournament."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM tournament_participants
                WHERE tournament_id = ?
                ORDER BY seed_number
                """,
                (tournament_id,),
            )
            rows = cursor.fetchall()

            return [
                TournamentParticipant(
                    id=row["id"],
                    tournament_id=row["tournament_id"],
                    model_id=row["model_id"],
                    model_name=row["model_name"],
                    seed_number=row["seed_number"],
                    eliminated_in_round=row["eliminated_in_round"],
                )
                for row in rows
            ]

    def eliminate_participant(
        self, tournament_id: int, model_id: str, round_number: int
    ) -> bool:
        """Mark participant as eliminated in specified round."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE tournament_participants
                SET eliminated_in_round = ?
                WHERE tournament_id = ? AND model_id = ?
                """,
                (round_number, tournament_id, model_id),
            )

            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.info(
                    f"Eliminated participant {model_id} in round {round_number}"
                )

            return updated

    def add_match(self, match: TournamentMatch) -> int:
        """Add match to tournament."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO tournament_matches (
                    tournament_id, round_number, match_number, model_a_id,
                    model_b_id, topic, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match.tournament_id,
                    match.round_number,
                    match.match_number,
                    match.model_a_id,
                    match.model_b_id,
                    match.topic,
                    match.status.value,
                ),
            )

            match_id = cursor.lastrowid
            if match_id is None:
                raise RuntimeError("Failed to get match ID from database")

            conn.commit()
            return match_id

    def get_matches(
        self, tournament_id: int, round_number: int | None = None
    ) -> List[TournamentMatch]:
        """Get matches for tournament, optionally filtered by round."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if round_number is not None:
                cursor.execute(
                    """
                    SELECT * FROM tournament_matches
                    WHERE tournament_id = ? AND round_number = ?
                    ORDER BY match_number
                    """,
                    (tournament_id, round_number),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM tournament_matches
                    WHERE tournament_id = ?
                    ORDER BY round_number, match_number
                    """,
                    (tournament_id,),
                )

            rows = cursor.fetchall()

            return [
                TournamentMatch(
                    id=row["id"],
                    tournament_id=row["tournament_id"],
                    round_number=row["round_number"],
                    match_number=row["match_number"],
                    model_a_id=row["model_a_id"],
                    model_b_id=row["model_b_id"],
                    winner_model_id=row["winner_model_id"],
                    debate_id=row["debate_id"],
                    topic=row["topic"],
                    status=MatchStatus(row["status"]),
                )
                for row in rows
            ]

    def update_match(self, match_id: int, status: MatchStatus, **kwargs) -> bool:
        """Update match status and optional fields."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build dynamic update query
            set_clauses = ["status = ?"]
            params: List[Any] = [status.value]

            if "winner_model_id" in kwargs:
                set_clauses.append("winner_model_id = ?")
                params.append(kwargs["winner_model_id"])

            if "debate_id" in kwargs:
                set_clauses.append("debate_id = ?")
                params.append(kwargs["debate_id"])

            params.append(match_id)

            query = f"""
                UPDATE tournament_matches
                SET {', '.join(set_clauses)}
                WHERE id = ?
            """

            cursor.execute(query, params)
            updated = cursor.rowcount > 0
            conn.commit()

            if updated:
                logger.info(f"Updated match {match_id} status to {status.value}")

            return updated

    def add_judge(self, judge: TournamentJudge) -> int:
        """Add judge to tournament."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO tournament_judges (
                    tournament_id, judge_model_id, judge_provider
                ) VALUES (?, ?, ?)
                """,
                (
                    judge.tournament_id,
                    judge.judge_model_id,
                    judge.judge_provider,
                ),
            )

            judge_id = cursor.lastrowid
            if judge_id is None:
                raise RuntimeError("Failed to get judge ID from database")

            conn.commit()
            return judge_id

    def get_judges(self, tournament_id: int) -> List[TournamentJudge]:
        """Get all judges for a tournament."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM tournament_judges WHERE tournament_id = ?
                """,
                (tournament_id,),
            )
            rows = cursor.fetchall()

            return [
                TournamentJudge(
                    id=row["id"],
                    tournament_id=row["tournament_id"],
                    judge_model_id=row["judge_model_id"],
                    judge_provider=row["judge_provider"],
                )
                for row in rows
            ]

    def get_bracket_data(self, tournament_id: int) -> BracketData | None:
        """Get complete bracket data for visualization."""
        tournament = self.get_tournament(tournament_id)
        if not tournament:
            return None

        participants = self.get_participants(tournament_id)
        matches = self.get_matches(tournament_id)
        judges = self.get_judges(tournament_id)

        return BracketData(
            tournament=tournament,
            participants=participants,
            matches=matches,
            judges=judges,
        )

    def get_round_status(self, tournament_id: int, round_number: int) -> dict[str, Any]:
        """Get status of all matches in a round."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT status, COUNT(*) as count
                FROM tournament_matches
                WHERE tournament_id = ? AND round_number = ?
                GROUP BY status
                """,
                (tournament_id, round_number),
            )
            rows = cursor.fetchall()

            status_counts = {row["status"]: row["count"] for row in rows}
            total_matches = sum(status_counts.values())
            completed_matches = status_counts.get("completed", 0)
            pending_matches = status_counts.get("pending", 0)
            in_progress_matches = status_counts.get("in_progress", 0)

            return {
                "round_number": round_number,
                "total_matches": total_matches,
                "completed_matches": completed_matches,
                "pending_matches": pending_matches,
                "in_progress_matches": in_progress_matches,
                "all_completed": completed_matches == total_matches
                and total_matches > 0,
            }
