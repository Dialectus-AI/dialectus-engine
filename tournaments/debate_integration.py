"""Tournament integration with debate system."""

import logging
from typing import Any, Callable, Dict, List, Optional

from .manager import TournamentManager

logger = logging.getLogger(__name__)


class TournamentDebateCallback:
    """Handles tournament-specific debate completion callbacks."""

    def __init__(self, tournament_manager: TournamentManager):
        self.tournament_manager = tournament_manager

    async def on_debate_completed(
        self, debate_id: str, judge_result: dict[str, Any] | None = None
    ) -> None:
        """Handle debate completion for tournament matches."""
        try:
            # Extract winner from judge results
            winner_id = self._extract_winner(judge_result)
            if not winner_id:
                logger.error(f"Could not determine winner for debate {debate_id}")
                return

            # Find tournament match associated with this debate
            tournament_id, match_id = await self._find_tournament_match(debate_id)
            if not tournament_id or not match_id:
                # Not a tournament match, ignore
                return

            # Update tournament with match result
            success = await self.tournament_manager.handle_match_completion(
                tournament_id, match_id, winner_id, debate_id
            )

            if success:
                logger.info(
                    f"Tournament match {match_id} completed: {winner_id} wins (debate {debate_id})"
                )
            else:
                logger.error(f"Failed to update tournament match {match_id}")

        except Exception as e:
            logger.error(f"Error handling tournament debate completion: {e}")

    def _extract_winner(self, judge_result: dict[str, Any] | None) -> str | None:
        """Extract winner ID from judge result."""
        if not judge_result:
            return None

        # Handle ensemble results
        if judge_result.get("type") == "ensemble":
            ensemble_summary = judge_result.get("ensemble_summary")
            if ensemble_summary:
                return ensemble_summary.get("final_winner_id")

        # Handle single judge results
        if "winner_id" in judge_result:
            return judge_result["winner_id"]

        # Handle legacy format
        if "judge_decisions" in judge_result and judge_result["judge_decisions"]:
            first_decision = judge_result["judge_decisions"][0]
            return first_decision.get("winner_id")

        return None

    async def _find_tournament_match(
        self, debate_id: str
    ) -> tuple[int | None, int | None]:
        """Find tournament and match ID for a debate ID."""
        try:
            # Search through all active tournaments to find match with this debate_id
            tournaments = self.tournament_manager.db.list_tournaments()

            for tournament_summary in tournaments:
                # Skip completed tournaments
                if tournament_summary.status.value == "completed":
                    continue

                # Get all matches for this tournament
                matches = self.tournament_manager.db.get_matches(tournament_summary.id)

                for match in matches:
                    if match.debate_id == debate_id:
                        return tournament_summary.id, match.id

            return None, None

        except Exception as e:
            logger.error(f"Error finding tournament match for debate {debate_id}: {e}")
            return None, None


class DebateManagerTournamentExtension:
    """Extension methods for DebateManager to support tournaments."""

    def __init__(self, debate_manager, tournament_manager: TournamentManager):
        self.debate_manager = debate_manager
        self.tournament_manager = tournament_manager
        self.tournament_callbacks: list[Callable] = []

        # Register tournament callback
        self.tournament_callback = TournamentDebateCallback(tournament_manager)
        self.add_tournament_callback(self.tournament_callback.on_debate_completed)

    def add_tournament_callback(self, callback_func: Callable) -> None:
        """Add callback function for tournament debate completion."""
        self.tournament_callbacks.append(callback_func)
        logger.info("Added tournament callback for debate completion")

    async def notify_tournament_completion(
        self, debate_id: str, judge_result: dict[str, Any] | None = None
    ) -> None:
        """Notify all tournament callbacks of debate completion."""
        for callback in self.tournament_callbacks:
            try:
                await callback(debate_id, judge_result)
            except Exception as e:
                logger.error(f"Tournament callback failed: {e}")

    async def create_tournament_debate(self, setup_data: dict[str, Any]) -> str | None:
        """Create a debate specifically for tournament matches."""
        try:
            # Convert tournament setup to DebateSetupRequest format
            from web.debate_setup_request import DebateSetupRequest

            setup = DebateSetupRequest(
                topic=setup_data["topic"],
                format=setup_data["format"],
                word_limit=setup_data["word_limit"],
                models=setup_data["models"],
                judge_models=setup_data["judge_models"],
            )

            # Create debate using existing debate manager
            debate_id = await self.debate_manager.create_debate(setup)

            # Store tournament match ID for callback lookup
            if debate_id and "tournament_match_id" in setup_data:
                # Update match with debate_id immediately
                match_id = setup_data["tournament_match_id"]
                # This will be handled by the tournament manager
                pass

            # Auto-start tournament debates
            if debate_id:
                await self.debate_manager.start_debate(debate_id)

            return debate_id

        except Exception as e:
            logger.error(f"Failed to create tournament debate: {e}")
            return None


def patch_debate_manager_for_tournaments(
    debate_manager, tournament_manager: TournamentManager
):
    """Patch existing DebateManager to support tournament callbacks."""

    # Add tournament callback functionality
    if not hasattr(debate_manager, "tournament_callbacks"):
        debate_manager.tournament_callbacks = []

    if not hasattr(debate_manager, "add_tournament_callback"):

        def add_tournament_callback(callback_func: Callable) -> None:
            debate_manager.tournament_callbacks.append(callback_func)
            logger.info("Added tournament callback to debate manager")

        debate_manager.add_tournament_callback = add_tournament_callback

    if not hasattr(debate_manager, "notify_tournament_completion"):

        async def notify_tournament_completion(
            debate_id: str, judge_result: dict[str, Any] | None = None
        ) -> None:
            for callback in debate_manager.tournament_callbacks:
                try:
                    await callback(debate_id, judge_result)
                except Exception as e:
                    logger.error(f"Tournament callback failed: {e}")

        debate_manager.notify_tournament_completion = notify_tournament_completion

    if not hasattr(debate_manager, "create_tournament_debate"):

        async def create_tournament_debate(setup_data: dict[str, Any]) -> str | None:
            try:
                from web.debate_setup_request import DebateSetupRequest

                setup = DebateSetupRequest(
                    topic=setup_data["topic"],
                    format=setup_data["format"],
                    word_limit=setup_data["word_limit"],
                    models=setup_data["models"],
                    judge_models=setup_data["judge_models"],
                )

                debate_id = await debate_manager.create_debate(setup)
                if debate_id:
                    await debate_manager.start_debate(debate_id)

                return debate_id

            except Exception as e:
                logger.error(f"Failed to create tournament debate: {e}")
                return None

        debate_manager.create_tournament_debate = create_tournament_debate

    # Register tournament callback
    tournament_callback = TournamentDebateCallback(tournament_manager)
    debate_manager.add_tournament_callback(tournament_callback.on_debate_completed)

    # Patch the _run_debate method to call tournament callbacks on completion
    original_run_debate = debate_manager._run_debate

    async def _run_debate_with_tournament_callbacks(debate_id: str) -> None:
        """Enhanced _run_debate that notifies tournament callbacks."""
        try:
            # Run original debate logic
            await original_run_debate(debate_id)

            # Get judge result and notify tournament callbacks
            debate_info = debate_manager.active_debates.get(debate_id)
            if debate_info and debate_info.get("status") == "completed":
                # Extract judge result from context
                context = debate_info.get("context")
                judge_result = None

                if context and hasattr(context, "judge_result"):
                    judge_result = context.judge_result
                elif context and hasattr(context, "judgement"):
                    judge_result = context.judgement

                # Notify tournament callbacks
                await debate_manager.notify_tournament_completion(
                    debate_id, judge_result
                )

        except Exception as e:
            logger.error(f"Error in tournament-enhanced debate execution: {e}")
            raise

    debate_manager._run_debate = _run_debate_with_tournament_callbacks

    logger.info("Successfully patched DebateManager for tournament support")
    return debate_manager
