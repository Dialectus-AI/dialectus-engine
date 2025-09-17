"""Tournament management and bracket generation."""

import logging
import math
import random
from datetime import datetime
from typing import Any

from models.manager import ModelManager
from .database import TournamentDatabaseManager
from .models import (
    BracketData,
    MatchStatus,
    Tournament,
    TournamentCreateRequest,
    TournamentJudge,
    TournamentMatch,
    TournamentParticipant,
    TournamentStatus,
    WeightClass,
)

logger = logging.getLogger(__name__)

# Cost ranges for weight classes (per 1K tokens)
WEIGHT_CLASS_RANGES = {
    WeightClass.FREE: (0.0, 0.0),  # Only truly free models
    WeightClass.BUDGET: (0.0, 0.001),  # $0 - $0.001/1K tokens
    WeightClass.ECONOMY: (0.001, 0.005),  # $0.001 - $0.005/1K tokens
    WeightClass.PREMIUM: (0.005, 0.02),  # $0.005 - $0.02/1K tokens
    WeightClass.ELITE: (0.02, float("inf")),  # $0.02+/1K tokens
}

# Default debate topics for tournament matches
DEFAULT_TOPICS = [
    "Technology companies should be broken up to prevent monopolies",
    "Universal basic income should be implemented globally",
    "Space exploration should be prioritized over climate change mitigation",
    "Artificial intelligence development should be regulated by government",
    "Nuclear energy is essential for achieving carbon neutrality",
    "Social media platforms should be treated as public utilities",
    "Remote work will fundamentally improve society",
    "Genetic engineering should be used to enhance human capabilities",
    "Cryptocurrency will replace traditional banking systems",
    "Autonomous vehicles should be mandated by law",
    "Privacy is more important than security in digital systems",
    "Free speech should have no limitations in democratic societies",
    "Economic growth is incompatible with environmental sustainability",
    "Education should be completely personalized using AI",
    "Global governance is necessary to solve international problems",
    "Traditional employment will become obsolete within 20 years",
    "Scientific research should be completely open and unrestricted",
    "Urban design should prioritize cars over pedestrians",
    "Democracy is the best form of government for all societies",
    "Cultural preservation should take priority over economic development",
    "Healthcare should be a human right regardless of cost",
    "Competition is more effective than cooperation for human progress",
    "Technological progress inevitably improves human well-being",
    "Individual freedom should never be sacrificed for collective good",
    "Globalization benefits developing countries more than developed ones",
    "Artificial scarcity in digital goods should be illegal",
    "Meritocracy is achievable in modern society",
    "Scientific consensus is the best basis for public policy",
    "Cultural diversity strengthens rather than weakens society",
    "Innovation requires accepting high levels of risk and failure",
    "Objective truth exists independently of human perception",
    "Traditional institutions are essential for social stability",
]


class TournamentManager:
    """Manages tournament creation, progression, and bracket generation."""

    def __init__(
        self,
        db_path: str = "tournaments.db",
        model_manager: ModelManager | None = None,
        debate_manager: Any | None = None,
    ):
        self.db = TournamentDatabaseManager(db_path)
        self.model_manager = model_manager
        self.debate_manager = debate_manager

    async def create_tournament(self, request: TournamentCreateRequest) -> int:
        """Create tournament with bracket generation."""
        logger.info(
            f"Creating tournament: {request.name} ({request.weight_class.value})"
        )

        # Get eligible models for weight class
        eligible_models = await self._get_eligible_models(request.weight_class)
        if len(eligible_models) < request.bracket_size:
            raise ValueError(
                f"Not enough eligible models for {request.weight_class.value} weight class. "
                f"Found {len(eligible_models)}, need {request.bracket_size}"
            )

        # Calculate tournament structure
        total_rounds = self._calculate_rounds(request.bracket_size)

        # Create tournament record
        tournament = Tournament(
            name=request.name,
            weight_class=request.weight_class,
            format=request.format,
            word_limit=request.word_limit,
            status=TournamentStatus.CREATED,
            bracket_size=request.bracket_size,
            current_round=1,
            total_rounds=total_rounds,
            tournament_metadata={
                "judge_models": request.judge_models,
                "judge_provider": request.judge_provider,
            },
        )

        tournament_id = self.db.create_tournament(tournament)

        # Add judges
        for judge_model in request.judge_models:
            judge = TournamentJudge(
                tournament_id=tournament_id,
                judge_model_id=judge_model,
                judge_provider=request.judge_provider,
            )
            self.db.add_judge(judge)

        # Generate seeded participants
        seeded_models = self._seed_models(eligible_models, request.bracket_size)
        for i, model in enumerate(seeded_models):
            participant = TournamentParticipant(
                tournament_id=tournament_id,
                model_id=model["id"],
                model_name=model["display_name"],
                seed_number=i + 1,
            )
            self.db.add_participant(participant)

        # Generate first round bracket
        self._generate_bracket(tournament_id, seeded_models)

        logger.info(
            f"Created tournament {tournament_id} with {len(seeded_models)} participants"
        )
        return tournament_id

    async def start_tournament(self, tournament_id: int) -> bool:
        """Begin tournament execution by starting first round matches."""
        tournament = self.db.get_tournament(tournament_id)
        if not tournament:
            raise ValueError(f"Tournament {tournament_id} not found")

        if tournament.status != TournamentStatus.CREATED:
            raise ValueError(f"Tournament {tournament_id} is not in CREATED status")

        # Get first round matches
        first_round_matches = self.db.get_matches(tournament_id, round_number=1)
        if not first_round_matches:
            raise ValueError(f"No matches found for tournament {tournament_id}")

        # Start all first round debates
        started_matches = 0
        for match in first_round_matches:
            if match.status == MatchStatus.PENDING:
                success = await self._start_match_debate(match)
                if success:
                    started_matches += 1

        # Update tournament status
        self.db.update_tournament_status(
            tournament_id, TournamentStatus.IN_PROGRESS, started_at=datetime.now()
        )

        logger.info(
            f"Started tournament {tournament_id} with {started_matches} matches"
        )
        return started_matches > 0

    async def advance_tournament(self, tournament_id: int) -> bool:
        """Advance tournament to next round after current round completion."""
        tournament = self.db.get_tournament(tournament_id)
        if not tournament:
            raise ValueError(f"Tournament {tournament_id} not found")

        if tournament.status != TournamentStatus.IN_PROGRESS:
            raise ValueError(f"Tournament {tournament_id} is not in progress")

        # Check if current round is complete
        round_status = self.db.get_round_status(tournament_id, tournament.current_round)
        if not round_status["all_completed"]:
            logger.warning(f"Round {tournament.current_round} not yet completed")
            return False

        # If this was the final round, crown champion
        if tournament.current_round == tournament.total_rounds:
            return await self._complete_tournament(tournament_id)

        # Advance to next round
        next_round = tournament.current_round + 1
        winners = await self._get_round_winners(tournament_id, tournament.current_round)

        # Generate next round matches
        await self._generate_next_round(tournament_id, next_round, winners)

        # Update tournament round
        self.db.update_tournament_status(
            tournament_id, TournamentStatus.IN_PROGRESS, current_round=next_round
        )

        # Start next round matches
        next_round_matches = self.db.get_matches(tournament_id, round_number=next_round)
        started_matches = 0
        for match in next_round_matches:
            if match.status == MatchStatus.PENDING:
                success = await self._start_match_debate(match)
                if success:
                    started_matches += 1

        logger.info(f"Advanced tournament {tournament_id} to round {next_round}")
        return True

    def get_tournament_status(self, tournament_id: int) -> Tournament | None:
        """Get current tournament state."""
        return self.db.get_tournament(tournament_id)

    def get_bracket_view(self, tournament_id: int) -> BracketData | None:
        """Get bracket visualization data."""
        return self.db.get_bracket_data(tournament_id)

    async def handle_match_completion(
        self, tournament_id: int, match_id: int, winner_model_id: str, debate_id: str
    ) -> bool:
        """Handle completion of a tournament match."""
        # Update match with winner
        success = self.db.update_match(
            match_id,
            MatchStatus.COMPLETED,
            winner_model_id=winner_model_id,
            debate_id=debate_id,
        )

        if not success:
            return False

        # Get tournament and current round status
        tournament = self.db.get_tournament(tournament_id)
        if not tournament:
            return False

        round_status = self.db.get_round_status(tournament_id, tournament.current_round)

        # If round is complete, try to advance tournament
        if round_status["all_completed"]:
            await self.advance_tournament(tournament_id)

        return True

    async def _get_eligible_models(
        self, weight_class: WeightClass
    ) -> list[dict[str, Any]]:
        """Get models eligible for the specified weight class."""
        if not self.model_manager:
            raise ValueError("Model manager not available")

        all_models = await self.model_manager.get_enhanced_models()

        # Filter by weight class and exclude preview models
        min_cost, max_cost = WEIGHT_CLASS_RANGES[weight_class]
        eligible_models = []

        for model in all_models:
            # Exclude preview models to prevent anonymous ringers
            if model.get("is_preview", False):
                continue

            avg_cost = model["pricing"]["avg_cost_per_1k"]

            # Handle free models specially
            if weight_class == WeightClass.FREE:
                if model["pricing"]["is_free"]:
                    eligible_models.append(model)
            else:
                if min_cost <= avg_cost < max_cost:
                    eligible_models.append(model)

        logger.info(
            f"Found {len(eligible_models)} eligible models for {weight_class.value} weight class"
        )
        return eligible_models

    def _seed_models(
        self, models: list[dict[str, Any]], bracket_size: int
    ) -> list[dict[str, Any]]:
        """Rank models by value_score and return top bracket_size models."""
        # Sort by value_score in descending order (highest value = best seed)
        sorted_models = sorted(
            models, key=lambda m: m.get("value_score", 0), reverse=True
        )

        # Take top models for bracket
        seeded_models = sorted_models[:bracket_size]

        logger.info(
            f"Seeded {len(seeded_models)} models for bracket "
            f"(top value_score: {seeded_models[0].get('value_score', 0):.2f}, "
            f"bottom: {seeded_models[-1].get('value_score', 0):.2f})"
        )

        return seeded_models

    def _generate_bracket(
        self, tournament_id: int, seeded_models: list[dict[str, Any]]
    ) -> None:
        """Generate complete tournament bracket with March Madness seeding."""
        bracket_size = len(seeded_models)

        # Generate unique topics for all matches
        all_topics = self._generate_unique_topics(
            bracket_size - 1
        )  # bracket_size - 1 total matches
        topic_index = 0

        # Generate first round matches with March Madness pairing
        first_round_matches = self._generate_first_round_pairs(seeded_models)

        for i, (model_a, model_b) in enumerate(first_round_matches):
            match = TournamentMatch(
                tournament_id=tournament_id,
                round_number=1,
                match_number=i + 1,
                model_a_id=model_a["id"],
                model_b_id=model_b["id"] if model_b else None,
                topic=all_topics[topic_index],
                status=MatchStatus.BYE if model_b is None else MatchStatus.PENDING,
            )
            self.db.add_match(match)
            topic_index += 1

        logger.info(f"Generated bracket for tournament {tournament_id}")

    def _generate_first_round_pairs(
        self, seeded_models: list[dict[str, Any]]
    ) -> list[tuple]:
        """Generate first round pairings using March Madness seeding."""
        n = len(seeded_models)

        # For perfect brackets (powers of 2), use classic seeding
        if n & (n - 1) == 0:  # Check if power of 2
            pairs = []
            for i in range(n // 2):
                seed_a = i + 1
                seed_b = n - i
                model_a = seeded_models[seed_a - 1]  # Convert to 0-based index
                model_b = seeded_models[seed_b - 1]
                pairs.append((model_a, model_b))
            return pairs

        # For non-power-of-2 brackets, some top seeds get byes
        next_power_of_2 = 2 ** math.ceil(math.log2(n))
        byes_needed = next_power_of_2 - n

        pairs = []
        used_seeds = set()

        # Top seeds get byes (advance automatically)
        for i in range(byes_needed):
            seed = i + 1
            model = seeded_models[seed - 1]
            pairs.append((model, None))  # None indicates bye
            used_seeds.add(seed)

        # Pair remaining seeds
        remaining_seeds = [i + 1 for i in range(n) if (i + 1) not in used_seeds]
        remaining_seeds.sort()

        for i in range(0, len(remaining_seeds), 2):
            if i + 1 < len(remaining_seeds):
                seed_a = remaining_seeds[i]
                seed_b = remaining_seeds[i + 1]
                model_a = seeded_models[seed_a - 1]
                model_b = seeded_models[seed_b - 1]
                pairs.append((model_a, model_b))

        return pairs

    def _generate_unique_topics(self, num_topics: int) -> list[str]:
        """Generate unique topics for tournament matches."""
        if num_topics <= len(DEFAULT_TOPICS):
            # Shuffle and take what we need
            topics = DEFAULT_TOPICS.copy()
            random.shuffle(topics)
            return topics[:num_topics]
        else:
            # Need more topics than we have defaults
            topics = DEFAULT_TOPICS.copy()
            for i in range(num_topics - len(DEFAULT_TOPICS)):
                topics.append(f"Generated topic #{i + 1} for extended tournament")
            random.shuffle(topics)
            return topics

    def _calculate_rounds(self, bracket_size: int) -> int:
        """Calculate total rounds needed for bracket size."""
        return math.ceil(math.log2(bracket_size))

    async def _start_match_debate(self, match: TournamentMatch) -> bool:
        """Start a debate for a tournament match."""
        if not self.debate_manager:
            logger.warning("Debate manager not available, cannot start match debates")
            return False

        try:
            # Ensure match has ID (database matches must have IDs)
            if match.id is None:
                raise ValueError(f"Match retrieved from database missing required ID")

            # Get tournament details
            tournament = self.db.get_tournament(match.tournament_id)
            if not tournament:
                return False

            # Get judges
            judges = self.db.get_judges(match.tournament_id)
            if not judges:
                return False

            # Handle bye matches
            if match.model_b_id is None:
                self.db.update_match(
                    match.id, MatchStatus.COMPLETED, winner_model_id=match.model_a_id
                )
                logger.info(f"Match {match.id} completed as bye for {match.model_a_id}")
                return True

            # Create debate setup (this would need to be adapted to your actual DebateSetupRequest)
            setup_data = {
                "topic": match.topic,
                "format": tournament.format,
                "word_limit": tournament.word_limit,
                "models": {
                    "model_a": {"id": match.model_a_id},
                    "model_b": {"id": match.model_b_id},
                },
                "judge_models": [judge.judge_model_id for judge in judges],
                "tournament_match_id": match.id,  # Tag for callback system
            }

            # Start the debate (implementation depends on your DebateManager interface)
            debate_id = await self.debate_manager.create_tournament_debate(setup_data)
            if debate_id:
                # Update match with debate ID and status
                self.db.update_match(
                    match.id, MatchStatus.IN_PROGRESS, debate_id=debate_id
                )
                logger.info(f"Started debate {debate_id} for match {match.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to start debate for match {match.id}: {e}")

        return False

    async def _get_round_winners(
        self, tournament_id: int, round_number: int
    ) -> list[str]:
        """Get winners from completed round."""
        matches = self.db.get_matches(tournament_id, round_number)
        winners = []

        for match in matches:
            if match.status == MatchStatus.COMPLETED and match.winner_model_id:
                winners.append(match.winner_model_id)

        return winners

    async def _generate_next_round(
        self, tournament_id: int, round_number: int, winners: list[str]
    ) -> None:
        """Generate matches for next round."""
        # Pair winners for next round
        topics = self._generate_unique_topics(len(winners) // 2)

        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                match = TournamentMatch(
                    tournament_id=tournament_id,
                    round_number=round_number,
                    match_number=(i // 2) + 1,
                    model_a_id=winners[i],
                    model_b_id=winners[i + 1],
                    topic=topics[i // 2],
                    status=MatchStatus.PENDING,
                )
                self.db.add_match(match)

    async def _complete_tournament(self, tournament_id: int) -> bool:
        """Complete tournament and crown champion."""
        # Get tournament details
        tournament = self.db.get_tournament(tournament_id)
        if not tournament:
            raise ValueError(f"Tournament {tournament_id} not found")

        # Get final match to determine winner
        final_matches = self.db.get_matches(tournament_id, round_number=None)
        final_match = None

        for match in final_matches:
            if match.round_number == tournament.total_rounds:
                final_match = match
                break

        if not final_match or not final_match.winner_model_id:
            logger.error(f"Could not determine winner for tournament {tournament_id}")
            return False

        # Update tournament as completed
        self.db.update_tournament_status(
            tournament_id,
            TournamentStatus.COMPLETED,
            completed_at=datetime.now(),
            winner_model_id=final_match.winner_model_id,
        )

        logger.info(
            f"Tournament {tournament_id} completed, winner: {final_match.winner_model_id}"
        )
        return True
