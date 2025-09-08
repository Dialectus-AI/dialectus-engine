"""Parliamentary debate format implementation."""

from typing import List, Dict
from .base import DebateFormat, FormatPhase, Position
from debate_engine.types import DebatePhase


class ParliamentaryFormat(DebateFormat):
    """Parliamentary debate format with government and opposition."""
    
    @property
    def name(self) -> str:
        return "parliamentary"
        
    @property
    def description(self) -> str:
        return "Parliamentary debate with Government and Opposition sides, formal procedures and structured speeches"
    
    def get_phases(self, participants: List[str]) -> List[FormatPhase]:
        """Parliamentary format: PM -> LO -> Deputy PM -> Deputy LO -> Rebuttals"""
        if len(participants) < 2:
            raise ValueError("Parliamentary format requires at least 2 participants")
            
        # For 2 participants: Prime Minister and Leader of Opposition
        pm = participants[0] 
        lo = participants[1]
        
        return [
            FormatPhase(
                phase=DebatePhase.OPENING,
                name="Prime Minister's Opening",
                instruction="As Prime Minister, define the motion, present the government's case, and outline key arguments. Set the framework for the debate.",
                speaking_order=[pm],
                time_multiplier=1.2  # Slightly longer for PM
            ),
            FormatPhase(
                phase=DebatePhase.OPENING, 
                name="Leader of Opposition's Response",
                instruction="As Leader of Opposition, directly challenge the government's case. Present alternative framework and counter-arguments.",
                speaking_order=[lo],
                time_multiplier=1.1
            ),
            FormatPhase(
                phase=DebatePhase.REBUTTAL,
                name="Government Rebuttal", 
                instruction="Defend government position against opposition attacks. Reinforce key government arguments and address opposition points.",
                speaking_order=[pm]
            ),
            FormatPhase(
                phase=DebatePhase.REBUTTAL,
                name="Opposition Rebuttal",
                instruction="Final opposition response. Demolish government arguments and consolidate opposition case.",
                speaking_order=[lo] 
            ),
            FormatPhase(
                phase=DebatePhase.CLOSING,
                name="Closing Statements",
                instruction="Make final appeals to convince the House. Summarize your side's victory in this debate.",
                speaking_order=[lo, pm]  # Opposition closes first, government gets last word
            )
        ]
    
    def get_position_assignments(self, participants: List[str]) -> Dict[str, Position]:
        """First participant is Government (PRO), second is Opposition (CON).""" 
        assignments = {}
        for i, participant in enumerate(participants[:2]):
            assignments[participant] = Position.PRO if i == 0 else Position.CON
        return assignments
    
    def get_side_labels(self, participants: List[str]) -> Dict[str, str]:
        """Return Parliamentary labels: Government and Opposition."""
        labels = {}
        for i, participant in enumerate(participants[:2]):
            labels[participant] = "Government" if i == 0 else "Opposition"
        return labels
        
    def get_format_instructions(self) -> str:
        """Parliamentary format-specific instructions."""
        return """PARLIAMENTARY DEBATE FORMAT:
- Address the Speaker and fellow members respectfully
- Government defends the motion, Opposition opposes it
- Use parliamentary language: "Honorable members", "The motion before the House"  
- Focus on policy implications and practical governance
- Challenge opposing arguments while maintaining decorum

Remember: You are debating in a formal parliamentary setting."""
        
    def get_max_participants(self) -> int:
        """Parliamentary can support up to 4 participants (2 per side)."""
        return 4