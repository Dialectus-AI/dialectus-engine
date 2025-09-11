"""Base classes and interfaces for debate formats."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from enum import Enum
from dataclasses import dataclass

from debate_engine.types import DebatePhase, Position


@dataclass 
class FormatPhase:
    """A phase within a specific debate format."""
    phase: DebatePhase
    name: str
    instruction: str
    speaking_order: List[str]  # Order of speaker IDs
    time_multiplier: float = 1.0  # Multiply base time limit
    
    
class DebateFormat(ABC):
    """Abstract base class for debate formats."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Format name."""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable format name for display in UI."""
        pass
    
    @property 
    @abstractmethod
    def description(self) -> str:
        """Format description."""
        pass
    
    @abstractmethod
    def get_phases(self, participants: List[str]) -> List[FormatPhase]:
        """Get the phases for this format given participants."""
        pass
    
    @abstractmethod
    def get_position_assignments(self, participants: List[str]) -> Dict[str, Position]:
        """Assign positions to participants."""
        pass
        
    @abstractmethod
    def get_format_instructions(self) -> str:
        """Get format-specific instructions for system prompts."""
        pass
    
    @abstractmethod
    def get_side_labels(self, participants: List[str]) -> Dict[str, str]:
        """Get format-specific labels for each participant's side/role."""
        pass
        
    def get_max_participants(self) -> int:
        """Maximum number of participants supported."""
        return 2
        
    def get_min_participants(self) -> int:
        """Minimum number of participants required."""
        return 2
        
    def get_topic_generation_messages(self) -> List[Dict[str, str]]:
        """Get format-specific messages for AI topic generation."""
        return [
            {
                "role": "system",
                "content": "You are an expert debate topic generator. Create engaging, balanced, and thought-provoking debate topics that have clear pro and con sides. Topics should be contemporary, relevant, and suitable for intellectual discourse."
            },
            {
                "role": "user", 
                "content": "Generate a single debate topic that would make for an interesting and balanced debate. The topic should be phrased as a clear statement that can be argued for or against. Make it thought-provoking and current. Respond with just the topic statement, no additional text."
            }
        ]
        
