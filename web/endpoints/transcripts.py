"""Transcript management endpoints."""

import logging

from fastapi import HTTPException, APIRouter

from debate_engine.transcript import TranscriptManager
from formats import format_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/transcripts")
async def get_transcripts(page: int = 1, limit: int = 20):
    """Get paginated list of saved transcripts from SQLite database."""
    try:
        # Validate pagination parameters
        if page < 1:
            page = 1
        if limit < 1 or limit > 100:
            limit = 20

        # Calculate offset for pagination
        offset = (page - 1) * limit

        # Get transcripts from database with explicit path
        db_path = "debates.db"
        transcript_manager = TranscriptManager(str(db_path))
        transcripts = transcript_manager.db_manager.list_debates_with_metadata(
            limit=limit, offset=offset
        )
        total_count = transcript_manager.get_debate_count()

        # Format response with pagination metadata
        return {
            "transcripts": transcripts,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_count,
                "total_pages": (total_count + limit - 1) // limit,
                "has_next": offset + limit < total_count,
                "has_prev": page > 1,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get transcripts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcripts/{transcript_id}")
async def get_transcript(transcript_id: int):
    """Get a specific transcript by ID with full message content."""
    try:
        db_path = "debates.db"
        transcript_manager = TranscriptManager(str(db_path))
        transcript_data = transcript_manager.load_transcript(transcript_id)

        if not transcript_data:
            raise HTTPException(status_code=404, detail="Transcript not found")

        # Enhance transcript data with human-readable phase names
        try:
            format_name = transcript_data["metadata"]["format"]
            debate_format = format_registry.get_format(format_name)

            # Create a mapping from phase enum values to human-readable names
            # We need to reconstruct this from the format's phases
            participants = list(transcript_data["metadata"]["participants"].keys())
            format_phases = debate_format.get_phases(participants)

            # Create phase name mapping: {enum_value: human_readable_name}
            phase_name_mapping = {}
            for format_phase in format_phases:
                phase_name_mapping[format_phase.phase.value] = format_phase.name

            # Add phase name mapping to context metadata for frontend use
            if "context_metadata" not in transcript_data:
                transcript_data["context_metadata"] = {}
            transcript_data["context_metadata"]["phase_names"] = phase_name_mapping

        except Exception as e:
            logger.warning(f"Failed to enhance transcript with phase names: {e}")
            # Continue without phase names if format lookup fails

        return transcript_data
    except Exception as e:
        logger.error(f"Failed to get transcript {transcript_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debates/{debate_id}/transcript")
async def get_debate_transcript(debate_id: str):
    """Get transcript for a specific debate by debate ID."""
    try:
        # Import debate manager from main api
        from web import api
        debate_manager = api.debate_manager

        # Check if debate exists in active debates
        if debate_id in debate_manager.active_debates:
            debate_info = debate_manager.active_debates[debate_id]
            context = debate_info.get("context")
            if context and "transcript_id" in context.metadata:
                transcript_id = context.metadata["transcript_id"]
                # Forward to the existing transcript endpoint
                return await get_transcript(transcript_id)

        raise HTTPException(
            status_code=404, detail=f"Transcript not found for debate {debate_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transcript for debate {debate_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))