#!/usr/bin/env python3
"""Web server entry point for the Dialectus AI Debate System."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web.api import app


def setup_logging():
    """Configure logging for the web server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noisy third-party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


if __name__ == "__main__":
    setup_logging()

    import uvicorn

    print("🎭 Starting AI Debate System Web Server...")
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    print("🔌 WebSocket: ws://localhost:8000/ws/debate/{id}")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
