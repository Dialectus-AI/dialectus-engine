#!/usr/bin/env python3
"""Main entry point for the Dialectus AI Debate System."""

import logging
import os
import sys
from pathlib import Path
from web.api import app

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


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


def print_usage():
    """Print usage information for local development."""

    print("Dialectus AI Debate System")
    print("=" * 40)
    print("Available entry points:")
    print()
    print("🌐 Web Server (API + WebSocket):")
    print("   python main.py --web")
    print("   python main.py  (starts web server)")
    print()
    print("🖥️  CLI Interface:")
    print("   cd ../dialectus-cli")
    print("   python cli.py")
    print()
    print("📦 Missing components? Clone the other repositories:")
    print("   CLI: https://github.com/psarno/dialectus-cli")
    print("   Web: https://github.com/psarno/dialectus-web")
    print()


def start_web_server():
    """Start the FastAPI web server."""

    setup_logging()
    
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print("🎭 Starting AI Debate System Web Server...")
    print(f"📡 API Documentation: http://localhost:{port}/docs")
    print(f"🌐 Web Interface: http://localhost:{port}")
    print(f"🔌 WebSocket: ws://localhost:{port}/ws/debate/{{id}}")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)


def main():
    """Main entry point."""
    # Check for production environment (Railway, Docker, Heroku, etc.)
    is_production = any([
        "RAILWAY_ENVIRONMENT" in os.environ,
        "PORT" in os.environ,
        "DYNO" in os.environ,  # Heroku
        os.environ.get("ENVIRONMENT") == "production"
    ])
    
    # Check command line arguments
    start_server = (
        is_production or 
        "--web" in sys.argv or 
        (len(sys.argv) == 1 and is_production)  # No args in production = start server
    )
    
    if start_server:
        start_web_server()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
    else:
        # Local development - show usage and optionally start server
        print_usage()
        print("💡 Tip: Use 'python main.py --web' to start the server")
        print("🚀 In production, server starts automatically")


if __name__ == "__main__":
    main()