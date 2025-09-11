#!/usr/bin/env python3
"""Main entry point for the Dialectus AI Debate System."""

import sys
from pathlib import Path

def main():
    """Main entry point with usage information."""
    print("Dialectus AI Debate System")
    print("=" * 40)
    print("Available entry points:")
    print()
    print("🌐 Web Server (API + WebSocket):")
    print("   python web_server.py")
    print()
    print("🖥️  CLI Interface:")
    print("   cd ../dialectus-cli")
    print("   python cli.py")
    print()
    print("📦 Missing components? Clone the other repositories:")
    print("   CLI: https://github.com/psarno/dialectus-cli")
    print("   Web: https://github.com/psarno/dialectus-web") 
    print()
    print("For the web interface, run the web server and visit:")
    print("http://localhost:8000")
    print()
    
    # Optionally launch web server if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        print("Starting web server...")
        import subprocess
        web_server_path = Path(__file__).parent / "web_server.py"
        subprocess.run([sys.executable, str(web_server_path)])

if __name__ == "__main__":
    main()