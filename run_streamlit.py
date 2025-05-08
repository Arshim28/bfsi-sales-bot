#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the Streamlit app with the correct configuration."""
    parser = argparse.ArgumentParser(description="Run the BFSI Sales Bot Generator Streamlit app")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")
    parser.add_argument("--api-url", type=str, help="FastAPI backend URL (default: from .env or http://localhost:8000)")
    args = parser.parse_args()
    
    # Set API URL environment variable if provided
    if args.api_url:
        os.environ["API_URL"] = args.api_url
    
    # Check if API_URL is set
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    
    # Print startup info
    print(f"Starting Streamlit app on port {args.port}")
    print(f"Connecting to FastAPI backend at: {api_url}")
    
    # Prepare command
    cmd = [
        "streamlit", "run", "streamlit_app.py",
        "--server.port", str(args.port),
        "--browser.serverAddress", "localhost",
        "--server.headless", "true"
    ]
    
    try:
        # Run streamlit as a subprocess
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit app...")
        sys.exit(0)
    except Exception as e:
        print(f"Error running Streamlit app: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 