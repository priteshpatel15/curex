#!/usr/bin/env python3
"""
Main entry point for the Natural Language Data Analysis System.
Provides options to run the CLI, web server, or API directly.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main entry point with argument parsing"""
    
    # If 'cli' is the first argument, delegate to Click CLI
    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        # Remove 'cli' from args and let Click handle the rest
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        
        try:
            from src.cli.main import cli
            cli()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        return
    
    # Handle server mode and help with argparse
    parser = argparse.ArgumentParser(
        description="Natural Language Data Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py cli                    # Start interactive CLI
  python main.py cli --help             # Show CLI help
  python main.py server                 # Start web server
  python main.py server --port 9000     # Start web server on port 9000
  python main.py --help                 # Show this help message
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')
    
    # CLI mode (just for help text)
    cli_parser = subparsers.add_parser('cli', help='Start interactive command line interface')
    cli_parser.add_argument('--help-cli', action='store_true', 
                           help='This will delegate to the Click CLI - use "python main.py cli --help" for CLI options')
    
    # Server mode
    server_parser = subparsers.add_parser('server', help='Start web server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    server_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.mode == 'server':
        # Run web server
        from src.web.api import run_server
        
        print(f"Starting web server on {args.host}:{args.port}")
        print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
        print(f"API documentation available at: http://{args.host}:{args.port}/docs")
        
        try:
            run_server(host=args.host, port=args.port, debug=args.debug)
        except KeyboardInterrupt:
            print("\nServer stopped.")
            sys.exit(0)
    
    else:
        # No mode specified or invalid mode, show help
        parser.print_help()
        
        # Offer quick start options
        print("\nQuick start:")
        print("1. For interactive analysis: python main.py cli")
        print("2. For web interface: python main.py server")
        print("3. For CLI help: python main.py cli --help")


if __name__ == "__main__":
    main() 