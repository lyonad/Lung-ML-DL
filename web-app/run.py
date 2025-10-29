#!/usr/bin/env python
"""
Quick start script for Lung Cancer Prediction Web Application

Usage:
    python run.py
    python run.py --port 8000
    python run.py --host 0.0.0.0 --port 8000 --debug
"""

import sys
import argparse
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.app import create_app


def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Lung Cancer Risk Prediction Web Application'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host address (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port number (default: 5000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--env',
        choices=['development', 'production', 'testing'],
        default='development',
        help='Environment (default: development)'
    )
    
    args = parser.parse_args()
    
    # Create Flask app
    app = create_app(args.env)
    
    # Override config with CLI arguments
    app.config['HOST'] = args.host
    app.config['PORT'] = args.port
    if args.debug:
        app.config['DEBUG'] = True
    
    # Print startup information
    print("\n" + "="*80)
    print("LUNG CANCER PREDICTION WEB APPLICATION")
    print("="*80)
    print(f"\n🌐 Server starting at: http://{args.host}:{args.port}")
    print(f"📊 API Endpoint: http://{args.host}:{args.port}/api/predict")
    print(f"📖 Documentation: http://{args.host}:{args.port}/documentation")
    print(f"\nEnvironment: {args.env}")
    print(f"Debug Mode: {app.config['DEBUG']}")
    print("\nPress CTRL+C to stop")
    print("="*80 + "\n")
    
    # Run application
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=app.config['DEBUG']
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("SERVER STOPPED")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()

