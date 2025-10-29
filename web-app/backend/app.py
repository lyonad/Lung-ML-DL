"""
Flask Application - Lung Cancer Risk Prediction Web Application

Main application file that initializes Flask, loads models, and sets up routes.

Author: Research Team
Date: October 28, 2025
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from backend.models.model_loader import ModelLoader
from backend.api.predict import predict_bp, init_predict_api


def create_app(config_name='development'):
    """
    Create and configure Flask application.
    
    Args:
        config_name: Configuration environment (development/production/testing)
        
    Returns:
        Flask app instance
    """
    # Initialize Flask app
    app = Flask(__name__,
                template_folder='../templates',
                static_folder='../static')
    
    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)
    
    # Create config instance for model loader
    config_instance = config_class()
    
    # Enable CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": config_class.CORS_ORIGINS,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # Initialize model loader
    print("\n" + "="*80)
    print("LUNG CANCER PREDICTION WEB APPLICATION")
    print("Deep Learning vs. Random Forest - Pakistan Healthcare")
    print("="*80 + "\n")
    
    model_loader = ModelLoader(config_instance)
    success = model_loader.load_models()
    
    if not success:
        print("\n[WARNING] Models failed to load. API will return errors.")
        print("   Please ensure trained models are available in:")
        print(f"   - {config_instance.MODELS_DIR}")
        print(f"   - {config_instance.RESULTS_DIR}\n")
    
    # Initialize prediction API with model loader
    init_predict_api(model_loader)
    
    # Register blueprints
    app.register_blueprint(predict_bp, url_prefix='/api')
    
    # Main routes
    @app.route('/')
    def index():
        """Main application page"""
        return render_template('index.html', config=config_class)
    
    @app.route('/about')
    def about():
        """About page"""
        return render_template('about.html', config=config_class)
    
    @app.route('/documentation')
    def documentation():
        """API documentation page"""
        return render_template('documentation.html', config=config_class)
    
    # Static file routes
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files"""
        return send_from_directory(app.static_folder, filename)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        return {'error': 'Internal server error'}, 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle all other exceptions"""
        app.logger.error(f"Unhandled exception: {str(error)}")
        return {'error': 'An unexpected error occurred'}, 500
    
    return app


def main():
    """
    Main entry point for running the application.
    """
    # Get environment
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Create app
    app = create_app(env)
    config = app.config
    
    # Run server
    print("\n" + "="*80)
    print("STARTING WEB SERVER")
    print("="*80)
    print(f"\nEnvironment: {env}")
    print(f"Debug Mode: {config['DEBUG']}")
    print(f"Host: {config['HOST']}")
    print(f"Port: {config['PORT']}")
    print(f"\n[WEB] Application URL: http://{config['HOST']}:{config['PORT']}")
    print(f"[API] API Endpoint: http://{config['HOST']}:{config['PORT']}/api/predict")
    print(f"[DOC] API Docs: http://{config['HOST']}:{config['PORT']}/documentation")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    try:
        app.run(
            host=config['HOST'],
            port=config['PORT'],
            debug=config['DEBUG']
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("SERVER STOPPED")
        print("="*80 + "\n")
    except Exception as e:
        print(f"\n✗ ERROR: Failed to start server: {str(e)}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()

