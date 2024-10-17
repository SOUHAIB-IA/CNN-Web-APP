from flask import Flask
from routes import routes
import os

def create_app():
    app = Flask(__name__)
    
    '''
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB
    '''
    # Register the routes blueprint
    app.register_blueprint(routes)

    return app

if __name__ == '__main__':
    # Create the app instance
    app = create_app()

    # Create uploads directory if it doesn't exist
    #if not os.path.exists(app.config['UPLOAD_FOLDER']):
    #    os.makedirs(app.config['UPLOAD_FOLDER'])

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
