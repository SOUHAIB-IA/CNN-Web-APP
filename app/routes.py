from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
#from model.train import train_model
#from model.data_processing import process_data

# Initialize Blueprint for routing
routes = Blueprint('routes', __name__)
'''
# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv'}

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
'''
# Route to homepage
@routes.route('/')
def index():
    return render_template('index.html')
'''
# Route to upload dataset
@routes.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process the uploaded data (e.g., CSV file)
        data = process_data(filepath)
        
        flash(f'Data uploaded and processed successfully: {filename}')
        return redirect(url_for('routes.configure_model'))

    else:
        flash('Invalid file format. Please upload a CSV file.')
        return redirect(request.url)

# Route to configure the neural network
@routes.route('/configure', methods=['GET', 'POST'])
def configure_model():
    if request.method == 'POST':
        # Extract the configuration from the form
        layers = int(request.form.get('layers'))
        neurons = int(request.form.get('neurons'))
        activation = request.form.get('activation')
        epochs = int(request.form.get('epochs'))
        learning_rate = float(request.form.get('learning_rate'))

        # Store the configuration or pass it to the training function
        config = {
            'layers': layers,
            'neurons': neurons,
            'activation': activation,
            'epochs': epochs,
            'learning_rate': learning_rate
        }

        # Start training the model
        results = train_model(config)

        return jsonify(results)

    return render_template('configure_model.html')

# Route to display training results
@routes.route('/results')
def display_results():
    # Here, you'd load results from the training process
    # E.g., loss curve, accuracy, etc.
    loss_curve_path = os.path.join('results', 'loss_curve.png')
    return render_template('results.html', loss_curve_path=loss_curve_path)

# API route to get real-time training progress (optional)
@routes.route('/progress', methods=['GET'])
def training_progress():
    # Example: Return JSON of training progress
    progress = {
        "epoch": 5,
        "accuracy": 0.85,
        "loss": 0.35
    }
    return jsonify(progress)
'''