from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
import os
from model.model import process_data, train_model, prepare_data, evaluate_model
import pandas as pd
from werkzeug.utils import secure_filename

# Initialize Blueprint for routing
routes = Blueprint('routes', __name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xls', 'xlsx'}

# Set the folder in the app config
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to homepage
@routes.route('/')
def index():
    data_preview = None  # Initialize data preview to None
    return render_template('index.html', data_preview=data_preview)

# Route to handle file uploads
@routes.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file part is present in the request
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Clean up old files
        old_file = session.get('file_path')
        if old_file and os.path.exists(old_file):
            os.remove(old_file)

        # Save the new file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Process the uploaded file
        try:
            data = process_data(file_path)
            data_preview = data.head(10)
            session['file_path'] = file_path  # Store file path in session
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)

        flash(f'File {filename} uploaded successfully')
        return render_template('index.html', data_preview=data_preview)

    flash('Invalid file type. Only CSV, XLS, or XLSX files are allowed.')
    return redirect(request.url)

# Route to get CNN model configuration from the form
@routes.route('/get_config', methods=['POST'])
def get_config():
    # Get CNN configuration from the form
    num_layers = request.form.get('numLayers', type=int)
    optimizer = request.form.get('optimizer')
    learning_rate = request.form.get('learning_rate', type=float)
    batch_size = request.form.get('batch_size', type=int)
    epochs = request.form.get('epochs', type=int)
    dropout_rate = request.form.get('dropout_rate', type=float)

    # Get the target column
    target_column = request.form.get('target_column')

    # Get the selected features (checkboxes)
    selected_features = request.form.getlist('features')

    # Get the train-test split percentage
    train_test_split = request.form.get('train_test_split', type=int)

    # Collect layers configuration
    layers_config = []
    for i in range(1, num_layers + 1):
        activation = request.form.get(f'activation_layer_{i}')
        neurons_nbr = request.form.get(f'neurons_layer_{i}', type=int)
        layers_config.append({
            'activation': activation,
            'neurons': neurons_nbr
        })

    # Store the configuration in the session
    model_config = {
        'nm_layers': num_layers,
        'layers': layers_config,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'dropout_rate': dropout_rate,
        'target_column': target_column,
        'columns': selected_features,
        'train_test_split': train_test_split,
    }

    session['model_config'] = model_config 

    flash('Model configured successfully!')
    return redirect(url_for('routes.index'))

# Route to train the CNN model
@routes.route('/train_model', methods=['POST'])
def train_model_route():
    # Ensure the file is uploaded and configuration is available
    if 'file_path' not in session or 'model_config' not in session:
        flash('File not uploaded or model configuration missing!')
        return redirect(url_for('routes.index'))

    # Load the configuration and file path from the session
    model_config = session['model_config']
    file_path = session['file_path']

    try:
        # Process the uploaded file and prepare data
        data = process_data(file_path)
        if data is None:
            flash('Error processing the file')
            return redirect(url_for('routes.index'))
        
        # Prepare the data for training
        X_train, X_test, y_train, y_test = prepare_data(data, model_config)
        
        # Train the model using the provided configuration
        history, model = train_model(model_config, X_train, y_train)

        # Evaluate the model
        accuracy = evaluate_model(model, X_test, y_test)

        flash(f'Model trained successfully with accuracy: {accuracy:.2f}')
        
        # Render the results to the UI
        return render_template('training_results.html', accuracy=accuracy, history=history.history)

    except Exception as e:
        flash(f'Error training model: {str(e)}')
        return redirect(url_for('routes.index'))
