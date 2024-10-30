import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session, Response, stream_with_context, send_file
import json
import pandas as pd
from werkzeug.utils import secure_filename
import time  # To simulate long training steps
from model.model import *

# Initialize Blueprint for routing
routes = Blueprint('routes', __name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xls', 'xlsx'}

# Ensure uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

progress = 0  # Initialize progress variable

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_progress(step, total_steps):
    global progress
    progress = int((step / total_steps) * 100)

def save_config(config, config_path='config/config.json'):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# Route to homepage
@routes.route('/')
def index():
    data_preview = None  #Initialize data preview to None
     # Generate epoch numbers

    return render_template('index.html', data_preview=data_preview)
 

# Route to handle file uploads
@routes.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Secure the filename and store it in session
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Remove old uploaded files if any
        old_file = session.get('uploaded_filename')
        if old_file and old_file != filename:
            old_file_path = os.path.join(UPLOAD_FOLDER, old_file)
            if os.path.exists(old_file_path):
                os.remove(old_file_path)

        try:
            data = process_data(file_path)
            data_preview = data.head(10)
            session['uploaded_filename'] = filename  # Store filename in session
            session.modified = True
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)

        flash(f'File {filename} uploaded successfully')
        return render_template('index.html', data_preview=data_preview)

    flash('Invalid file type. Only CSV, XLS, or XLSX files are allowed.')
    return redirect(request.url)

# Helper to get the uploaded file path from the session
def get_uploaded_file_path():
    filename = session.get('uploaded_filename')
    if filename:
        return os.path.join(UPLOAD_FOLDER, filename)
    return None

# Route to handle configuration and model training
@routes.route('/get_config', methods=['POST'])
def get_config():
    try:
        # Collect model parameters from the form
        num_layers = request.form.get('numLayers', type=int)
        optimizer = request.form.get('optimizer')
        learning_rate = request.form.get('learning_rate', type=float)
        batch_size = request.form.get('batch_size', type=int)
        epochs = request.form.get('epochs', type=int)
        dropout_rate = request.form.get('dropout_rate', type=float)
        train_size = request.form.get('train_test_split', type=int)
        features = request.form.getlist('features')
        target_feature = request.form.get('target_column')
        categorical_columns = request.form.getlist('categorical_columns')

        # Collect layer configurations
        layers_config = [
            {
                'activation': request.form.get(f'activation_layer_{i}'),
                'neurons': request.form.get(f'neurons_layer_{i}', type=int),
            }
            for i in range(1, num_layers + 1)
        ]

        # Build model configuration dictionary
        model_config = {
            'nm_layers': num_layers,
            'layers': layers_config,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'dropout_rate': dropout_rate,
            'target_column': target_feature,
            'columns': features,
            'train_size': train_size,
            'categorical_columns':categorical_columns
        }

        # Save the configuration to JSON
        config_dir = 'config'
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        config_path = os.path.join(config_dir, 'config.json')
        save_config(model_config, config_path)

        # Get the uploaded file path
        file_path = get_uploaded_file_path()
        if not file_path:
            flash('No file uploaded or session expired!')
            return redirect(url_for('routes.index'))

        # Train the model using the uploaded file and saved config
        history, model, stats, img_path = train_model(config_path, file_path)

        # Save training results
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("training_history.csv", index=False)
        model.save('model.keras')   
        with open('stats.txt', 'w') as f:
            f.write(f"Accuracy: {stats['accuracy']}\n")
            f.write(stats['classification_report'])
            f.write(f"\nConfusion Matrix:\n{stats['confusion_matrix']}")

        flash('Model trained and saved successfully!')
    except Exception as e:
        flash(f'Error during model training: {str(e)}')
        return redirect(url_for('routes.index'))

    return redirect(url_for('routes.generate_loss_plot_route'))

@routes.route('/train', methods=['GET'])
def start_training():
    global progress
    progress = 0  # Reset progress

    config_path = 'config/config.json'
    file_path = get_uploaded_file_path()

    if not os.path.exists(config_path) or not file_path:
        flash('Configuration or data file is missing!')
        return redirect(url_for('routes.index'))

    @stream_with_context
    def train_and_stream():
        try:
            with open(config_path) as f:
                model_config = json.load(f)

            total_steps = model_config['epochs']

            for step in range(1, total_steps + 1):
                time.sleep(0.5)  # Simulating training
                update_progress(step, total_steps)
                yield f"data:{progress}\n\n"

                if step % 5 == 0:
                    yield "data:heartbeat\n\n"

            yield "data:done\n\n"

        except Exception as e:
            yield f"data:error:{str(e)}\n\n"

    response = Response(train_and_stream(), content_type='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

@routes.route('/progress')
def get_progress():
    return jsonify(progress=progress)

@routes.route('/generate_loss_plot')
def generate_loss_plot_route(history):
    img_path = generate_loss_plot(history)
    return send_file(img_path, mimetype='image/png')
