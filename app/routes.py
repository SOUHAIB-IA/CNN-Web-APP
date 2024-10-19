from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import os
from model.model import process_data
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
    # Assuming you have some logic to define data_preview
    data_preview = None  # Or some logic that fetches a DataFrame or appropriate data

    # Pass data_preview to the template
    return render_template('index.html', data_preview=data_preview)

# Route to handle file uploads
@routes.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains a file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    # Check if the file is allowed and save it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        flash(f'File {filename} uploaded successfully')

        # Process the uploaded file with pandas (example with CSV)
        data = pd.read_csv(file_path)

        # Send the data to the template for previewing
        data_preview = data.head(10)  # Show the first 10 rows for example

        return render_template('index.html', data_preview=data_preview)

    else:
        flash('Invalid file type. Only CSV files are allowed.')
        return redirect(request.url)
@routes.route('/get_config',methods=['post'])
def get_config():
    num_layers = request.form.get('numLayers', type=int)
    layers_config = []

    for i in range(1, num_layers + 1):
        activation = request.form.get(f'activation_layer_{i}')
        neurons_nbr = request.form.get(f'neurons_layer_{i}')
        layers_config.append({'activation': activation, 'neurons_nbr':neurons_nbr})
        
    

    return jsonify(layers_config)
