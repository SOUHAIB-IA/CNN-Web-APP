import json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model configuration
def load_config(config_path='config/config.json'):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None
def clean_currency(value):
    return float(value.replace(',', ''))
# Save model configuration
def save_config(config, config_path='config/config.json'):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# Process data file
def process_data(filepath):
    try:
        data = pd.read_csv(filepath,converters={'Unit Cost': clean_currency,'Unit Price': clean_currency}).dropna()  # Drop missing values
        return data
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Train model
def train_model(config_path, file_path):
    config = load_config(config_path)
    data = process_data(file_path)
    print(data[config['categorical_columns']])
    if config is None or data is None:
        raise ValueError("Configuration or data file is missing or invalid")
    if data.isnull().any().any():
        print("Des valeurs manquantes détectées. Remplissage ou suppression nécessaire.")
        data = data.fillna(0)
    data[config['columns']] = data[config['columns']].apply(pd.to_numeric, errors='coerce')
    if 'categorical_columns' in config and config['categorical_columns']:
    # Check for missing values in categorical columns first
        if data[config['categorical_columns']].isnull().any().any():
            print("Des valeurs manquantes détectées dans les colonnes catégorielles. Remplissage ou suppression nécessaire.")
            data[config['categorical_columns']] = data[config['categorical_columns']].fillna('unknown')  # Fill with a placeholder    data[config['target_column']] = pd.to_numeric(data[config['target_column']], errors='coerce')
    if 'categorical_columns' in config and config['categorical_columns']:
        print("Colonnes catégorielles before astype:", data[config['categorical_columns']])
        data[config['categorical_columns']] = data[config['categorical_columns']].astype(str)
        print("Colonnes catégorielles:", data[config['categorical_columns']])
        missing_columns = [col for col in config['categorical_columns'] if col not in data.columns]
        print(missing_columns)
        if missing_columns:
            print(f"Colonnes catégorielles manquantes: {missing_columns}")
        else:
            print('before encoder')
            try:
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                print(data[config['categorical_columns']])
                encoded_features = encoder.fit_transform(data[config['categorical_columns']])
                print("Encoded features shape:", encoded_features.shape)
            except Exception as e:
                print(f"Error during encoding: {e}")
                return  # Exit if there's an error
            print('encoder')
            encoded_feature_names = encoder.get_feature_names_out(config['categorical_columns'])
            print("Encoded features shape:", encoded_feature_names)
            encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
            data = pd.concat([data.drop(columns=config['categorical_columns']), encoded_df], axis=1)
    scaler = StandardScaler()
    x = scaler.fit_transform(data[config['columns']])
    y = data[config['target_column']].values
    if np.any(np.isnan(y)):
        print("Des valeurs NaN détectées dans y.")
    print(x)
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=config['train_size'] / 100)
    model = Sequential()
    for layer_config in config['layers']:
        model.add(Dense(layer_config['neurons'], activation=layer_config['activation']))
    model.add(Dense(1))  # Single output for regression
    optimizer = {
        'adam': Adam(config['learning_rate']),
        'sgd': SGD(config['learning_rate']),
        'rmsprop': RMSprop(config['learning_rate']),
        'adagrad': Adagrad(config['learning_rate']),
    }[config['optimizer']]
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    history = model.fit(
        x_train, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2
    )
    y_pred = model.predict(x_test).flatten()
    accuracy = accuracy_score(y_test.round(), y_pred.round())
    report = classification_report(y_test.round(), y_pred.round())
    cm = confusion_matrix(y_test.round(), y_pred.round())
    stats = {'accuracy': accuracy, 'classification_report': report, 'confusion_matrix': cm}

    model.save('model.h5')
    pd.DataFrame(history.history).to_csv('training_history.csv', index=False)
    with open('stats.txt', 'w') as f:
        f.write(f"Accuracy: {stats['accuracy']}\n")
        f.write(stats['classification_report'])
        f.write(f"\nConfusion Matrix:\n{stats['confusion_matrix']}")
    return history, model, stats

# Predict with the trained model
from tensorflow.keras.models import load_model

def load_config(config_path):
    """Load the model configuration from a JSON file."""
    import json
    with open(config_path, 'r') as file:
        return json.load(file)

def model_predict(input_features, config_path='config/config.json', model_path='model.h5'):
    # Load the model configuration
    config = load_config(config_path)
    if config is None:
        raise ValueError("Configuration file not found or invalid")

    # Load the trained model
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Create input DataFrame for preprocessing
    # Combine numerical and categorical features
    columns = config['columns'] + config['categorical_columns']  # Combine both lists
    input_data = pd.DataFrame([input_features], columns=columns)

    # One-hot encode categorical features if any
    if 'categorical_columns' in config and config['categorical_columns']:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_features = encoder.fit_transform(input_data[config['categorical_columns']])
        encoded_feature_names = encoder.get_feature_names_out(config['categorical_columns'])
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=input_data.index)
        input_data = pd.concat([input_data.drop(columns=config['categorical_columns']), encoded_df], axis=1)

    # Scale input features
    scaler = StandardScaler()
    # Assuming scaler has been fitted before (in training phase)
    scaled_features = scaler.fit_transform(input_data)

    # Make predictions using the trained model
    prediction = model.predict(scaled_features).flatten()
    return float(prediction[0])  # Convert to float before returning


def generate_loss_plot(history_data):

    if history_data is None or 'loss' not in history_data or 'val_loss' not in history_data:
        raise ValueError("Invalid history data: Loss values are missing.")
    loss = history_data['loss']
    val_loss = history_data['val_loss']
    # Créez le graphique
    plt.figure(figsize=(10, 10))
    plt.plot(loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    print("saving------------")
    # Sauvegardez l'image dans un fichier
    img_dir = os.path.join('app','static', 'loss_plot.png')  # Adjust this path if necessary
    plt.savefig(img_dir)
    plt.close()
    
    return 'loss_plot.png'