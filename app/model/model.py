import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Save model configuration
def save_config(config, config_path='config/config.json'):
    with open(config_path, 'w') as f:
        json.dump(config, f)

# Process data file
def process_data(filepath):
    try:
        data = pd.read_csv(filepath).dropna()  # Drop missing values
        return data
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Train model
def train_model(config_path, file_path):
    config = load_config(config_path)
    data = process_data(file_path)

    if config is None or data is None:
        raise ValueError("Configuration or data file is missing or invalid")

    data[config['columns']] = data[config['columns']].apply(pd.to_numeric, errors='coerce')
    data[config['target_column']] = pd.to_numeric(data[config['target_column']], errors='coerce')

    scaler = StandardScaler()
    x = scaler.fit_transform(data[config['columns']])
    y = data[config['target_column']].values

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
def predict(model, x_pred):
    x_pred = x_pred.reshape((-1, model.input_shape[1]))
    return model.predict(x_pred)
