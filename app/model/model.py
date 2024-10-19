import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
def process_data(filepath):
    # Exemple: traiter un fichier CSV avec pandas
    try:
        data = pd.read_csv(filepath)
        # Fais ici le traitement souhaité sur 'data'
        return data
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def train_linear_model(data,target_column,train_size):
    X = data.drop(columns=[target_column])  # Caractéristiques
    y = data[target_column]                  # Cible

    # Sépare le jeu de données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Crée et entraîne le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Prédictions et évaluation
    return model,y_pred

def train_model(config, X, y):
    # Create a Sequential model
    model = Sequential()

    # Add layers based on the configuration
    for layer_config in config['layers']:
        model.add(Dense(layer_config['neurons'], activation=layer_config['activation']))

   

    # Add dropout layer if configured
    if config['dropout_rate'] > 0:
        model.add(Dropout(config['dropout_rate']))

    # Compile the model with the chosen optimizer
    optimizer_mapping = {
        'adam': Adam(learning_rate=config['learning_rate']),
        'sgd': SGD(learning_rate=config['learning_rate']),
        'rmsprop': RMSprop(learning_rate=config['learning_rate']),
        'adagrad': Adagrad(learning_rate=config['learning_rate']),
    }
    model.compile(optimizer=optimizer_mapping.get(config['optimizer']), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X, y, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=0.2)

    return history,model

def predict(model,x_pred):
    x_pred = x_pred.reshape((-1, model.input_shape[1]))
    y_pred = model.predict(x_pred)
    return y_pred

def evaluate_model(model,x,y_true):
    # Calculate various metrics
    y_pred = predict(model,x)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
