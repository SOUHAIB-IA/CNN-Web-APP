import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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

def train_model(config, x_train, y_train):
    # Build the model
    model = Sequential()
    
    # Add layers to the model based on the configuration
    for _ in range(config['nm_layers']):
        model.add(Dense(config['neurons'], activation=config['activation']))
    
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    # Compile the model


    # Train the model
    history = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=32, validation_split=0.2)

    return history
