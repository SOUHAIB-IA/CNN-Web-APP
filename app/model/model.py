import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
    optimizer = None
    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'adagrad':
        optimizer = Adagrad(learning_rate=config['learning_rate'])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X, y, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=0.2)

    return history,model
