import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from sklearn.metrics import accuracy_score

def process_data(filepath):
    """
    Function to load and process data from CSV.
    Assumes the first row of the CSV contains column headers.
    """
    try:
        data = pd.read_csv(filepath)
        # Add custom processing logic if necessary
        return data
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def prepare_data(data, config):
    """
    Function to prepare the dataset for training based on the selected features and target.
    """
    # Extract the selected features and target
    X = data[config['columns']]
    y = data[config['target_column']]

    # Encode categorical target if necessary
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Perform any additional preprocessing (e.g., scaling for numerical features)
    X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['train_test_split'] / 100, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_model(config, X_train, y_train):
    """
    Function to create, compile, and train a neural network model.
    :param config: Dictionary containing model configuration
    :param X_train: Training features
    :param y_train: Training target
    :return: history, trained model
    """
    # Create a Sequential model
    model = Sequential()
    for layer_config in config['layers']:
        model.add(Dense(layer_config['neurons'], activation=layer_config['activation'], input_shape=(X_train.shape[1],)))

    # Add dropout layer if configured
    if config['dropout_rate'] > 0:
        model.add(Dropout(config['dropout_rate']))

    # Add final output layer (binary classification assumed)
    model.add(Dense(1, activation='sigmoid'))

    # Choose the optimizer
    optimizer = None
    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=config['learning_rate'])
    elif config['optimizer'] == 'adagrad':
        optimizer = Adagrad(learning_rate=config['learning_rate'])

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=0.2)
    print(history)
    return history, model

def evaluate_model(model, X_test, y_test):
    """
    Function to evaluate the trained model on test data.
    :param model: Trained model
    :param X_test: Test features
    :param y_test: Test target
    :return: Accuracy score
    """
    # Predict on test data
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
