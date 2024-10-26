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

def train_model(config, filepath):
    data = process_data(filepath)
    if data is None:
        return None, None, None  # Handle the case where data processing fails

    # Ensure columns are numeric
    print(data)
    print(config)
    data[config['columns']] = data[config['columns']].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    x = data[config['columns']]
    y = data[config['target_column']]
    if len(x) == 0 or len(y) == 0:
        raise ValueError("The dataset is empty. Please check the data source.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=config['train_size'] / 100)

    model = Sequential()
    for layer_config in config['layers']:
        model.add(Dense(layer_config['neurons'], activation=layer_config['activation']))

    if config['dropout_rate'] > 0:
        model.add(Dropout(config['dropout_rate']))


    optimizer_mapping = {
        'adam': Adam(learning_rate=config['learning_rate']),
        'sgd': SGD(learning_rate=config['learning_rate']),
        'rmsprop': RMSprop(learning_rate=config['learning_rate']),
        'adagrad': Adagrad(learning_rate=config['learning_rate']),
    }
    model.compile(optimizer=optimizer_mapping.get(config['optimizer']), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=0.2)

    y_pred = predict(model, x_test.values)  # Make sure x_test is in the right shape
    y_pred = (y_pred > 0.5).astype(int)  # Assuming binary classification
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    stats = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
    return history, model, stats


def predict(model,x_pred):
    x_pred = x_pred.reshape((-1, model.input_shape[1]))
    y_pred = model.predict(x_pred)
    return y_pred
