import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load and preprocess data
def load_data(file_path):
    """
    Load the data from the provided file path and preprocess it for model training.
    Args:
    - file_path (str): Path to the uploaded dataset.
    Returns:
    - X (DataFrame): Features for model training.
    - y (Series): Target labels (Crash or Not).
    - data (DataFrame): Original dataset after preprocessing.
    """
    data = pd.read_csv(file_path, encoding='latin1')  # Handle potential encoding issues

    # Convert 'Date' to datetime format and set it as the index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Handle invalid dates
        data.set_index('Date', inplace=True)
    else:
        raise ValueError("'Date' column is missing in the dataset")

    # Calculate additional features
    data['Returns'] = data['Close'].pct_change()  # Daily returns
    data['Volatility'] = data['Returns'].rolling(window=20).std()  # 20-day rolling volatility
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day simple moving average

    # Label market crashes (if the next 20-day return is below -10%)
    data['Forward_Returns'] = data['Close'].shift(-20).pct_change(20)
    data['Crash'] = (data['Forward_Returns'] < -0.10).astype(int)  # Crash label (1 for crash, 0 for no crash)
    
    # Drop rows with missing data
    data.dropna(inplace=True)

    # Define features and target
    features = ['Returns', 'Volatility', 'SMA_50']
    target = 'Crash'

    # Create feature matrix (X) and target vector (y)
    X = data[features]
    y = data[target]

    return X, y, data

# Train and save the model
def train_model(X_train, y_train):
    """
    Train the XGBoost model using the provided training data.
    Args:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target labels.
    Returns:
    - model (XGBClassifier): Trained model.
    """
    model = XGBClassifier(random_state=42)  # Initialize the XGBoost model
    model.fit(X_train, y_train)  # Train the model
    
    # Save the trained model for future use
    joblib.dump(model, 'xgb_model.pkl')
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using classification metrics.
    Args:
    - model (XGBClassifier): Trained model.
    - X_test (DataFrame): Test features.
    - y_test (Series): True labels.
    Returns:
    - report (str): Classification report as a string.
    """
    y_pred = model.predict(X_test)  # Make predictions
    return classification_report(y_test, y_pred)  # Return the classification report

# Preprocessing: standardizing and imputing missing values
def preprocess_data(X):
    """
    Preprocess the input data by imputing missing values and scaling the features.
    Args:
    - X (DataFrame): Feature matrix.
    Returns:
    - X_scaled (DataFrame): Scaled feature matrix.
    - scaler (StandardScaler): Scaler used for feature scaling.
    - imputer (SimpleImputer): Imputer used for handling missing values.
    """
    scaler = StandardScaler()  # Initialize the scaler
    imputer = SimpleImputer(strategy='mean')  # Initialize the imputer to replace missing values with the mean

    # Impute missing values
    X_imputed = imputer.fit_transform(X)

    # Scale features
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, scaler, imputer
