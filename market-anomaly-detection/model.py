import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Preprocessing the data
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Calculating features
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Labeling crashes: if the next 20-day return is negative by 10%, it's a crash
    data['Forward_Returns'] = data['Close'].shift(-20).pct_change(20)
    data['Crash'] = (data['Forward_Returns'] < -0.10).astype(int)
    
    # Drop rows with missing data
    data.dropna(inplace=True)

    features = ['Returns', 'Volatility', 'SMA_50']
    target = 'Crash'

    X = data[features]
    y = data[target]

    return X, y, data

# Train and save the model
def train_model(X_train, y_train):
    # Initialize model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model for future use
    joblib.dump(model, 'xgb_model.pkl')
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    # Predict using the model
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

# Preprocessing - standardizing and imputing missing values
def preprocess_data(X):
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')

    # Impute missing values
    X_imputed = imputer.fit_transform(X)
    # Scale features
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, scaler, imputer
