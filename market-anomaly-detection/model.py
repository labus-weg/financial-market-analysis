import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class MarketAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination  # Proportion of anomalies
        self.model = IsolationForest(contamination=self.contamination)
        self.scaler = StandardScaler()  # Standard scaler for preprocessing
        self.imputer = SimpleImputer(strategy='mean')  # Imputer to fill missing values

    def preprocess_data(self, data):
        """Preprocess the input data for the model"""
        # Separate numeric and non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        non_numeric_data = data.select_dtypes(exclude=[np.number])

        # Handle missing values for numeric columns
        numeric_data_imputed = pd.DataFrame(self.imputer.fit_transform(numeric_data), columns=numeric_data.columns)

        # Scale numeric data
        numeric_data_scaled = pd.DataFrame(self.scaler.fit_transform(numeric_data_imputed), columns=numeric_data.columns)

        # Combine the processed numeric data with non-numeric data (e.g., datetime columns)
        processed_data = pd.concat([numeric_data_scaled, non_numeric_data], axis=1)

        return processed_data

    def fit(self, X):
        """Fit the Isolation Forest model"""
        X_preprocessed = self.preprocess_data(X)
        self.model.fit(X_preprocessed)
        return self

    def predict(self, X):
        """Predict anomalies"""
        X_preprocessed = self.preprocess_data(X)
        anomalies = self.model.predict(X_preprocessed)
        return anomalies

    def anomaly_scores(self, X):
        """Get anomaly scores from the Isolation Forest model"""
        X_preprocessed = self.preprocess_data(X)
        scores = self.model.decision_function(X_preprocessed)
        return scores

# Function to load and preprocess the data
def load_data(file_path):
    """Load market data from a file"""
    # Read the uploaded file (either CSV or Excel)
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Automatically fill missing values with the column mean
    data = data.fillna(data.mean())

    # Return the entire dataset without filtering
    return data
