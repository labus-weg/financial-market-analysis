import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class MarketAnomalyDetector:
    def __init__(self, contamination=0.1, model_type="IsolationForest"):
        """
        Initializes the anomaly detector with the specified model type and contamination level.
        Default model is IsolationForest.
        """
        self.contamination = contamination  # Proportion of anomalies
        self.model_type = model_type  # Choose between 'IsolationForest', 'OneClassSVM', 'LOF'
        self.model = self._initialize_model()
        self.scaler = StandardScaler()  # Standard scaler for preprocessing
        self.imputer = SimpleImputer(strategy='mean')  # Imputer to fill missing values

    def _initialize_model(self):
        """Initialize the model based on the model type"""
        if self.model_type == "IsolationForest":
            return IsolationForest(contamination=self.contamination, random_state=42)
        elif self.model_type == "OneClassSVM":
            return OneClassSVM(nu=self.contamination, kernel="rbf", gamma='auto')
        elif self.model_type == "LOF":
            return LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def preprocess_data(self, data):
        """Preprocess the input data for the model"""
        # Impute missing values
        data_imputed = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)
        # Scale features
        data_scaled = pd.DataFrame(self.scaler.fit_transform(data_imputed), columns=data.columns)
        return data_scaled

    def fit(self, X):
        """Fit the selected anomaly detection model"""
        X_preprocessed = self.preprocess_data(X)
        self.model.fit(X_preprocessed)
        return self

    def predict(self, X):
        """Predict anomalies using the fitted model"""
        X_preprocessed = self.preprocess_data(X)
        if isinstance(self.model, LocalOutlierFactor):
            anomalies = self.model.fit_predict(X_preprocessed)
        else:
            anomalies = self.model.predict(X_preprocessed)
        return anomalies == -1  # Anomalies are marked as -1

    def anomaly_scores(self, X):
        """Get anomaly scores from the model"""
        X_preprocessed = self.preprocess_data(X)
        if isinstance(self.model, LocalOutlierFactor):
            scores = self.model.negative_outlier_factor_  # LOF has negative outlier factors
        else:
            scores = self.model.decision_function(X_preprocessed)  # Other models use decision_function
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
