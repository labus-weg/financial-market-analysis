import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix
import joblib

class MarketAnomalyDetector:
    def __init__(self, contamination=0.1):
        # Initialize base models with weights
        self.models = {
            'isolation_forest': {
                'model': IsolationForest(
                    contamination=contamination,
                    n_estimators=100,
                    random_state=42
                ),
                'weight': 0.5  # Higher weight due to effectiveness with market data
            },
            'one_class_svm': {
                'model': OneClassSVM(
                    kernel='rbf',
                    nu=contamination
                ),
                'weight': 0.3
            },
            'lof': {
                'model': LocalOutlierFactor(
                    n_neighbors=20,
                    contamination=contamination,
                    novelty=True
                ),
                'weight': 0.2
            }
        }
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Generate financial features for anomaly detection"""
        features = pd.DataFrame()
        
        # Price-based features
        if 'Close' in df.columns:
            features['returns'] = df['Close'].pct_change()
            features['rolling_vol'] = features['returns'].rolling(window=20).std()
            features['rolling_mean'] = features['returns'].rolling(window=20).mean()
        
        # Volume-based features
        if 'Volume' in df.columns:
            features['volume_change'] = df['Volume'].pct_change()
            features['volume_ma'] = df['Volume'].rolling(window=20).mean()
        
        # Technical indicators
        if 'Close' in df.columns:
            features['rsi'] = self._calculate_rsi(df['Close'])
            features['macd'] = self._calculate_macd(df['Close'])
        
        # Drop NaN values and return
        return features.dropna()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD technical indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def fit(self, X):
        """Train all models in the ensemble"""
        self.X_scaled = self.scaler.fit_transform(X)
        
        for model_dict in self.models.values():
            model_dict['model'].fit(self.X_scaled)
        
        return self
    
    def predict(self, X):
        """Generate weighted ensemble predictions"""
        X_scaled = self.scaler.transform(X)
        weighted_predictions = np.zeros(len(X))
        
        for model_dict in self.models.values():
            model_preds = model_dict['model'].predict(X_scaled)
            # Convert -1/1 predictions to 1/0 (1 for anomaly)
            model_preds = (model_preds == -1).astype(int)
            weighted_predictions += model_preds * model_dict['weight']
        
        # Return 1 (anomaly) if weighted sum exceeds 0.5
        return (weighted_predictions >= 0.5).astype(int)
    
    def anomaly_scores(self, X):
        """Calculate anomaly scores for each data point"""
        X_scaled = self.scaler.transform(X)
        scores = np.zeros(len(X))
        
        for model_dict in self.models.values():
            if hasattr(model_dict['model'], 'score_samples'):
                model_scores = -model_dict['model'].score_samples(X_scaled)
            else:
                model_scores = -model_dict['model'].decision_function(X_scaled)
            scores += model_scores * model_dict['weight']
        
        return scores
    
    def save(self, filepath):
        """Save the trained model"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a trained model"""
        return joblib.load(filepath)

def load_data(file_path):
    """Load and prepare data for anomaly detection"""
    # Handle different file types
    if hasattr(file_path, 'name'):
        file_extension = file_path.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Initialize detector and prepare features
    detector = MarketAnomalyDetector()
    X = detector.prepare_features(df)
    y = np.zeros(len(X))  # Placeholder labels
    
    return X, y, df

def train_model(X, y=None):
    """Train the anomaly detection model"""
    detector = MarketAnomalyDetector()
    detector.fit(X)
    return detector

def evaluate_model(model, X, y):
    """Evaluate the model and generate report"""
    predictions = model.predict(X)
    scores = model.anomaly_scores(X)
    
    report = f"""
    Model Evaluation Report:
    Total samples analyzed: {len(X)}
    Anomalies detected: {sum(predictions)}
    Anomaly percentage: {(sum(predictions)/len(predictions))*100:.2f}%
    Average anomaly score: {np.mean(scores):.4f}
    Max anomaly score: {np.max(scores):.4f}
    """
    return report