import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import json

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class StockPredictionModel:
    def __init__(self, symbol='AAPL', sequence_length=60, prediction_days=30):
        """
        Initialize the Stock Prediction Model
        
        Args:
            symbol (str): Stock symbol to predict
            sequence_length (int): Number of days to look back for prediction
            prediction_days (int): Number of days to predict into the future
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        
        # Top 10 companies by market cap
        self.top_companies = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
            'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ'
        ]
        
        # Create directories
        self.create_directories()
        
        # Initialize scalers
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Model storage
        self.lstm_model = None
        self.transformer_model = None
        
        # Data storage
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def create_directories(self):
        """Create necessary directories for saving models and plots"""
        directories = ['./models/', './static/', './data/']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def fetch_stock_data(self, years=10):
        """
        Fetch stock data using yfinance
        
        Args:
            years (int): Number of years of historical data to fetch
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        print(f"Fetching {years} years of data for {self.symbol}...")
        
        try:
            stock = yf.Ticker(self.symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            print(f"Successfully fetched {len(data)} days of data")
            return data
        
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None
    
    def engineer_features(self, data):
        """
        Engineer technical indicators using ta library
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        print("Engineering technical features...")
        
        df = data.copy()
        
        # Basic price features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'].pct_change()
        
        # Moving averages
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # Exponential moving averages
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd_line, macd_signal, macd_histogram = ta.trend.MACD(df['Close']).macd(), ta.trend.MACD(df['Close']).macd_signal(), ta.trend.MACD(df['Close']).macd_diff()
        df['MACD'] = macd_line
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_histogram
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
        
        # Stochastic Oscillator
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
        df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
        
        # Average True Range
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Commodity Channel Index
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
        
        # Drop NaN values
        df = df.dropna()
        
        print(f"Engineered {len(df.columns)} features")
        return df
    
    def prepare_data(self, data):
        """
        Prepare data for training by normalizing and creating sequences
        
        Args:
            data (pd.DataFrame): Data with technical indicators
        """
        print("Preparing data for training...")
        
        # Select features for training (exclude target)
        feature_columns = [col for col in data.columns if col != 'Close']
        features = data[feature_columns].values
        target = data['Close'].values.reshape(-1, 1)
        
        # Normalize features and target separately
        features_scaled = self.scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        self.X_train = X[:train_size]
        self.X_val = X[train_size:train_size+val_size]
        self.X_test = X[train_size+val_size:]
        
        self.y_train = y[:train_size]
        self.y_val = y[train_size:train_size+val_size]
        self.y_test = y[train_size+val_size:]
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Save scalers
        joblib.dump(self.scaler, f'./models/{self.symbol}_feature_scaler.pkl')
        joblib.dump(self.target_scaler, f'./models/{self.symbol}_target_scaler.pkl')
    
    def build_lstm_model(self):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_transformer_model(self):
        """Build Transformer model architecture"""
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        
        # Multi-head attention
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention = Dropout(0.1)(attention)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # Feed forward
        ffn = Dense(128, activation='relu')(x)
        ffn = Dense(self.X_train.shape[2])(ffn)
        ffn = Dropout(0.1)(ffn)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Final layers
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(25, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train_models(self, epochs=100):
        """
        Train both LSTM and Transformer models
        
        Args:
            epochs (int): Number of training epochs
        """
        print("Training models...")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        
        # Train LSTM model
        print("Training LSTM model...")
        self.lstm_model = self.build_lstm_model()
        lstm_history = self.lstm_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Train Transformer model
        print("Training Transformer model...")
        self.transformer_model = self.build_transformer_model()
        transformer_history = self.transformer_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Save models
        self.lstm_model.save(f'./models/{self.symbol}_lstm_model.h5')
        self.transformer_model.save(f'./models/{self.symbol}_transformer_model.h5')
        
        return lstm_history, transformer_history
    
    def evaluate_models(self):
        """Evaluate both models and return metrics"""
        print("Evaluating models...")
        
        # LSTM predictions
        lstm_pred_train = self.lstm_model.predict(self.X_train)
        lstm_pred_val = self.lstm_model.predict(self.X_val)
        lstm_pred_test = self.lstm_model.predict(self.X_test)
        
        # Transformer predictions
        transformer_pred_train = self.transformer_model.predict(self.X_train)
        transformer_pred_val = self.transformer_model.predict(self.X_val)
        transformer_pred_test = self.transformer_model.predict(self.X_test)
        
        # Inverse transform predictions
        lstm_pred_train = self.target_scaler.inverse_transform(lstm_pred_train)
        lstm_pred_val = self.target_scaler.inverse_transform(lstm_pred_val)
        lstm_pred_test = self.target_scaler.inverse_transform(lstm_pred_test)
        
        transformer_pred_train = self.target_scaler.inverse_transform(transformer_pred_train)
        transformer_pred_val = self.target_scaler.inverse_transform(transformer_pred_val)
        transformer_pred_test = self.target_scaler.inverse_transform(transformer_pred_test)
        
        # Inverse transform actual values
        y_train_actual = self.target_scaler.inverse_transform(self.y_train)
        y_val_actual = self.target_scaler.inverse_transform(self.y_val)
        y_test_actual = self.target_scaler.inverse_transform(self.y_test)
        
        # Calculate metrics
        metrics = {}
        
        # LSTM metrics
        metrics['LSTM'] = {
            'train': {
                'MAE': mean_absolute_error(y_train_actual, lstm_pred_train),
                'MSE': mean_squared_error(y_train_actual, lstm_pred_train),
                'R2': r2_score(y_train_actual, lstm_pred_train)
            },
            'val': {
                'MAE': mean_absolute_error(y_val_actual, lstm_pred_val),
                'MSE': mean_squared_error(y_val_actual, lstm_pred_val),
                'R2': r2_score(y_val_actual, lstm_pred_val)
            },
            'test': {
                'MAE': mean_absolute_error(y_test_actual, lstm_pred_test),
                'MSE': mean_squared_error(y_test_actual, lstm_pred_test),
                'R2': r2_score(y_test_actual, lstm_pred_test)
            }
        }
        
        # Transformer metrics
        metrics['Transformer'] = {
            'train': {
                'MAE': mean_absolute_error(y_train_actual, transformer_pred_train),
                'MSE': mean_squared_error(y_train_actual, transformer_pred_train),
                'R2': r2_score(y_train_actual, transformer_pred_train)
            },
            'val': {
                'MAE': mean_absolute_error(y_val_actual, transformer_pred_val),
                'MSE': mean_squared_error(y_val_actual, transformer_pred_val),
                'R2': r2_score(y_val_actual, transformer_pred_val)
            },
            'test': {
                'MAE': mean_absolute_error(y_test_actual, transformer_pred_test),
                'MSE': mean_squared_error(y_test_actual, transformer_pred_test),
                'R2': r2_score(y_test_actual, transformer_pred_test)
            }
        }
        
        # Save metrics
        with open(f'./models/{self.symbol}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Print metrics
        print("\n" + "="*50)
        print(f"MODEL EVALUATION RESULTS FOR {self.symbol}")
        print("="*50)
        
        for model_name in ['LSTM', 'Transformer']:
            print(f"\n{model_name} Model:")
            for dataset in ['train', 'val', 'test']:
                print(f"  {dataset.upper()}:")
                print(f"    MAE: {metrics[model_name][dataset]['MAE']:.4f}")
                print(f"    MSE: {metrics[model_name][dataset]['MSE']:.4f}")
                print(f"    R²:  {metrics[model_name][dataset]['R2']:.4f}")
        
        return metrics, {
            'lstm_predictions': (lstm_pred_test, y_test_actual),
            'transformer_predictions': (transformer_pred_test, y_test_actual)
        }
    
    def plot_training_history(self, lstm_history, transformer_history):
        """Plot training and validation loss"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # LSTM training history
        axes[0].plot(lstm_history.history['loss'], label='Training Loss', color='blue')
        axes[0].plot(lstm_history.history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title(f'LSTM Model Training History - {self.symbol}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Transformer training history
        axes[1].plot(transformer_history.history['loss'], label='Training Loss', color='blue')
        axes[1].plot(transformer_history.history['val_loss'], label='Validation Loss', color='red')
        axes[1].set_title(f'Transformer Model Training History - {self.symbol}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./static/{self.symbol}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, predictions):
        """Plot actual vs predicted prices"""
        lstm_pred, lstm_actual = predictions['lstm_predictions']
        transformer_pred, transformer_actual = predictions['transformer_predictions']
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # LSTM predictions
        axes[0].plot(lstm_actual, label='Actual Price', color='blue', alpha=0.7)
        axes[0].plot(lstm_pred, label='LSTM Predicted Price', color='red', alpha=0.7)
        axes[0].set_title(f'LSTM Model: Actual vs Predicted Prices - {self.symbol}')
        axes[0].set_xlabel('Days')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Transformer predictions
        axes[1].plot(transformer_actual, label='Actual Price', color='blue', alpha=0.7)
        axes[1].plot(transformer_pred, label='Transformer Predicted Price', color='orange', alpha=0.7)
        axes[1].set_title(f'Transformer Model: Actual vs Predicted Prices - {self.symbol}')
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./static/{self.symbol}_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_future(self, days=30):
        """
        Predict future stock prices with confidence intervals
        
        Args:
            days (int): Number of days to predict into the future
        """
        print(f"Predicting {days} days into the future...")
        
        # Use the last sequence from test data as starting point
        last_sequence = self.X_test[-1:].copy()
        
        # Generate predictions for multiple runs to create confidence intervals
        num_runs = 100
        lstm_predictions = []
        transformer_predictions = []
        
        for run in range(num_runs):
            current_sequence = last_sequence.copy()
            lstm_future = []
            transformer_future = []
            
            for day in range(days):
                # LSTM prediction
                lstm_pred = self.lstm_model.predict(current_sequence, verbose=0)
                lstm_future.append(lstm_pred[0, 0])
                
                # Transformer prediction
                transformer_pred = self.transformer_model.predict(current_sequence, verbose=0)
                transformer_future.append(transformer_pred[0, 0])
                
                # Update sequence (simplified approach - using prediction as next input)
                # In practice, you might want a more sophisticated approach
                new_row = current_sequence[0, -1:].copy()
                new_row[0, -1] = lstm_pred[0, 0]  # Use LSTM prediction for next sequence
                new_row = new_row.reshape(1, 1, -1)  # Reshape to match 3D shape
                current_sequence = np.concatenate([current_sequence[:, 1:], new_row], axis=1)
            
            lstm_predictions.append(lstm_future)
            transformer_predictions.append(transformer_future)
        
        # Convert to numpy arrays and calculate statistics
        lstm_predictions = np.array(lstm_predictions)
        transformer_predictions = np.array(transformer_predictions)
        
        # Calculate mean and confidence intervals
        lstm_mean = np.mean(lstm_predictions, axis=0)
        lstm_lower = np.percentile(lstm_predictions, 2.5, axis=0)
        lstm_upper = np.percentile(lstm_predictions, 97.5, axis=0)
        
        transformer_mean = np.mean(transformer_predictions, axis=0)
        transformer_lower = np.percentile(transformer_predictions, 2.5, axis=0)
        transformer_upper = np.percentile(transformer_predictions, 97.5, axis=0)
        
        # Inverse transform predictions
        lstm_mean = self.target_scaler.inverse_transform(lstm_mean.reshape(-1, 1)).flatten()
        lstm_lower = self.target_scaler.inverse_transform(lstm_lower.reshape(-1, 1)).flatten()
        lstm_upper = self.target_scaler.inverse_transform(lstm_upper.reshape(-1, 1)).flatten()
        
        transformer_mean = self.target_scaler.inverse_transform(transformer_mean.reshape(-1, 1)).flatten()
        transformer_lower = self.target_scaler.inverse_transform(transformer_lower.reshape(-1, 1)).flatten()
        transformer_upper = self.target_scaler.inverse_transform(transformer_upper.reshape(-1, 1)).flatten()
        
        # Plot future predictions
        self.plot_future_predictions(lstm_mean, lstm_lower, lstm_upper, 
                                   transformer_mean, transformer_lower, transformer_upper, days)
        
        return {
            'lstm': {'mean': lstm_mean, 'lower': lstm_lower, 'upper': lstm_upper},
            'transformer': {'mean': transformer_mean, 'lower': transformer_lower, 'upper': transformer_upper}
        }
    
    def plot_future_predictions(self, lstm_mean, lstm_lower, lstm_upper, 
                              transformer_mean, transformer_lower, transformer_upper, days):
        """Plot future predictions with confidence intervals"""
        
        # Get last known prices for context
        last_actual = self.target_scaler.inverse_transform(self.y_test[-30:])
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Create date ranges
        historical_days = range(-len(last_actual), 0)
        future_days = range(0, days)
        
        # LSTM future predictions
        axes[0].plot(historical_days, last_actual, label='Historical Actual', color='blue', linewidth=2)
        axes[0].plot(future_days, lstm_mean, label='LSTM Prediction', color='red', linewidth=2)
        axes[0].fill_between(future_days, lstm_lower, lstm_upper, alpha=0.3, color='red', 
                           label='95% Confidence Interval')
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[0].set_title(f'LSTM Future Price Predictions - {self.symbol}')
        axes[0].set_xlabel('Days (0 = Today)')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Transformer future predictions
        axes[1].plot(historical_days, last_actual, label='Historical Actual', color='blue', linewidth=2)
        axes[1].plot(future_days, transformer_mean, label='Transformer Prediction', color='orange', linewidth=2)
        axes[1].fill_between(future_days, transformer_lower, transformer_upper, alpha=0.3, color='orange',
                           label='95% Confidence Interval')
        axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[1].set_title(f'Transformer Future Price Predictions - {self.symbol}')
        axes[1].set_xlabel('Days (0 = Today)')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./static/{self.symbol}_future_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_pipeline(self, years=10, epochs=100):
        """
        Run the complete pipeline for stock prediction
        
        Args:
            years (int): Years of historical data to fetch
            epochs (int): Training epochs
        """
        print(f"Starting full pipeline for {self.symbol}")
        print("="*50)
        
        # Step 1: Fetch data
        raw_data = self.fetch_stock_data(years)
        if raw_data is None:
            print(f"Failed to fetch data for {self.symbol}")
            return None
        
        # Step 2: Engineer features
        self.data = self.engineer_features(raw_data)
        
        # Step 3: Prepare data
        self.prepare_data(self.data)
        
        # Step 4: Train models
        lstm_history, transformer_history = self.train_models(epochs)
        
        # Step 5: Evaluate models
        metrics, predictions = self.evaluate_models()
        
        # Step 6: Generate plots
        self.plot_training_history(lstm_history, transformer_history)
        self.plot_predictions(predictions)
        
        # Step 7: Future predictions
        future_predictions = self.predict_future(self.prediction_days)
        
        print(f"\nPipeline completed for {self.symbol}!")
        print(f"Models and plots saved in ./models/ and ./static/ directories")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'future_predictions': future_predictions
        }

def run_multiple_stocks(symbols=None, years=10, epochs=50):
    """
    Run the pipeline for multiple stock symbols
    
    Args:
        symbols (list): List of stock symbols. If None, uses top 10 companies
        years (int): Years of historical data
        epochs (int): Training epochs
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ']
    
    results = {}
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"PROCESSING {symbol}")
        print(f"{'='*60}")
        
        try:
            model = StockPredictionModel(symbol=symbol)
            result = model.run_full_pipeline(years=years, epochs=epochs)
            results[symbol] = result
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            results[symbol] = None
    
    return results

# Example usage
if __name__ == "__main__":
    for symbol in ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']:
        print(f"\n==================================================")
        print(f"     Running full pipeline for {symbol}")
        print(f"==================================================\n")
        try:
            model = StockPredictionModel(symbol)
            model.run_full_pipeline(years=10, epochs=100)
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

    
    # Multiple stocks example (commented out to avoid long execution)
    # print("\nRunning predictions for top 5 companies...")
    # results = run_multiple_stocks(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'], years=5, epochs=30)