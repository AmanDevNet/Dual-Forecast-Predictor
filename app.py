import os
import base64
import logging
from io import BytesIO
from datetime import datetime, timedelta
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
import tensorflow.keras.backend as K

# Define custom Attention layer
class Attention(Layer):
    def __init__(self, units=64, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.W = None
        self.V = None

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                shape=(input_shape[-1], self.units),
                                initializer='glorot_uniform',
                                trainable=True)
        self.V = self.add_weight(name='attention_v',
                                 shape=(self.units, 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Score calculation
        score = K.tanh(K.dot(x, self.W))
        # Attention weights
        attention_weights = K.softmax(K.dot(score, self.V), axis=1)
        # Context vector
        context_vector = attention_weights * x
        context_vector = K.sum(context_vector, axis=1)
        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Global variables for models and scalers
house_model = None
house_scaler = None
stock_models = {}  # Changed from stock_model/stock_scaler to dictionary

def load_house_models():
    """Load house price prediction models and scaler"""
    global house_model, house_scaler
    try:
        house_model = joblib.load('models/house_model_xgb.pkl')
        house_scaler = joblib.load('models/house_scaler.pkl')
        logger.info("[DEBUG] House price prediction models loaded successfully")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load house models: {str(e)}")
        raise

def load_stock_models():
    global stock_models
    
    try:
        supported_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META',
                           'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
        missing_files = []

        custom_objects = {
            'Attention': Attention,
            'MultiHeadAttention': MultiHeadAttention,
            'LayerNormalization': LayerNormalization
        }

        for ticker in supported_tickers:
            try:
                model_path = os.path.join('models', f'{ticker}_transformer_model.h5')
                feature_scaler_path = os.path.join('models', f'{ticker}_feature_scaler.pkl')
                target_scaler_path = os.path.join('models', f'{ticker}_target_scaler.pkl')

                if not all(os.path.exists(p) for p in [model_path, feature_scaler_path, target_scaler_path]):
                    missing_files.append(ticker)
                    logger.warning(f"Missing files for {ticker}")
                    continue

                # Load model with custom objects
                model = load_model(model_path, custom_objects=custom_objects)
                f_scaler = joblib.load(feature_scaler_path)
                t_scaler = joblib.load(target_scaler_path)

                stock_models[ticker] = {
                    'model': model,
                    'feature_scaler': f_scaler,
                    'target_scaler': t_scaler
                }
                logger.info(f"Successfully loaded model for {ticker}")
                
            except Exception as e:
                logger.error(f"Error loading {ticker}: {str(e)}", exc_info=True)
                continue

        if missing_files:
            logger.warning(f"Missing files for tickers: {', '.join(missing_files)}")

        return len(stock_models) > 0

    except Exception as e:
        logger.error(f"Failed to load stock models: {str(e)}", exc_info=True)
        return False


def create_plot_base64(fig):
    """Convert matplotlib figure to base64 encoded image"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

@app.route('/')
def home():
    return render_template('house_prediction.html')

@app.route('/house')
def house_prediction():
    return render_template('house_prediction.html')

@app.route('/stock')
def stock_prediction():
    return render_template('stock_prediction.html')

@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    try:
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['median_income']),
            float(data['house_age']),
            float(data['avg_rooms']),
            float(data['avg_bedrooms']),
            float(data['population']),
            float(data['latitude']),
            float(data['longitude'])
        ]
        
        # Feature engineering
        median_income_squared = features[0] ** 2
        rooms_per_bedroom = features[2] / features[3] if features[3] != 0 else features[2]
        population_density = features[4] / (abs(features[5] * features[6]) + 1)
        
        # Add engineered features
        features.extend([median_income_squared, rooms_per_bedroom, population_density])
        
        # Scale features
        features_scaled = house_scaler.transform([features])
        
        # Predict
        prediction = float(round(house_model.predict(features_scaled)[0] * 100000, 2))
        
        # Generate projection
        projection_years = int(data.get('projection_years', 3))
        projections = generate_house_projection(features_scaled, projection_years)
        projection_plot = plot_house_projection(projections)
        
        return jsonify({
            'predicted_price': float(round(prediction, 2)), 
            'projection_plot': projection_plot,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"[ERROR] House price prediction failed: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 400

def generate_house_projection(features, years):
    """Generate house price projections for given years"""
    projections = []
    current_features = features.copy()
    
    for year in range(1, years + 1):
        pred = house_model.predict(current_features)[0]
        projections.append(float(round(pred * 100000, 2)))
        
        # Modify features slightly for next year's projection
        current_features[0][0] *= 1.02  # Median income grows 2%
        current_features[0][1] += 1     # House age increases
        current_features[0][4] *= 1.01  # Population grows 1%
    
    return projections

def plot_house_projection(projections):
    """Create a house price projection plot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Determine line color based on trend
    if len(projections) > 1:
        if projections[-1] > projections[0] * 1.05:  # 5% growth threshold
            line_color = 'green'
        elif projections[-1] < projections[0] * 0.95:  # 5% decline threshold
            line_color = 'red'
        else:
            line_color = 'orange'
    else:
        line_color = 'blue'
    
    ax.plot(range(len(projections)), projections, color=line_color, marker='o')
    ax.set_title('House Price Projection')
    ax.set_xlabel('Years')
    ax.set_ylabel('Predicted Price')
    ax.grid(True)
    
    return create_plot_base64(fig)

@app.route('/predict_stock_portfolio', methods=['POST'])
def predict_stock_portfolio():
    try:
        data = request.get_json()
        
        # Validate input data
        if not data:
            return jsonify({'error': 'No input data provided', 'status': 'error'}), 400
            
        tickers = data.get('tickers', [])
        if not tickers or not isinstance(tickers, list):
            return jsonify({'error': 'Invalid or missing tickers', 'status': 'error'}), 400
            
        try:
            investment = float(data.get('investment', 0))
            if investment < 100 or investment > 1000000:
                return jsonify({'error': 'Investment must be between $100 and $1,000,000', 'status': 'error'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid investment amount', 'status': 'error'}), 400
            
        try:
            days = int(data.get('days', 0))
            if days < 7 or days > 365:
                return jsonify({'error': 'Prediction period must be 7-365 days', 'status': 'error'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid prediction period', 'status': 'error'}), 400

        # Filter to only tickers we have models for
        valid_tickers = [t for t in tickers if t in stock_models]
        if not valid_tickers:
            available_tickers = list(stock_models.keys())
            return jsonify({
                'error': 'No valid models for selected tickers',
                'message': f'Available tickers: {", ".join(available_tickers)}',
                'available_tickers': available_tickers,
                'status': 'error'
            }), 400

        # Prepare portfolio results
        portfolio_results = {
            'dates': [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days+1)],
            'values': np.zeros(days).tolist(),
            'allocation': {},
            'final_value': investment,
            'return_percentage': 0
        }
        
        allocation_per_stock = investment / len(valid_tickers)
        
        # Process each valid ticker
        for ticker in valid_tickers:
            try:
                model_info = stock_models[ticker]
                
                # Download recent data (last 3 years)
                df = yf.download(ticker, period='3y', progress=False)
                if df.empty:
                    logger.warning(f"No data downloaded for {ticker}")
                    continue
                    
                # Calculate technical indicators
                df = calculate_technical_indicators(df)
                if df.empty:
                    logger.warning(f"No indicators calculated for {ticker}")
                    continue
                
                # Prepare features
                features = df[['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 
                              'EMA_50', 'BB_Upper', 'BB_Lower', 'OBV']].values
                features_scaled = model_info['feature_scaler'].transform(features)
                
                # Make prediction (using last 60 days as input)
                X = np.array([features_scaled[-60:]])  # Shape: (1, 60, num_features)
                pred = model_info['model'].predict(X)
                pred_prices = model_info['target_scaler'].inverse_transform(pred)
                
                # Calculate returns
                start_price = df['Close'].iloc[-1]
                if start_price <= 0:
                    logger.error(f"Invalid start price for {ticker}")
                    continue
                    
                returns = (pred_prices.flatten()[:days] / start_price) * allocation_per_stock
                
                # Add to portfolio results
                portfolio_results['values'] = [sum(x) for x in zip(portfolio_results['values'], returns)]
                portfolio_results['allocation'][ticker] = round(100/len(valid_tickers), 2)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

        # Calculate final metrics
        if portfolio_results['values']:
            portfolio_results['final_value'] = round(portfolio_results['values'][-1], 2)
            portfolio_results['return_percentage'] = round(
                ((portfolio_results['final_value'] - investment) / investment * 100), 2)

        # Generate plot
        try:
            plot_data = plot_portfolio_projection(portfolio_results['dates'], portfolio_results['values'])
        except Exception as e:
            logger.error(f"Plot generation failed: {str(e)}")
            plot_data = None

        return jsonify({
            'status': 'success',
            'projection_plot': plot_data,
            'initial_investment': investment,
            'final_value': portfolio_results['final_value'],
            'return_percentage': portfolio_results['return_percentage'],
            'allocation': portfolio_results['allocation'],
            'projection_dates': portfolio_results['dates'],
            'used_tickers': valid_tickers
        })
        
    except Exception as e:
        logger.error(f"Unexpected prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred',
            'message': str(e),
            'status': 'error'
        }), 500

def calculate_technical_indicators(df):
    """Calculate all technical indicators matching train_stock_model.py"""
    try:
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
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Indicator calculation failed: {str(e)}")
        raise

def plot_portfolio_projection(dates, values):
    """Create professional portfolio projection plot"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Main plot line
    line_color = '#4e79a7'
    if len(values) > 1:
        line_color = 'green' if values[-1] > values[0] else 'red'
    
    ax.plot(dates, values, color=line_color, linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(dates, values, min(values)*0.98, color=line_color, alpha=0.1)
    
    # Styling
    ax.set_title('Portfolio Value Projection', pad=20, fontsize=14)
    ax.set_xlabel('Date', labelpad=10)
    ax.set_ylabel('Value ($)', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value markers
    for i, (date, val) in enumerate(zip(dates, values)):
        if i == 0 or i == len(values)-1 or i == len(values)//2:
            ax.annotate(f'${val:,.0f}', (date, val), 
                       textcoords="offset points", xytext=(0,10), ha='center')
    
    # Clean layout
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    return create_plot_base64(fig)

if __name__ == '__main__':
    print("[INFO] Loading models...")
    if not load_stock_models():
        print("[WARNING] Some stock models failed to load")
    load_house_models()
    print("[INFO] Starting Flask app...")
    app.run(debug=True)