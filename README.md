# Dual-Forecast-Predictor

This is a Flask-based machine learning web application that predicts **house prices** and estimates **stock portfolio values** using trained models. 
The app provides users with visual predictions through interactive charts and simple input forms.

## 🔹 What It Does

### 🏠 House Price Prediction
- Predicts house price based on:
  - Median income
  - House age
  - Number of rooms and bedrooms
  - Location (latitude & longitude)
  - Population
- Shows price projection for up to 3 years
- Model used: XGBoost (trained and working)

### 📈 Stock Portfolio Forecast (⚠️ In Progress)
- Lets users select popular stocks (AAPL, MSFT, GOOG, etc.)
- Takes investment amount and prediction days
- Aims to project future value and allocation
- ⚠️ **Currently not working** due to model loading issues (under fix)

## 📁 Folder Structure
project-root/
│
├── app.py # Flask backend
├── train_house_model.py # Trains house price model
├── train_stock_model.py # Trains stock models (LSTM)
│
├── models/ # Contains .h5 and .pkl models
├── static/
│ ├── css/ # CSS files
│ └── images/ # Output graphs
│
├── templates/
│ ├── house_prediction.html
│ └── stock_prediction.html

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AmanDevNet/Dual-Forecast-Predictor
   cd Dual-Forecast-Predictor

2.Create and activate a virtual environment:
bash --- 
python -m venv venv
venv\Scripts\activate  # Windows

3.Install required packages:
bash---
pip install -r requirements.txt

4. Train the Models (First Time Only)
bash ---
python train_house_model.py
python train_stock_model.py

5.Run the app:
bash ---
python app.py

Output ---- 
House Price -----
![image](https://github.com/user-attachments/assets/238dee84-855f-4200-a8eb-28e74661fc47)
![image](https://github.com/user-attachments/assets/1d69a22a-06b9-404d-94f9-a5c81b9c6db5)

Stock Price(----Under Working----)
![image](https://github.com/user-attachments/assets/11770a1c-80dd-4e52-a225-08f4bf5d2363)
![image](https://github.com/user-attachments/assets/6aa1e620-a6c1-4878-a179-8efe8afaedd7)

👨‍💻 Author
Aman Sharma
LinkedIn :- www.linkedin.com/in/aman-sharma-842b66318
