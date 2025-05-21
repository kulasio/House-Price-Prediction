from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = '5de5b409a1da2e239f9a29953396c6c16537a168290a3290cd77c3665d2d9aa4'  # Secure secret key for flash messages

# Database initialization
def init_db():
    conn = sqlite3.connect('house_predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area REAL,
            bedrooms INTEGER,
            bathrooms REAL,
            stories INTEGER,
            mainroad TEXT,
            guestroom TEXT,
            basement TEXT,
            hotwaterheating TEXT,
            airconditioning TEXT,
            parking INTEGER,
            prefarea TEXT,
            predicted_price REAL,
            submission_date TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database when app starts
init_db()

# Load and prepare the data
def create_model():
    # Read the dataset
    df = pd.read_csv('Housing.csv')
    
    # Convert price from INR to USD (assuming 1 USD = 83 INR)
    df['price_usd'] = df['price'] / 85.47
    
    # Select more features for the model
    features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
                'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']
    
    # Convert categorical variables to numeric
    df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
    df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
    df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
    df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
    df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
    df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
    
    X = df[features]
    y = df['price_usd']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Save the model and scaler
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return model, scaler

# Create the model when the app starts
model, scaler = create_model()

def validate_input(form_data):
    errors = []
    
    # Validate numeric fields
    numeric_fields = {
        'area': {'min': 100, 'max': 10000},
        'bedrooms': {'min': 1, 'max': 10},
        'bathrooms': {'min': 1, 'max': 5},
        'stories': {'min': 1, 'max': 4},
        'parking': {'min': 0, 'max': 5}
    }
    
    for field, limits in numeric_fields.items():
        try:
            value = float(form_data[field])
            if value < limits['min'] or value > limits['max']:
                errors.append(f"{field.capitalize()} must be between {limits['min']} and {limits['max']}")
        except (ValueError, KeyError):
            errors.append(f"Please enter a valid {field}")
    
    # Validate select fields
    select_fields = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for field in select_fields:
        if field not in form_data or form_data[field] not in ['yes', 'no']:
            errors.append(f"Please select a valid option for {field.replace('_', ' ')}")
    
    return errors

def save_prediction(form_data, prediction):
    conn = sqlite3.connect('house_predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (
            area, bedrooms, bathrooms, stories, mainroad, guestroom,
            basement, hotwaterheating, airconditioning, parking, prefarea,
            predicted_price, submission_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        form_data['area'],
        form_data['bedrooms'],
        form_data['bathrooms'],
        form_data['stories'],
        form_data['mainroad'],
        form_data['guestroom'],
        form_data['basement'],
        form_data['hotwaterheating'],
        form_data['airconditioning'],
        form_data['parking'],
        form_data['prefarea'],
        prediction,
        datetime.now()
    ))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Validate input
        errors = validate_input(request.form)
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('home'))
        
        try:
            # Get values from the form
            area = float(request.form['area'])
            bedrooms = float(request.form['bedrooms'])
            bathrooms = float(request.form['bathrooms'])
            stories = float(request.form['stories'])
            mainroad = request.form['mainroad']
            guestroom = request.form['guestroom']
            basement = request.form['basement']
            hotwaterheating = request.form['hotwaterheating']
            airconditioning = request.form['airconditioning']
            parking = float(request.form['parking'])
            prefarea = request.form['prefarea']
            
            # Make prediction
            features = np.array([[area, bedrooms, bathrooms, stories, 
                                1 if mainroad == 'yes' else 0,
                                1 if guestroom == 'yes' else 0,
                                1 if basement == 'yes' else 0,
                                1 if hotwaterheating == 'yes' else 0,
                                1 if airconditioning == 'yes' else 0,
                                parking,
                                1 if prefarea == 'yes' else 0]])
            features_scaled = scaler.transform(features)
            prediction = round(model.predict(features_scaled)[0], 2)
            
            # Save the prediction to database
            save_prediction(request.form, prediction)
            
        except Exception as e:
            flash('An error occurred while processing your request. Please try again.', 'error')
            return redirect(url_for('home'))
    
    return render_template('index.html', prediction=prediction, current_time=datetime.now())

@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    conn = sqlite3.connect('house_predictions.db')
    c = conn.cursor()
    
    # Get total count of predictions
    c.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = c.fetchone()[0]
    
    # Calculate total pages
    total_pages = (total_predictions + per_page - 1) // per_page
    
    # Get predictions for current page
    offset = (page - 1) * per_page
    c.execute('SELECT * FROM predictions ORDER BY submission_date DESC LIMIT ? OFFSET ?', (per_page, offset))
    predictions = c.fetchall()
    conn.close()
    
    # Convert date strings to datetime objects
    formatted_predictions = []
    for pred in predictions:
        pred_list = list(pred)
        if isinstance(pred_list[13], str):
            pred_list[13] = datetime.strptime(pred_list[13], '%Y-%m-%d %H:%M:%S.%f')
        formatted_predictions.append(pred_list)
    
    return render_template('history.html', 
                         predictions=formatted_predictions,
                         current_page=page,
                         total_pages=total_pages,
                         has_prev=page > 1,
                         has_next=page < total_pages)

if __name__ == '__main__':
    app.run(debug=True)
