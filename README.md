# House Price Prediction Web Application

A machine learning-powered web application that predicts house prices based on various features. The application uses a Random Forest Regressor model trained on housing data to make accurate price predictions.

## Features

- Interactive web interface for house price predictions
- Input validation for all fields
- Historical prediction tracking
- Price predictions in USD
- Responsive design
- SQLite database for storing predictions

## Technologies Used

- Python 3.x
- Flask (Web Framework)
- scikit-learn (Machine Learning)
- SQLite (Database)
- HTML/CSS (Frontend)
- Bootstrap (UI Framework)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter the house details in the form:
   - Area (in square feet)
   - Number of bedrooms
   - Number of bathrooms
   - Number of stories
   - Main road access (yes/no)
   - Guest room availability (yes/no)
   - Basement availability (yes/no)
   - Hot water heating (yes/no)
   - Air conditioning (yes/no)
   - Number of parking spaces
   - Preferred area (yes/no)

4. Click "Predict Price" to get the estimated house price in USD

## Project Structure

```
house-price-prediction/
├── app.py              # Main Flask application
├── model.joblib        # Trained machine learning model
├── scaler.joblib       # Feature scaler
├── Housing.csv         # Training dataset
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Main page template
└── house_predictions.db # SQLite database
```

## Model Details

- Algorithm: Random Forest Regressor
- Features used:
  - Area
  - Number of bedrooms
  - Number of bathrooms
  - Number of stories
  - Main road access
  - Guest room availability
  - Basement availability
  - Hot water heating
  - Air conditioning
  - Number of parking spaces
  - Preferred area

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: Housing.csv
- Flask documentation
- scikit-learn documentation 