# House Price Prediction - Technical Documentation

## Project Overview
This project implements a machine learning-based house price prediction system using a web interface. The system uses historical housing data to train a model that can predict house prices based on various features.

## Data Analysis and Preprocessing

### Dataset
- Source: Housing.csv
- Size: 547 entries
- Features: 11 input features and 1 target variable (price)
- Data Type: Mixed (numerical and categorical)

### Feature Engineering
1. **Numerical Features**:
   - Area (square feet)
   - Bedrooms (count)
   - Bathrooms (count)
   - Stories (count)
   - Parking (count)

2. **Categorical Features** (converted to binary):
   - Mainroad (yes/no)
   - Guestroom (yes/no)
   - Basement (yes/no)
   - Hotwaterheating (yes/no)
   - Airconditioning (yes/no)
   - Prefarea (yes/no)

3. **Target Variable**:
   - Price (converted from INR to USD using exchange rate 1 USD = 85.47 INR)

### Data Preprocessing Steps
1. Categorical variables are encoded using binary mapping (yes=1, no=0)
2. Numerical features are scaled using StandardScaler
3. No missing values handling required as the dataset is complete
4. No outlier removal as the data represents real-world housing scenarios

## Machine Learning Model

### Model Selection
- Algorithm: Random Forest Regressor
- Rationale for selection:
  - Handles both numerical and categorical features well
  - Robust to outliers
  - Provides feature importance
  - Good performance on regression tasks
  - Less prone to overfitting compared to single decision trees

### Model Configuration
```python
RandomForestRegressor(
    n_estimators=100,  # Number of trees in the forest
    random_state=42    # For reproducibility
)
```

### Model Training Process
1. Data splitting: Using cross-validation (5 folds)
2. Feature scaling: StandardScaler for numerical features
3. Training: Using the entire dataset as it's relatively small
4. Model evaluation: Using cross-validation scores

### Model Performance
- Cross-validation is implemented using 5 folds
- Performance metrics:
  - Mean squared error
  - R-squared score
  - Cross-validation scores for model stability assessment

## Web Application Implementation

### Backend (Flask)
1. **Routes**:
   - `/`: Main prediction page
   - `/history`: Historical predictions view

2. **Database**:
   - SQLite database (house_predictions.db)
   - Schema:
     ```sql
     CREATE TABLE predictions (
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
     ```

3. **Input Validation**:
   - Numeric range validation
   - Categorical value validation
   - Error handling and user feedback

### Frontend
- HTML/CSS with Bootstrap for responsive design
- Form validation
- Dynamic price display
- Historical predictions table

## Model Persistence
- Trained model saved as: `model.joblib`
- Feature scaler saved as: `scaler.joblib`
- Both files are loaded at application startup

## API Endpoints

### Prediction Endpoint
- Method: POST
- Input: Form data with house features
- Output: Predicted price in USD
- Validation: Server-side validation of all inputs

### History Endpoint
- Method: GET
- Output: Paginated list of previous predictions
- Features: 10 predictions per page

## Error Handling
1. **Input Validation**:
   - Range checks for numerical inputs
   - Valid option checks for categorical inputs
   - Flash messages for user feedback

2. **Model Prediction**:
   - Exception handling for prediction errors
   - Graceful error messages to users

## Security Considerations
1. **Input Sanitization**:
   - All user inputs are validated
   - SQL injection prevention through parameterized queries

2. **Secret Key**:
   - Flask secret key for session management
   - Flash message security

## Performance Optimization
1. **Model Loading**:
   - Model loaded once at startup
   - Cached for subsequent predictions

2. **Database**:
   - Indexed primary key
   - Efficient query structure

## Future Improvements
1. **Model Enhancements**:
   - Feature importance visualization
   - Model retraining capability
   - Additional algorithms comparison

2. **Application Features**:
   - User authentication
   - Export predictions to CSV
   - Advanced filtering in history view
   - API documentation
   - Unit tests implementation

3. **Technical Improvements**:
   - Docker containerization
   - CI/CD pipeline
   - Automated testing
   - Performance monitoring

## Dependencies
- Flask 2.3.3: Web framework
- scikit-learn 1.3.0: Machine learning
- pandas 2.0.3: Data manipulation
- numpy 1.24.3: Numerical computations
- joblib 1.2.0: Model persistence

## Development Setup
1. Python virtual environment
2. Dependencies installation
3. Database initialization
4. Model training and saving
5. Application testing

## Deployment Considerations
1. **Server Requirements**:
   - Python 3.x
   - Sufficient memory for model loading
   - SQLite database storage

2. **Scaling**:
   - Database optimization for larger datasets
   - Caching implementation
   - Load balancing for multiple instances

## Monitoring and Maintenance
1. **Logging**:
   - Application logs
   - Error tracking
   - Prediction history

2. **Updates**:
   - Regular dependency updates
   - Model retraining schedule
   - Security patches 