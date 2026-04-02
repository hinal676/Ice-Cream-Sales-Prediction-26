# 🍦 Ice Cream Sales Predictor

A machine learning web application that predicts ice cream sales based on weather conditions and day information using **Linear Regression** and **Flask**.

## 📋 Project Overview

This project uses a Linear Regression model to predict daily ice cream sales based on:

- **Temperature**: Current temperature in Fahrenheit
- **Rainfall**: Amount of rainfall in inches
- **Day of Week**: The day of the week
- **Month**: The month of the year

## 🎯 Features

✨ **Machine Learning Model**: Linear Regression trained on historical ice cream sales data
🌐 **Web Interface**: Beautiful, responsive Flask web application
📊 **Data Visualization**: Statistical overview and historical top performers
🎨 **Aesthetic Design**: Modern UI with gradient backgrounds and smooth animations
📱 **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices
⚡ **Real-time Predictions**: Get instant sales predictions based on input parameters

## 📁 Project Structure

```
ice-cream-sales-predictor/
├── ice-cream.csv              # Dataset with historical ice cream sales
├── train_model.py             # Script to train the ML model
├── app.py                     # Flask application
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html            # Frontend HTML template
├── static/
│   └── style.css             # CSS styling
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Navigate to the project directory:**

   ```bash
   cd ice-cream-sales-predictor
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the machine learning model:**

   ```bash
   python train_model.py
   ```

   This will:
   - Load the ice-cream.csv dataset
   - Train a Linear Regression model
   - Display model performance metrics
   - Save the model and label encoders for Flask to use

4. **Run the Flask web application:**

   ```bash
   python app.py
   ```

5. **Open your browser:**
   - Navigate to `http://localhost:5000`
   - The web application will open automatically

## 📊 How It Works

### Training Phase

The `train_model.py` script:

1. Loads data from `ice-cream.csv`
2. Encodes categorical variables (Day of Week, Month)
3. Trains a Linear Regression model on features: Temperature, Rainfall, DayOfWeek, Month
4. Saves the model and encoders as pickle files

### Prediction Phase

When you make a prediction through the web interface:

1. User enters Temperature, Rainfall, Day of Week, and Month
2. Flask receives the input via POST request
3. The trained model predicts ice cream sales
4. Results are displayed with all input parameters

### API Endpoints

- **GET `/`** - Main page with prediction interface
- **POST `/predict`** - Submit prediction request
- **GET `/api/statistics`** - Get dataset statistics
- **GET `/api/days`** - Get available days
- **GET `/api/months`** - Get available months
- **GET `/api/historical`** - Get top 10 sales days

## 🎨 UI Features

- **Prediction Form**: Easy-to-use input fields with validation
- **Real-time Results**: Instant prediction display with input summary
- **Dataset Statistics**: 9 different statistical metrics
- **Historical Data**: Shows top 10 sales days with details
- **Responsive Design**: Optimized for all screen sizes
- **Smooth Animations**: Professional transitions and effects

## 🧠 Machine Learning Model Details

### Algorithm: Linear Regression

- **Type**: Supervised learning
- **Use Case**: Regression (predicting continuous values)
- **Model Complexity**: Simple and interpretable

### Features

1. Temperature (°F)
2. Rainfall (inches)
3. Day of Week (encoded)
4. Month (encoded)

### Model Performance

Run `python train_model.py` to see the R² score and coefficients for each feature.

## 📝 Data Format

The `ice-cream.csv` should contain:

```
Date,DayOfWeek,Month,Temperature,Rainfall,IceCreamsSold
01-04-2025,Tuesday,April,59.4,0.74,61
...
```

## 🔧 Configuration

### Port

To change the port Flask runs on, edit `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change 5000 to your desired port
```

### Debug Mode

For development, debug mode is enabled. For production:

```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## 🐛 Troubleshooting

**Issue**: Model files not found

- **Solution**: Make sure you've run `python train_model.py` first

**Issue**: Port already in use

- **Solution**: Change the port in `app.py` or kill the process using that port

**Issue**: Dependencies not installed

- **Solution**: Run `pip install -r requirements.txt` again

## 🎓 Learning Resources

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Data Analysis](https://pandas.pydata.org/)

## 📈 Extending the Project

You can enhance this project by:

- Adding more features (humidity, wind speed, etc.)
- Implementing different algorithms (polynomial regression, decision trees)
- Adding model evaluation metrics visualization
- Implementing data preprocessing pipeline
- Adding user authentication
- Creating prediction history storage
- Building API rate limiting

## 📄 License

This project is open-source and available for educational purposes.

## 👨‍💻 Author

Created as a machine learning and web development project.

---

**Happy Predicting! 🍦**
