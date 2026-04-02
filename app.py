from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('le_day.pkl', 'rb') as f:
    le_day = pickle.load(f)

with open('le_month.pkl', 'rb') as f:
    le_month = pickle.load(f)

# Load the original dataset for statistics
df = pd.read_csv('ice-cream.csv')


@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on user input"""
    try:
        # Get input data from the form
        temperature = float(request.form.get('temperature'))
        rainfall = float(request.form.get('rainfall'))
        day_of_week = request.form.get('day_of_week')
        month = request.form.get('month')

        # Validate inputs
        if temperature < -50 or temperature > 150:
            return jsonify({'error': 'Temperature must be between -50 and 150'}), 400

        if rainfall < 0 or rainfall > 10:
            return jsonify({'error': 'Rainfall must be between 0 and 10'}), 400

        # Encode categorical variables
        day_encoded = le_day.transform([day_of_week])[0]
        month_encoded = le_month.transform([month])[0]

        # Create feature array
        features = np.array(
            [[temperature, rainfall, day_encoded, month_encoded]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Ensure prediction is non-negative
        prediction = max(0, prediction)

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'temperature': temperature,
            'rainfall': rainfall,
            'day_of_week': day_of_week,
            'month': month
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/api/statistics')
def get_statistics():
    """Return dataset statistics and model performance"""
    from sklearn.preprocessing import LabelEncoder

    # Calculate model performance metrics
    le_day_temp = LabelEncoder()
    le_month_temp = LabelEncoder()

    df_temp = df.copy()
    df_temp['DayOfWeek_encoded'] = le_day_temp.fit_transform(
        df_temp['DayOfWeek'])
    df_temp['Month_encoded'] = le_month_temp.fit_transform(df_temp['Month'])

    X = df_temp[['Temperature', 'Rainfall',
                 'DayOfWeek_encoded', 'Month_encoded']]
    y = df_temp['IceCreamsSold']

    r2_score = model.score(X, y)

    stats = {
        'avg_temperature': round(df['Temperature'].mean(), 2),
        'max_temperature': round(df['Temperature'].max(), 2),
        'min_temperature': round(df['Temperature'].min(), 2),
        'avg_rainfall': round(df['Rainfall'].mean(), 2),
        'max_rainfall': round(df['Rainfall'].max(), 2),
        'avg_sales': round(df['IceCreamsSold'].mean(), 2),
        'max_sales': int(df['IceCreamsSold'].max()),
        'min_sales': int(df['IceCreamsSold'].min()),
        'model_r2': round(r2_score, 4),
        'model_intercept': round(model.intercept_, 4),
        'model_coef_temp': round(model.coef_[0], 4),
        'model_coef_rain': round(model.coef_[1], 4)
    }
    return jsonify(stats)


@app.route('/api/days')
def get_days():
    """Return available days of week"""
    days = le_day.classes_.tolist()
    return jsonify({'days': days})


@app.route('/api/months')
def get_months():
    """Return available months"""
    months = le_month.classes_.tolist()
    return jsonify({'months': months})


@app.route('/api/historical')
def get_historical():
    """Return historical data for visualization"""
    historical = df.nlargest(10, 'IceCreamsSold')[
        ['Temperature', 'Rainfall', 'IceCreamsSold', 'DayOfWeek']].to_dict('records')
    return jsonify({'historical': historical})


if __name__ == '__main__':
    print("🍦 Ice Cream Sales Predictor is running!")
    print("Visit: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
