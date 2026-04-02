import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load the data
df = pd.read_csv('ice-cream.csv')

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())

# Prepare data for modeling
# Encode categorical variables
le_day = LabelEncoder()
le_month = LabelEncoder()

df['DayOfWeek_encoded'] = le_day.fit_transform(df['DayOfWeek'])
df['Month_encoded'] = le_month.fit_transform(df['Month'])

# Select features for the model
X = df[['Temperature', 'Rainfall', 'DayOfWeek_encoded', 'Month_encoded']]
y = df['IceCreamsSold']

# Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Print model performance
train_score = model.score(X, y)
print(f"\n\nModel R-squared Score: {train_score:.4f}")
print(f"Model Coefficients:")
print(f"  Temperature: {model.coef_[0]:.4f}")
print(f"  Rainfall: {model.coef_[1]:.4f}")
print(f"  DayOfWeek: {model.coef_[2]:.4f}")
print(f"  Month: {model.coef_[3]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Save the model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('le_day.pkl', 'wb') as f:
    pickle.dump(le_day, f)

with open('le_month.pkl', 'wb') as f:
    pickle.dump(le_month, f)

print("\n✓ Model trained and saved successfully!")
print("Files created: model.pkl, le_day.pkl, le_month.pkl")
