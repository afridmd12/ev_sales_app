from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

app = Flask(__name__)

DATA_PATH = 'data/EV_Dataset.csv'
MODEL_PATH = 'model/model.pkl'

# ===== TRAIN MODEL IF NOT EXISTS =====
def train_and_save_model():
    df = pd.read_csv(DATA_PATH)

    # Clean and preprocess
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(['Month_Name'], axis=1, inplace=True)

    categorical = ['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']
    for col in categorical:
        df[col] = df[col].astype(str).str.strip()
    df = pd.get_dummies(df, columns=categorical, drop_first=False)

    X = df.drop(['EV_Sales_Quantity', 'Date'], axis=1)
    y = df['EV_Sales_Quantity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs('model', exist_ok=True)
    joblib.dump((model, X.columns.tolist()), MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")

# Train model if not exists
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# Load model
model, model_columns = joblib.load(MODEL_PATH)

# ===== ROUTES =====
@app.route('/')
def home():
    df_raw = pd.read_csv(DATA_PATH)

    # Strip spaces to fix dropdown issues
    for col in ['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']:
        df_raw[col] = df_raw[col].astype(str).str.strip()

    states = sorted(df_raw['State'].unique())
    vehicle_classes = sorted(df_raw['Vehicle_Class'].unique())
    vehicle_categories = sorted(df_raw['Vehicle_Category'].unique())
    vehicle_types = sorted(df_raw['Vehicle_Type'].unique())

    return render_template(
        'index.html',
        states=states,
        vehicle_classes=vehicle_classes,
        vehicle_categories=vehicle_categories,
        vehicle_types=vehicle_types
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Parse date
    selected_date = request.form['date']
    date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    # Get other inputs
    state = request.form['state'].strip()
    vehicle_class = request.form['vehicle_class'].strip()
    vehicle_category = request.form['vehicle_category'].strip()
    vehicle_type = request.form['vehicle_type'].strip()

    # ===== LOOKUP FIRST =====
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Year'].astype(int)

    for col in ['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']:
        df[col] = df[col].astype(str).str.strip()

    match = df[
        (df['Year'] == year) &
        (df['Date'].dt.month == month) &
        (df['Date'].dt.day == day) &
        (df['State'] == state) &
        (df['Vehicle_Class'] == vehicle_class) &
        (df['Vehicle_Category'] == vehicle_category) &
        (df['Vehicle_Type'] == vehicle_type)
    ]

    if not match.empty:
        prediction = int(match['EV_Sales_Quantity'].iloc[0])
        return render_template('result.html', prediction=prediction)

    # ===== FALLBACK TO ML PREDICTION =====
    input_data = pd.DataFrame([[0] * len(model_columns)], columns=model_columns)
    input_data['Year'] = year
    input_data['Month'] = month
    input_data['Day'] = day

    if f"State_{state}" in input_data.columns:
        input_data[f"State_{state}"] = 1
    if f"Vehicle_Class_{vehicle_class}" in input_data.columns:
        input_data[f"Vehicle_Class_{vehicle_class}"] = 1
    if f"Vehicle_Category_{vehicle_category}" in input_data.columns:
        input_data[f"Vehicle_Category_{vehicle_category}"] = 1
    if f"Vehicle_Type_{vehicle_type}" in input_data.columns:
        input_data[f"Vehicle_Type_{vehicle_type}"] = 1

    prediction = model.predict(input_data)[0]
    return render_template('result.html', prediction=int(prediction))

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 2004))
    app.run(host='0.0.0.0', port=port, debug=False)