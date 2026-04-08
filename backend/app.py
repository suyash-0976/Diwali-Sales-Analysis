from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import LabelEncoder
from flask.json.provider import DefaultJSONProvider

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Diwali Sales Data.csv')

print("Models directory:", MODELS_DIR)
print("Models exist:", os.path.exists(MODELS_DIR))
print("Data path:", DATA_PATH)

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Import EDA functions
from analysis.eda import (
    get_summary_stats,
    get_gender_analysis,
    get_age_analysis,
    get_state_analysis,
    get_marital_analysis,
    get_occupation_analysis,
    get_category_analysis
)

app = Flask(__name__)
app.json_provider_class = CustomJSONProvider
app.json = CustomJSONProvider(app)
CORS(app)

# Load CSV data at startup
try:
    df = pd.read_csv(DATA_PATH, encoding='unicode_escape')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = None

def encode_prediction_features(gender, age, marital_status, orders, state, zone, occupation, product_category):
    """Encode features for prediction using same logic as train_models.py"""
    
    # Encode Gender: F=1, M=0
    gender_encoded = 1 if gender == 'F' else 0
    
    # Encode Age Group as ordinal (determine age group from age)
    if age <= 17:
        age_group_encoded = 0
    elif age <= 25:
        age_group_encoded = 1
    elif age <= 35:
        age_group_encoded = 2
    elif age <= 45:
        age_group_encoded = 3
    elif age <= 50:
        age_group_encoded = 4
    elif age <= 55:
        age_group_encoded = 5
    else:
        age_group_encoded = 6
    
    # Load saved label encoders
    try:
        state_encoder = joblib.load(os.path.join(MODELS_DIR, 'state_encoder.pkl'))
        zone_encoder = joblib.load(os.path.join(MODELS_DIR, 'zone_encoder.pkl'))
        occupation_encoder = joblib.load(os.path.join(MODELS_DIR, 'occupation_encoder.pkl'))
        category_encoder = joblib.load(os.path.join(MODELS_DIR, 'product_category_encoder.pkl'))
        
        # Handle unseen labels
        try:
            state_encoded = state_encoder.transform([state])[0]
        except ValueError:
            state_encoded = 0  # Default to first class if unseen
            
        try:
            zone_encoded = zone_encoder.transform([zone])[0]
        except ValueError:
            zone_encoded = 0  # Default to first class if unseen
            
        try:
            occupation_encoded = occupation_encoder.transform([occupation])[0]
        except ValueError:
            occupation_encoded = 0  # Default to first class if unseen
            
        try:
            category_encoded = category_encoder.transform([product_category])[0]
        except ValueError:
            category_encoded = 0  # Default to first class if unseen
            
    except Exception as e:
        print(f"Error loading encoders: {e}")
        # Fallback to simple encoding
        state_encoded = 0
        zone_encoded = 0
        occupation_encoded = 0
        category_encoded = 0
    
    # Create feature array in the same order as training
    features = np.array([[
        gender_encoded,
        age_group_encoded,
        marital_status,
        orders,
        state_encoded,
        zone_encoded,
        occupation_encoded,
        category_encoded
    ]])
    
    return features

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary statistics"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        result = get_summary_stats(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/eda/gender', methods=['GET'])
def get_gender():
    """Get gender analysis"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        result = get_gender_analysis(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/eda/age', methods=['GET'])
def get_age():
    """Get age group analysis"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        result = get_age_analysis(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/eda/state', methods=['GET'])
def get_state():
    """Get state analysis"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        result = get_state_analysis(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/eda/marital', methods=['GET'])
def get_marital():
    """Get marital status analysis"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        result = get_marital_analysis(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/eda/occupation', methods=['GET'])
def get_occupation():
    """Get occupation analysis"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        result = get_occupation_analysis(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/eda/category', methods=['GET'])
def get_category():
    """Get product category analysis"""
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        result = get_category_analysis(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/results', methods=['GET'])
def get_ml_results():
    """Get ML model results"""
    try:
        results_path = os.path.join(MODELS_DIR, 'model_results.json')
        with open(results_path, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    except FileNotFoundError:
        return jsonify({"error": "Run train_models.py first"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/predict', methods=['GET'])
def predict():
    try:
        print("Received params:", request.args)
        
        import os, joblib, numpy as np
        MODELS_DIR = os.path.join(os.path.dirname(
                     os.path.abspath(__file__)), 'models', 'saved')
        
        # Get parameters
        gender = request.args.get('gender', 'F')
        age = int(request.args.get('age', 30))
        marital_status = int(request.args.get('marital_status', 0))
        orders = int(request.args.get('orders', 2))
        state = request.args.get('state', 'Uttar Pradesh')
        zone = request.args.get('zone', 'Central')
        occupation = request.args.get('occupation', 'IT Sector')
        product_category = request.args.get('product_category', 'Food')
        
        # Encode gender
        gender_encoded = 1 if gender == 'F' else 0
        
        # Encode Age Group from age number
        if age <= 17:
            age_group_encoded = 0
        elif age <= 25:
            age_group_encoded = 1
        elif age <= 35:
            age_group_encoded = 2
        elif age <= 45:
            age_group_encoded = 3
        elif age <= 50:
            age_group_encoded = 4
        elif age <= 55:
            age_group_encoded = 5
        else:
            age_group_encoded = 6
        
        # Load label encoders
        state_enc = joblib.load(os.path.join(MODELS_DIR, 'state_encoder.pkl'))
        zone_enc = joblib.load(os.path.join(MODELS_DIR, 'zone_encoder.pkl'))
        occ_enc = joblib.load(os.path.join(MODELS_DIR, 'occupation_encoder.pkl'))
        cat_enc = joblib.load(os.path.join(MODELS_DIR, 'product_category_encoder.pkl'))
        
        # Transform with encoders safely
        try:
            state_encoded = int(state_enc.transform([state])[0])
        except:
            state_encoded = 0
        try:
            zone_encoded = int(zone_enc.transform([zone])[0])
        except:
            zone_encoded = 0
        try:
            occ_encoded = int(occ_enc.transform([occupation])[0])
        except:
            occ_encoded = 0
        try:
            cat_encoded = int(cat_enc.transform([product_category])[0])
        except:
            cat_encoded = 0
        
        # Build feature array - same order as training
        features = np.array([[gender_encoded, age_group_encoded, 
                               marital_status, orders,
                               state_encoded, zone_encoded, 
                               occ_encoded, cat_encoded]])
        
        print("Feature array:", features)
        
        # Load models and predict
        models = {
            'Linear Regression': 'linear_regression_model.pkl',
            'Lasso': 'lasso_model.pkl',
            'Ridge': 'ridge_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl'
        }
        
        results = {}
        for model_name, filename in models.items():
            model = joblib.load(os.path.join(MODELS_DIR, filename))
            pred = float(model.predict(features)[0])
            results[model_name] = round(pred, 2)
        
        print("Predictions:", results)
        return jsonify(results)
        
    except Exception as e:
        import traceback
        print("PREDICTION ERROR:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)