import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
import os

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved')
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), 'data', 'Diwali Sales Data.csv')

print("Saving models to:", MODELS_DIR)

def load_and_clean_data():
    """Load and clean the Diwali Sales dataset."""
    df = pd.read_csv(DATA_PATH, encoding='unicode_escape')
    
    # Drop unnecessary columns
    df = df.drop(columns=['Status', 'unnamed1'], errors='ignore')
    
    # Drop rows where Amount is null
    df = df.dropna(subset=['Amount'])
    
    # Convert Amount to int
    df['Amount'] = df['Amount'].astype(int)
    
    return df

def encode_features(df):
    """Perform feature engineering and encoding."""
    df_encoded = df.copy()
    
    # Encode Gender: F=1, M=0
    df_encoded['Gender'] = df_encoded['Gender'].map({'F': 1, 'M': 0})
    
    # Encode Age Group as ordinal
    age_mapping = {
        '0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, 
        '46-50': 4, '51-55': 5, '55+': 6
    }
    df_encoded['Age Group'] = df_encoded['Age Group'].map(age_mapping)
    
    # Label encode categorical columns
    label_encoders = {}
    categorical_columns = ['State', 'Zone', 'Occupation', 'Product_Category']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    return df_encoded, label_encoders

def train_models():
    """Train and evaluate ML models."""
    # Load and clean data
    df = load_and_clean_data()
    print(f"Dataset loaded and cleaned. Shape: {df.shape}")
    
    # Feature engineering
    df_encoded, label_encoders = encode_features(df)
    
    # Define features and target
    feature_columns = [
        'Gender', 'Age Group', 'Marital_Status', 'Orders',
        'State', 'Zone', 'Occupation', 'Product_Category'
    ]
    
    X = df_encoded[feature_columns]
    y = df_encoded['Amount']
    
    # Save feature names
    feature_names = feature_columns.copy()
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save label encoders for reuse in prediction
    joblib.dump(label_encoders['State'], os.path.join(MODELS_DIR, 'state_encoder.pkl'))
    joblib.dump(label_encoders['Zone'], os.path.join(MODELS_DIR, 'zone_encoder.pkl'))
    joblib.dump(label_encoders['Occupation'], os.path.join(MODELS_DIR, 'occupation_encoder.pkl'))
    joblib.dump(label_encoders['Product_Category'], os.path.join(MODELS_DIR, 'product_category_encoder.pkl'))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(alpha=1.0),
        'Ridge': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Store results
        results[name] = {
            "r2": float(test_r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "train_r2": float(train_r2)
        }
        
        # Save model
        model_filename = name.lower().replace(' ', '_') + '_model.pkl'
        joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
        
        print(f"{name} - Test R²: {test_r2:.4f}, Train R²: {train_r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Save results
    with open(os.path.join(MODELS_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Model Training Complete ===")
    print("\nFinal Results:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Test R²: {metrics['r2']:.4f}")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
    
    print(f"\nModels and results saved to {MODELS_DIR}")
    print(f"Feature names saved to {MODELS_DIR}")
    
    return results

if __name__ == "__main__":
    train_models()