# 🪔 Diwali Sales Dashboard

A comprehensive sales intelligence platform powered by Machine Learning and Data Analysis for Diwali sales data.

## 🚀 Features

- **📊 Exploratory Data Analysis**: Deep insights into customer demographics, geographic patterns, and product preferences
- **🤖 Machine Learning Predictions**: 4 ML models (Linear, Lasso, Ridge, Decision Tree) for purchase amount prediction
- **📈 Interactive Visualizations**: 6 different Chart.js visualizations with dark Diwali theme
- **🎨 Beautiful UI**: Modern dark theme with festive Diwali colors and smooth animations

## 🛠️ Tech Stack

**Backend:**
- Python 3.x
- Flask (REST API)
- Pandas, NumPy (Data Processing)
- Scikit-learn (Machine Learning)
- Joblib (Model Persistence)

**Frontend:**
- HTML5, CSS3, JavaScript
- Tailwind CSS (Styling)
- Chart.js (Data Visualization)
- Inter Font (Typography)

## 📋 Setup Instructions

### Step 1: Navigate to Backend
```bash
cd backend
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train ML Models
```bash
python models/train_models.py
```
This will:
- Load and clean the Diwali Sales dataset
- Train 4 ML models for purchase amount prediction
- Save models and performance metrics to `models/saved/`

### Step 4: Start Flask Server
```bash
python app.py
```
The API server will start on `http://localhost:5000`

### Step 5: Open Frontend
Open `frontend/index.html` in your web browser to access the dashboard.

## 📁 Project Structure

```
diwali-sales-dashboard/
├── backend/
│   ├── app.py                 # Flask REST API server
│   ├── requirements.txt        # Python dependencies
│   ├── data/
│   │   └── Diwali Sales Data.csv  # Dataset (place here)
│   ├── models/
│   │   ├── train_models.py    # ML model training script
│   │   └── saved/            # Trained models & results
│   └── analysis/
│       └── eda.py            # EDA functions
├── frontend/
│   ├── index.html            # Main dashboard
│   ├── style.css             # Custom styles
│   ├── script.js             # Shared JavaScript
│   └── pages/
│       ├── ml_predictions.html    # ML predictions page
│       ├── eda_analysis.html      # EDA insights page
│       └── visualizations.html    # Charts page
└── README.md
```

## 🔧 API Endpoints

### Summary & EDA
- `GET /api/summary` - Dashboard summary statistics
- `GET /api/eda/gender` - Gender-wise analysis
- `GET /api/eda/age` - Age group analysis
- `GET /api/eda/state` - State-wise performance
- `GET /api/eda/marital` - Marital status analysis
- `GET /api/eda/occupation` - Occupation analysis
- `GET /api/eda/category` - Product category analysis

### Machine Learning
- `GET /api/ml/results` - Model performance metrics
- `GET /api/ml/predict` - Live prediction with query parameters

## 🎯 Key Insights

The dashboard reveals that:
- **Married women aged 26-35** from **Uttar Pradesh, Maharashtra & Karnataka** working in **IT, Healthcare and Aviation** are most likely to purchase **Food, Clothing & Electronics** during Diwali.

## 🎨 Design Features

- **Dark Diwali Theme**: Festive color palette with #0f0f1a, #1a1a2e, #e94560, #f5a623
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Smooth Animations**: Counter animations, hover effects, and loading spinners
- **Interactive Charts**: 6 different Chart.js visualizations with tooltips
- **Modern UI**: Card-based layout with backdrop blur and gradient effects

## 📊 ML Models

The system trains and evaluates 4 regression models:
1. **Linear Regression** - Baseline model
2. **Lasso Regression** - L1 regularization
3. **Ridge Regression** - L2 regularization  
4. **Decision Tree** - Non-linear model with max_depth=5

Each model is evaluated using R², MAE, and RMSE metrics.

## 🌟 Usage Tips

1. **Ensure the dataset** `Diwali Sales Data.csv` is placed in `backend/data/`
2. **Train models first** before accessing ML features
3. **Use the prediction form** to estimate purchase amounts for different customer profiles
4. **Explore visualizations** to understand patterns and trends
5. **Check EDA Analysis** for detailed demographic insights

## 🎄 Happy Diwali! 🎆

Built with ❤️ for the festive season! 🪔✨
