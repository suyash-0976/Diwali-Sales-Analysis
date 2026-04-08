import pandas as pd

def clean_dataframe(df):
    """Clean the dataframe by dropping unnecessary columns and handling null values."""
    df = df.drop(columns=['Status', 'unnamed1'], errors='ignore')
    df = df.dropna(subset=['Amount'])
    df['Amount'] = df['Amount'].astype(int)
    return df

def get_gender_analysis(df):
    """Analyze gender distribution and spending."""
    df = clean_dataframe(df)
    
    gender_counts = df['Gender'].value_counts().to_dict()
    gender_amounts = df.groupby('Gender')['Amount'].sum().to_dict()
    
    # Convert numpy types to native Python types
    gender_counts = {k: int(v) for k, v in gender_counts.items()}
    gender_amounts = {k: int(v) for k, v in gender_amounts.items()}
    
    return {
        "counts": gender_counts,
        "amounts": gender_amounts
    }

def get_age_analysis(df):
    """Analyze age group distribution and spending."""
    df = clean_dataframe(df)
    
    age_analysis = df.groupby('Age Group').agg({
        'Age Group': 'count',
        'Amount': 'sum'
    }).rename(columns={'Age Group': 'counts'}).reset_index()
    
    age_analysis = age_analysis.sort_values('Amount', ascending=False)
    
    return {
        "age_groups": age_analysis['Age Group'].tolist(),
        "counts": [int(x) for x in age_analysis['counts'].tolist()],
        "amounts": [int(x) for x in age_analysis['Amount'].tolist()]
    }

def get_state_analysis(df):
    """Analyze state-wise orders and amounts (top 10 by amount)."""
    df = clean_dataframe(df)
    
    state_analysis = df.groupby('State').agg({
        'Orders': 'sum',
        'Amount': 'sum'
    }).reset_index()
    
    state_analysis = state_analysis.sort_values('Amount', ascending=False).head(10)
    
    return {
        "states": state_analysis['State'].tolist(),
        "orders": [int(x) for x in state_analysis['Orders'].tolist()],
        "amounts": [int(x) for x in state_analysis['Amount'].tolist()]
    }

def get_marital_analysis(df):
    """Analyze spending by marital status and gender."""
    df = clean_dataframe(df)
    
    marital_analysis = df.groupby(['Marital_Status', 'Gender'])['Amount'].sum().unstack(fill_value=0)
    
    return {
        "status": [int(x) for x in marital_analysis.index.tolist()],
        "male_amounts": [int(x) for x in marital_analysis.get('M', [0, 0]).tolist()],
        "female_amounts": [int(x) for x in marital_analysis.get('F', [0, 0]).tolist()]
    }

def get_occupation_analysis(df):
    """Analyze occupation-wise spending (top 10 by amount)."""
    df = clean_dataframe(df)
    
    occupation_analysis = df.groupby('Occupation')['Amount'].sum().reset_index()
    occupation_analysis = occupation_analysis.sort_values('Amount', ascending=False).head(10)
    
    return {
        "occupations": occupation_analysis['Occupation'].tolist(),
        "amounts": [int(x) for x in occupation_analysis['Amount'].tolist()]
    }

def get_category_analysis(df):
    """Analyze product category performance (top 10 by amount)."""
    df = clean_dataframe(df)
    
    category_analysis = df.groupby('Product_Category').agg({
        'Product_Category': 'count',
        'Amount': 'sum'
    }).rename(columns={'Product_Category': 'counts'}).reset_index()
    
    category_analysis = category_analysis.sort_values('Amount', ascending=False).head(10)
    
    return {
        "categories": category_analysis['Product_Category'].tolist(),
        "counts": [int(x) for x in category_analysis['counts'].tolist()],
        "amounts": [int(x) for x in category_analysis['Amount'].tolist()]
    }

def get_summary_stats(df):
    """Generate summary statistics for the dataset."""
    df = clean_dataframe(df)
    
    # Get top performers
    state_analysis = df.groupby('State')['Amount'].sum().sort_values(ascending=False)
    category_analysis = df.groupby('Product_Category')['Amount'].sum().sort_values(ascending=False)
    occupation_analysis = df.groupby('Occupation')['Amount'].sum().sort_values(ascending=False)
    age_analysis = df.groupby('Age Group')['Amount'].sum().sort_values(ascending=False)
    
    return {
        "total_records": int(len(df)),
        "total_revenue": int(df['Amount'].sum()),
        "total_orders": int(df['Orders'].sum()),
        "unique_customers": int(df['User_ID'].nunique()),
        "top_state": str(state_analysis.index[0]) if len(state_analysis) > 0 else "",
        "top_category": str(category_analysis.index[0]) if len(category_analysis) > 0 else "",
        "top_occupation": str(occupation_analysis.index[0]) if len(occupation_analysis) > 0 else "",
        "top_age_group": str(age_analysis.index[0]) if len(age_analysis) > 0 else ""
    }
