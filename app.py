"""Streamlit app to predict delivery time using the trained model
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Page config
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling with dark theme
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
        background-color: #0E1117;
    }
    .stPlotlyChart {
        background-color: #1E2126;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #1E2126;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin: 1rem 0;
        border: 1px solid #2E3137;
    }
    .prediction-box {
        background-color: #1E2126;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-top: 1rem;
        border: 1px solid #2E3137;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #1E2126;
    }
    h1 {
        color: #FFFFFF;
        padding-bottom: 2rem;
    }
    h3 {
        color: #FFFFFF;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    p = Path(__file__).parent / 'models' / 'best_model.pkl'
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_data
def load_sample_data():
    p = Path(__file__).parent / 'data' / 'processed' / 'amazon_delivery_processed.csv'
    if not p.exists():
        return None
    return pd.read_csv(p)


def plot_feature_importance(model):
    """Plot feature importance for the model"""
    # Define default feature names used in training
    default_features = [
        'Agent_Age', 'Agent_Rating', 'Distance', 'Waiting_Time',
        'Order_Hour', 'Pickup_Hour', 'Weather_Encoded', 'Traffic_Encoded',
        'Vehicle_Encoded', 'Area_Encoded', 'Category_Encoded'
    ]

    if not isinstance(model, (RandomForestRegressor, XGBRegressor)):
        return None
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Get feature names with fallback to default
    try:
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
        elif hasattr(model, 'feature_names'):
            features = model.feature_names
        else:
            features = default_features[:len(importance)]
    except Exception:
        features = default_features[:len(importance)]
    
    # Ensure features and importance have same length
    if len(features) != len(importance):
        st.warning("Feature length mismatch. Using default feature names.")
        features = [f"Feature_{i}" for i in range(len(importance))]
    
    # Create feature importance dataframe
    imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=imp_df['Importance'],
        y=imp_df['Feature'],
        orientation='h'
    ))
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        showlegend=False,
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def main():
    # Page title with icon
    st.title('🚚 Amazon Delivery Time Predictor')
    
    # Load model and sample data
    model = load_model()
    sample_data = load_sample_data()
    
    if model is None:
        st.error('⚠️ No trained model found. Please train the model first using:')
        st.code('python train.py')
        return

    # Sidebar with organized sections
    st.sidebar.markdown("""
    <div style='background-color: #1E2126; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h3 style='color: #FFFFFF; margin: 0;'>📊 Order Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent Information Section
    st.sidebar.markdown("<h4 style='color: #FFFFFF;'>👤 Agent Information</h4>", unsafe_allow_html=True)
    age = st.sidebar.number_input('Agent Age', value=30, min_value=18, max_value=70)
    rating = st.sidebar.slider('Agent Rating', 0.0, 5.0, 4.5, 0.1)
    
    # Delivery Information Section
    st.sidebar.markdown("<h4 style='color: #FFFFFF;'>� Delivery Information</h4>", unsafe_allow_html=True)
    distance = st.sidebar.number_input('Distance (km)', value=5.0, min_value=0.0, step=0.1)
    vehicle = st.sidebar.selectbox('Vehicle Type', ['Motorcycle', 'Scooter', 'Van'], index=0)
    area = st.sidebar.selectbox('Area Type', ['Urban', 'Metropolitan'], index=0)
    
    # Time Information Section
    st.sidebar.markdown("<h4 style='color: #FFFFFF;'>⏰ Time Information</h4>", unsafe_allow_html=True)
    order_hour = st.sidebar.slider('Order Hour', 0, 23, 12)
    pickup_hour = st.sidebar.slider('Pickup Hour', 0, 23, 13)
    
    # Conditions Section
    st.sidebar.markdown("<h4 style='color: #FFFFFF;'>🌤️ Conditions</h4>", unsafe_allow_html=True)
    weather = st.sidebar.selectbox('Weather', ['Sunny', 'Cloudy', 'Rainy', 'Stormy'], index=0)
    traffic = st.sidebar.selectbox('Traffic', ['Low', 'Medium', 'High', 'Jam'], index=1)
    category = st.sidebar.selectbox('Product Category', 
                                  ['Electronics', 'Clothing', 'Grocery', 'Cosmetics', 'Books'], 
                                  index=0)

    # Main content area with two columns
    col1, col2 = st.columns([2, 1])
    
    # Feature Importance Plot
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📈 Feature Importance</h3>
        </div>
        """, unsafe_allow_html=True)
        imp_fig = plot_feature_importance(model)
        if imp_fig:
            st.plotly_chart(imp_fig, use_container_width=True)
    
    # Prediction Section
    with col2:
        st.markdown("""
        <div class='prediction-box'>
            <h3>🎯 Delivery Time Prediction</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button('Calculate Delivery Time', type='primary'):
            # Create input features
            X = pd.DataFrame([{
                'Agent_Age': age,
                'Agent_Rating': rating,
                'Distance': distance,
                'Waiting_Time': (pickup_hour - order_hour) * 60,
                'Order_Hour': order_hour,
                'Pickup_Hour': pickup_hour,
                'Weather_Encoded': ['Sunny', 'Cloudy', 'Rainy', 'Stormy'].index(weather),
                'Traffic_Encoded': ['Low', 'Medium', 'High', 'Jam'].index(traffic),
                'Vehicle_Encoded': ['Motorcycle', 'Scooter', 'Van'].index(vehicle),
                'Area_Encoded': ['Urban', 'Metropolitan'].index(area),
                'Category_Encoded': ['Electronics', 'Clothing', 'Grocery', 'Cosmetics', 'Books'].index(category)
            }])
            
            # Make prediction
            pred = model.predict(X)[0]
            
            # Display prediction with styling
            st.markdown(f"""
            <div style='background-color: #1E2126; padding: 2rem; border-radius: 0.5rem; text-align: center;'>
                <h2 style='color: #FF4B4B; margin-bottom: 1rem;'>Estimated Delivery Time</h2>
                <h1 style='color: #FFFFFF; font-size: 3rem;'>{pred:.1f} minutes</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Display additional metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Distance", f"{distance:.1f} km")
            with col2:
                st.metric("Wait Time", f"{(pickup_hour - order_hour) * 60} min")

    # Information box at the bottom
    st.markdown("""
    <div style='background-color: #1E2126; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;'>
        <p style='color: #FFFFFF; margin: 0;'>
            💡 <strong>How it works:</strong> This model uses machine learning to predict delivery times based on historical data and current conditions.
            It considers factors like weather, traffic, vehicle type, and agent performance to provide accurate estimates.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
