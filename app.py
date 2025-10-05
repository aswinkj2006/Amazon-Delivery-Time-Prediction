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

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    if not isinstance(model, (RandomForestRegressor, XGBRegressor)):
        return None
    
    # Get feature importance
    if isinstance(model, RandomForestRegressor):
        importance = model.feature_importances_
    else:  # XGBoost
        importance = model.feature_importances_
    
    # Create feature importance dataframe
    features = ['Agent Age', 'Agent Rating', 'Distance (km)', 'Pickup Delay (min)', 'Order Hour']
    imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
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
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def main():
    st.title('🚚 Amazon Delivery Time Predictor')
    
    # Load model and sample data
    model = load_model()
    sample_data = load_sample_data()
    
    if model is None:
        st.error('⚠️ No trained model found. Please run train.py first.')
        return

    # Sidebar - Input Parameters
    st.sidebar.header('📊 Order Details')
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.number_input('Agent Age', value=30, min_value=18, max_value=70)
        rating = st.slider('Agent Rating', 0.0, 5.0, 4.5, 0.1)
        distance = st.number_input('Distance (km)', value=5.0, min_value=0.0, step=0.1)
    with col2:
        pickup_delay = st.number_input('Pickup Delay (min)', value=10.0, min_value=0.0)
        hour = st.slider('Order Hour', 0, 23, 12)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    # Prediction section
    with col2:
        st.markdown("### 🎯 Prediction")
        if st.button('Predict Delivery Time', type='primary'):
            X = pd.DataFrame([{
                'Agent_Age': age,
                'Agent_Rating': rating,
                'distance_km': distance,
                'pickup_delay_min': pickup_delay,
                'order_hour': hour
            }])
            pred = model.predict(X)[0]
            
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin:0">Estimated Delivery Time</h3>
                <h2 style="color:#FF9900;margin:0">{:.1f} minutes</h2>
            </div>
            """.format(pred), unsafe_allow_html=True)

            # Add context
            if sample_data is not None:
                avg_time = sample_data['Delivery_Time'].mean()
                if pred < avg_time:
                    st.success(f'⚡ Faster than average by {avg_time - pred:.1f} minutes!')
                else:
                    st.warning(f'⏳ Longer than average by {pred - avg_time:.1f} minutes')

    # Analytics Dashboard
    with col1:
        st.markdown("### 📈 Analytics Dashboard")
        
        # Feature importance plot
        imp_fig = plot_feature_importance(model)
        if imp_fig:
            st.plotly_chart(imp_fig, use_container_width=True)
        
        st.info('💡 The model uses machine learning to predict delivery times based on historical data and current conditions.')


if __name__ == '__main__':
    main()
