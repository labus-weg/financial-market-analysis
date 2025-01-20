import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import MarketAnomalyDetector
from datetime import datetime, timedelta

# Page config remains same...
[Previous page config and CSS code remains the same]

def create_technical_indicators(df):
    """Calculate technical indicators for visualization"""
    df = df.copy()
    if 'Close' in df.columns:
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    return df

def plot_market_analysis(data, predictions, scores):
    """Create comprehensive market analysis plots"""
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_20'],
        name='20-day MA',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BB_upper'],
        name='Upper BB',
        line=dict(color='gray', width=1, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BB_lower'],
        name='Lower BB',
        line=dict(color='gray', width=1, dash='dot'),
        fill='tonexty'
    ))
    
    # Add anomaly points
    anomaly_points = data[predictions == 1]
    fig.add_trace(go.Scatter(
        x=anomaly_points.index,
        y=anomaly_points['Close'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title='Market Analysis with Anomalies',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        hovermode='x unified'
    )
    
    return fig

def generate_market_insights(data, predictions, scores):
    """Generate detailed market insights based on anomalies"""
    # Get recent data
    recent_data = data.iloc[-30:]  # Last 30 data points
    recent_anomalies = predictions[-30:]
    recent_scores = scores[-30:]
    
    # Calculate key metrics
    volatility = recent_data['Volatility'].mean()
    rsi = recent_data['RSI'].iloc[-1]
    price_trend = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0] * 100
    
    # Generate insights
    insights = {
        'market_status': 'High Risk' if predictions[-1] == 1 else 'Normal',
        'anomaly_score': scores[-1],
        'risk_level': 'High' if volatility > 0.02 else 'Moderate' if volatility > 0.01 else 'Low',
        'trend': 'Bullish' if price_trend > 0 else 'Bearish',
        'overbought': rsi > 70,
        'oversold': rsi < 30,
        'recommendations': []
    }
    
    # Generate recommendations based on conditions
    if predictions[-1] == 1:  # Anomaly detected
        if insights['trend'] == 'Bearish':
            insights['recommendations'].extend([
                "⚠️ Consider reducing position sizes",
                "🛡️ Review and tighten stop-loss levels",
                "💰 Increase cash holdings to 40-50%",
                "📊 Monitor volatility closely"
            ])
        else:
            insights['recommendations'].extend([
                "⚠️ Exercise caution despite upward trend",
                "🎯 Consider taking partial profits",
                "⚖️ Reduce leverage if using any",
                "📈 Watch for potential trend reversal"
            ])
    else:  # Normal conditions
        if insights['trend'] == 'Bullish':
            insights['recommendations'].extend([
                "✅ Maintain current positions",
                "📈 Look for potential entry points",
                "🎯 Consider scaling into new positions",
                "📊 Regular portfolio rebalancing"
            ])
        else:
            insights['recommendations'].extend([
                "📉 Consider defensive positions",
                "💰 Gradual accumulation on dips",
                "⚖️ Maintain balanced portfolio",
                "📊 Review sector allocation"
            ])
    
    return insights

# Main app code
[Previous sidebar code remains the same]

if uploaded_file is not None:
    try:
        # Load and process data
        features, original_data = load_and_prepare_data(uploaded_file)
        original_data = create_technical_indicators(original_data)
        
        # Model processing
        model = MarketAnomalyDetector(contamination=contamination)
        model.fit(features)
        predictions = model.predict(features)
        scores = model.anomaly_scores(features)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Analysis", "🎯 Insights", "📊 Technical Indicators", "📑 Details"])
        
        with tab1:
            # Main visualization
            st.plotly_chart(plot_market_analysis(original_data, predictions, scores), use_container_width=True)
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                anomaly_count = sum(predictions == 1)
                st.metric("Anomalies Detected", anomaly_count)
            with col2:
                current_volatility = original_data['Volatility'].iloc[-1] * 100
                st.metric("Current Volatility", f"{current_volatility:.1f}%")
            with col3:
                rsi_value = original_data['RSI'].iloc[-1]
                st.metric("RSI", f"{rsi_value:.1f}")
        
        with tab2:
            # Generate and display insights
            insights = generate_market_insights(original_data, predictions, scores)
            
            # Display market status
            st.header(f"Market Status: {insights['market_status']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Risk Level: {insights['risk_level']}")
                st.write(f"Market Trend: {insights['trend']}")
            with col2:
                if insights['overbought']:
                    st.warning("Market is Overbought")
                elif insights['oversold']:
                    st.warning("Market is Oversold")
                st.write(f"Anomaly Score: {insights['anomaly_score']:.2f}")
            
            # Display recommendations
            st.subheader("Recommended Actions:")
            for rec in insights['recommendations']:
                st.write(rec)
        
        with tab3:
            # Technical indicators plots
            fig = px.line(original_data, y=['RSI'], title='Relative Strength Index (RSI)')
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.line(original_data, y=['Volatility'], title='Volatility')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Detailed view of data and predictions
            st.dataframe(pd.DataFrame({
                'Date': original_data.index,
                'Close': original_data['Close'],
                'Anomaly': predictions,
                'Anomaly Score': scores,
                'RSI': original_data['RSI'],
                'Volatility': original_data['Volatility']
            }))
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your data format and try again.")

[Previous welcome message and footer code remains the same]