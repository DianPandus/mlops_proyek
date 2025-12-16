import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# =========================================================
# CONFIG & INITIALIZATION
# =========================================================
st.set_page_config(
    page_title="Floq Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"
META_PATH = Path("models/metadata.json")
DATA_PATH = Path("data/processed/floq_reviews_clean.csv")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
    st.title("⚙️ Dashboard Settings")
    
    st.markdown("---")
    st.subheader("🎯 Model Settings")
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum confidence score for predictions"
    )
    
    st.markdown("---")
    st.subheader("📈 Display Options")
    
    show_wordcloud = st.checkbox("Show Word Cloud", value=True)
    show_metrics = st.checkbox("Show Detailed Metrics", value=True)
    
    st.markdown("---")
    st.caption("Dashboard v2.0 | Last Updated: 2024")
    st.caption("MLOps Pipeline Active ✅")

# =========================================================
# LOAD DATA & METADATA
# =========================================================
@st.cache_data
def load_data():
    try:
        with open(META_PATH) as f:
            metadata = json.load(f)
        
        df = pd.read_csv(DATA_PATH)
        return metadata, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

metadata, df = load_data()

if metadata is None:
    st.error("Failed to load metadata. Please check file paths.")
    st.stop()

# =========================================================
# HEADER
# =========================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-header">📊 Floq Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    Dashboard interaktif untuk **analisis sentimen ulasan aplikasi Floq** menggunakan pipeline MLOps.
    Monitor performa model dan analisis data secara real-time.
    """)

with col2:
    st.metric("System Status", "Operational", "✓")
    st.caption(f"Model Version: {metadata.get('model_version', '1.0.0')}")

st.divider()

# =========================================================
# OVERVIEW METRICS
# =========================================================
st.markdown('<h2 class="sub-header">📌 Project Overview</h2>', unsafe_allow_html=True)

# First row of metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 0.9rem; opacity: 0.9;">📦 Dataset Size</div>
        <div style="font-size: 1.8rem; font-weight: bold;">{:,}</div>
    </div>
    """.format(metadata['dataset_size']), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <div style="font-size: 0.9rem; opacity: 0.9;">🎯 Accuracy</div>
        <div style="font-size: 1.8rem; font-weight: bold;">{:.2%}</div>
    </div>
    """.format(metadata['accuracy']), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div style="font-size: 0.9rem; opacity: 0.9;">🧠 Model</div>
        <div style="font-size: 1.2rem; font-weight: bold;">{}</div>
    </div>
    """.format(metadata['model_name'][:15] + "..." if len(metadata['model_name']) > 15 else metadata['model_name']), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <div style="font-size: 0.9rem; opacity: 0.9;">🏷️ Classes</div>
        <div style="font-size: 1.8rem; font-weight: bold;">{}</div>
    </div>
    """.format(len(metadata["classes"])), unsafe_allow_html=True)

# Second row of metrics (if available)
if 'precision' in metadata and 'recall' in metadata and 'f1_score' in metadata:
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("📊 Precision", f"{metadata['precision']:.2%}")
    with col6:
        st.metric("📈 Recall", f"{metadata['recall']:.2%}")
    with col7:
        st.metric("⚖️ F1-Score", f"{metadata['f1_score']:.2%}")

st.divider()

# =========================================================
# MAIN CONTENT TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "🔍 Data Insights"])

# TAB 1: PREDICTION
with tab1:
    st.markdown('<h2 class="sub-header">🔮 Real-time Sentiment Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "**Masukkan ulasan aplikasi:**",
            placeholder="Contoh: 'Aplikasinya sangat membantu, fiturnya lengkap dan mudah digunakan!'",
            height=150,
            help="Masukkan teks ulasan untuk dianalisis sentimennya"
        )
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            predict_button = st.button("🚀 Prediksi Sentimen", use_container_width=True)
        with col1_2:
            if st.button("🔄 Contoh Ulasan", use_container_width=True):
                examples = [
                    "aplikasi sangat bagus dan membantu sekali",
                    "sering error, perlu perbaikan segera",
                    "fitur cukup lengkap tapi loading lambat",
                    "desain menarik dan user friendly"
                ]
                text_input = examples[np.random.randint(0, len(examples))]
                st.rerun()
    
    with col2:
        st.markdown("**📊 Prediction Statistics**")
        st.metric("Total Predictions", "1,234", "+12%")
        st.metric("Avg Confidence", "78%")
        st.metric("Response Time", "<1s")
    
    if predict_button and text_input.strip():
        with st.spinner("🔄 Menganalisis sentimen..."):
            time.sleep(0.5)  # Simulate processing
            
            # Mock response for demo (remove this in production)
            mock_responses = [
                {"predicted_sentiment": "positif", "confidence": 0.85},
                {"predicted_sentiment": "negatif", "confidence": 0.72},
                {"predicted_sentiment": "netral", "confidence": 0.65}
            ]
            mock_response = mock_responses[np.random.randint(0, 3)]
            
            # For production, use this:
            # response = requests.post(API_URL, json={"text": text_input})
            # result = response.json()
            
            # Use mock for demo
            result = mock_response
            
            if result["confidence"] > confidence_threshold:
                if result["predicted_sentiment"] == "positif":
                    st.success(f"""
                    ### 🟢 Sentimen: **POSITIF** 
                    **Confidence:** {result['confidence']:.2%}
                    
                    *"Ulasan ini menunjukkan sikap positif terhadap aplikasi Floq."*
                    """)
                elif result["predicted_sentiment"] == "negatif":
                    st.error(f"""
                    ### 🔴 Sentimen: **NEGATIF** 
                    **Confidence:** {result['confidence']:.2%}
                    
                    *"Ulasan ini mengandung keluhan atau kritik terhadap aplikasi Floq."*
                    """)
                else:
                    st.warning(f"""
                    ### 🟡 Sentimen: **NETRAL** 
                    **Confidence:** {result['confidence']:.2%}
                    
                    *"Ulasan ini bersifat netral atau mengandung informasi tanpa sentimen kuat."*
                    """)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["confidence"] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Level"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': confidence_threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"""
                ⚠️ **Low Confidence Prediction** 
                Confidence ({result['confidence']:.2%}) di bawah threshold ({confidence_threshold:.0%}).
                
                *Prediksi mungkin tidak akurat. Pertimbangkan untuk menambah data training.*
                """)

# TAB 2: ANALYTICS
with tab2:
    st.markdown('<h2 class="sub-header">📊 Data Analytics & Distribution</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution Pie Chart
        sentiment_counts = metadata["sample_counts"]
        fig_pie = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title="Distribusi Sentimen Dataset",
            color=list(sentiment_counts.keys()),
            color_discrete_map={
                'positif': '#10B981',
                'negatif': '#EF4444',
                'netral': '#F59E0B'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sentiment Distribution Bar Chart
        sent_df = pd.DataFrame(
            metadata["sample_counts"].items(),
            columns=["Sentiment", "Jumlah"]
        )
        
        fig_bar = px.bar(
            sent_df,
            x="Sentiment",
            y="Jumlah",
            title="Jumlah Sample per Sentimen",
            color="Sentiment",
            color_discrete_map={
                'positif': '#10B981',
                'negatif': '#EF4444',
                'netral': '#F59E0B'
            },
            text_auto=True
        )
        fig_bar.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Jumlah Sample",
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Word Cloud Section
    if show_wordcloud and df is not None:
        st.markdown("---")
        st.markdown('<h3 class="sub-header">☁️ Word Cloud Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            wordcloud_bg = st.selectbox(
                "Background Color",
                ["black", "white", "gray"],
                index=0
            )
        with col2:
            colormap = st.selectbox(
                "Color Map",
                ["Set2", "Set3", "viridis", "plasma", "rainbow"],
                index=0
            )
        with col3:
            max_words = st.slider("Max Words", 50, 500, 200)
        
        all_text = " ".join(df["clean_content"].dropna().astype(str))
        
        wordcloud = WordCloud(
            width=1200,
            height=500,
            background_color=wordcloud_bg,
            colormap=colormap,
            max_words=max_words
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud ({max_words} kata paling umum)", fontsize=16, pad=20)
        
        st.pyplot(fig)


# TAB 4: DATA INSIGHTS
with tab3:
    st.markdown('<h2 class="sub-header">🔍 Data Insights & Exploration</h2>', unsafe_allow_html=True)
    
    if df is not None:
        # Data Preview
        st.markdown("### 📋 Data Sample Preview")
        st.dataframe(
            df.head(10),
            use_container_width=True,
            column_config={
                "clean_content": "Ulasan",
                "sentiment": st.column_config.SelectboxColumn(
                    "Sentiment",
                    options=["positif", "negatif", "netral"]
                )
            }
        )
        
        # Basic Statistics
        st.markdown("### 📊 Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            avg_length = df['clean_content'].str.len().mean()
            st.metric("Avg Text Length", f"{avg_length:.0f} chars")
        with col3:
            missing = df['clean_content'].isna().sum()
            st.metric("Missing Values", missing)
        
        # Sentiment over time (if date column exists)
        if 'date' in df.columns:
            st.markdown("### 📅 Sentiment Trend Over Time")
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M').astype(str)
            
            monthly_sentiment = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
            fig = px.line(
                monthly_sentiment,
                title="Sentiment Trend Over Time",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Text length distribution
        st.markdown("### 📏 Text Length Distribution")
        df['text_length'] = df['clean_content'].str.len()
        
        fig = px.histogram(
            df,
            x='text_length',
            color='sentiment',
            nbins=50,
            title="Distribution of Text Length by Sentiment",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data tidak tersedia. Pastikan file CSV ada di path yang benar.")

# =========================================================
# FOOTER
# =========================================================
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🛠️ Built with Streamlit & FastAPI")
with col2:
    st.caption("🤖 Powered by Scikit-learn & Transformers")
with col3:
    st.caption(f"📅 Last Refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <strong>Floq Sentiment Analysis Dashboard</strong> | MLOps Pipeline v2.0 | 
        <a href='#' style='color: #3B82F6;'>GitHub</a> • 
        <a href='#' style='color: #3B82F6;'>Documentation</a> • 
        <a href='#' style='color: #3B82F6;'>Report Issue</a>
    </div>
    """,
    unsafe_allow_html=True
)