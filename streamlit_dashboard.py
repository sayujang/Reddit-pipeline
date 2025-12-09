import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import os
from datetime import datetime, timedelta
from collections import Counter
import re

st.set_page_config(
    page_title="Nepal Gen Z Protest Sentiment Dashboard",
    page_icon="🇳🇵",
    layout="wide"
)

@st.cache_resource
def get_db_connection():
    """Create database connection"""
    db_host = os.getenv('POSTGRES_HOST', 'postgres')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = 'airflow_reddit'
    db_user = 'postgres'
    db_password = 'postgres'
    
    connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return engine
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

@st.cache_data(ttl=300)
def load_data(days_back=7):
    """Load data from Postgres"""
    engine = get_db_connection()
    if engine is None:
        return None
    
    try:
        query = f"""
        SELECT * FROM reddit_sentiment
        WHERE processed_at >= NOW() - INTERVAL '{days_back} days'
        ORDER BY created_utc DESC
        """
        df = pd.read_sql(query, engine)
        
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_datetime(df['created_utc'])
        if 'processed_at' in df.columns:
            df['processed_at'] = pd.to_datetime(df['processed_at'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Title
st.title("🇳🇵 Nepal Gen Z Protest: Multi-Dimensional Sentiment Analysis")
st.markdown("""
This dashboard tracks the emotional pulse of Nepal's Gen Z protest movement through Reddit discourse.
It analyzes sentiment evolution, emotional intensity, narrative framing, and community engagement.
""")

# Check connection
if get_db_connection() is None:
    st.error("❌ Cannot connect to database")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Filters")
days_back = st.sidebar.slider("Days of data", 1, 30, 7)
min_score = st.sidebar.number_input("Minimum post score", value=0)
min_engagement = st.sidebar.slider("Minimum engagement weight", 0.0, 10.0, 0.0)

# Load data
with st.spinner("Loading data..."):
    df = load_data(days_back)

if df is None or len(df) == 0:
    st.warning("No data available. Run the ETL pipeline first.")
    st.stop()

# Apply filters
df = df[df['score'] >= min_score]
if 'engagement_weight' in df.columns:
    df = df[df['engagement_weight'] >= min_engagement]

if len(df) == 0:
    st.warning("No data matches filters.")
    st.stop()

# === KEY METRICS ===
st.header("📊 Overview Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Posts", len(df))

with col2:
    avg_sentiment = df['sentiment_score'].mean()
    st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

with col3:
    if 'engagement_weight' in df.columns:
        avg_engagement = df['engagement_weight'].mean()
        st.metric("Avg Engagement", f"{avg_engagement:.2f}")

with col4:
    if 'dominant_emotion' in df.columns:
        top_emotion = df['dominant_emotion'].mode()[0] if len(df) > 0 else 'N/A'
        st.metric("Dominant Emotion", top_emotion.title())

with col5:
    total_comments = df['num_comments'].sum()
    st.metric("Total Comments", f"{total_comments:,}")

# === SENTIMENT EVOLUTION OVER TIME ===
st.header("📈 Sentiment Evolution Over Time")

df['date'] = df['created_utc'].dt.date

# Daily weighted sentiment
if 'weighted_sentiment' in df.columns:
    daily_weighted = df.groupby('date').agg({
        'weighted_sentiment': 'sum',
        'engagement_weight': 'sum'
    }).reset_index()
    daily_weighted['avg_weighted_sentiment'] = (
        daily_weighted['weighted_sentiment'] / daily_weighted['engagement_weight']
    )
    
    fig_evolution = go.Figure()
    
    # Add weighted sentiment line
    fig_evolution.add_trace(go.Scatter(
        x=daily_weighted['date'],
        y=daily_weighted['avg_weighted_sentiment'],
        mode='lines+markers',
        name='Weighted Sentiment',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=8)
    ))
    
    # Add zero line
    fig_evolution.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig_evolution.update_layout(
        title="Engagement-Weighted Sentiment Over Time (Higher weight = more influential posts)",
        xaxis_title="Date",
        yaxis_title="Weighted Sentiment Score",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_evolution, use_container_width=True)

# === EMOTION ANALYSIS ===
st.header("😢😡😊 Emotional Intensity Tracking")

emotion_cols = ['emotion_anger', 'emotion_fear', 'emotion_joy', 
                'emotion_sadness', 'emotion_frustration', 'emotion_hope']

if all(col in df.columns for col in emotion_cols):
    col1, col2 = st.columns(2)
    
    with col1:
        # Average emotion intensity
        avg_emotions = df[emotion_cols].mean()
        avg_emotions.index = avg_emotions.index.str.replace('emotion_', '').str.title()
        
        fig_emotions = px.bar(
            x=avg_emotions.index,
            y=avg_emotions.values,
            title="Average Emotional Intensity",
            labels={'x': 'Emotion', 'y': 'Intensity (0-1)'},
            color=avg_emotions.values,
            color_continuous_scale=['green', 'yellow', 'red']
        )
        st.plotly_chart(fig_emotions, use_container_width=True)
    
    with col2:
        # Dominant emotion distribution
        if 'dominant_emotion' in df.columns:
            emotion_counts = df['dominant_emotion'].value_counts()
            
            fig_dominant = px.pie(
                values=emotion_counts.values,
                names=[e.title() for e in emotion_counts.index],
                title="Dominant Emotion Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_dominant, use_container_width=True)
    
    # Emotion timeline
    st.subheader("Emotional Trajectory Over Time")
    
    daily_emotions = df.groupby('date')[emotion_cols].mean().reset_index()
    daily_emotions_melted = daily_emotions.melt(
        id_vars='date',
        value_vars=emotion_cols,
        var_name='emotion',
        value_name='intensity'
    )
    daily_emotions_melted['emotion'] = daily_emotions_melted['emotion'].str.replace('emotion_', '').str.title()
    
    fig_emotion_timeline = px.line(
        daily_emotions_melted,
        x='date',
        y='intensity',
        color='emotion',
        title="How Emotions Evolved During the Protest Period",
        labels={'intensity': 'Emotion Intensity', 'date': 'Date'}
    )
    st.plotly_chart(fig_emotion_timeline, use_container_width=True)

# === NARRATIVE FRAMING ===
st.header("📰 Narrative Frame Analysis")

if 'narrative_frame' in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Narrative distribution
        narrative_counts = df['narrative_frame'].value_counts()
        narrative_labels = narrative_counts.index.str.replace('_', ' ').str.title()
        
        fig_narratives = px.bar(
            x=narrative_counts.values,
            y=narrative_labels,
            orientation='h',
            title="Which Narratives Dominated the Discourse?",
            labels={'x': 'Number of Posts', 'y': 'Narrative Frame'},
            color=narrative_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_narratives, use_container_width=True)
    
    with col2:
        # Sentiment by narrative
        narrative_sentiment = df.groupby('narrative_frame')['sentiment_score'].mean().sort_values()
        narrative_sentiment.index = narrative_sentiment.index.str.replace('_', ' ').str.title()
        
        fig_narrative_sent = px.bar(
            x=narrative_sentiment.values,
            y=narrative_sentiment.index,
            orientation='h',
            title="Average Sentiment by Narrative Frame",
            labels={'x': 'Avg Sentiment Score', 'y': 'Narrative'},
            color=narrative_sentiment.values,
            color_continuous_scale='RdYlGn'
        )
        fig_narrative_sent.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_narrative_sent, use_container_width=True)
    
    # Narrative evolution
    st.subheader("Narrative Shifts Over Time")
    
    daily_narratives = df.groupby(['date', 'narrative_frame']).size().reset_index(name='count')
    daily_narratives['narrative_frame'] = daily_narratives['narrative_frame'].str.replace('_', ' ').str.title()
    
    fig_narrative_timeline = px.area(
        daily_narratives,
        x='date',
        y='count',
        color='narrative_frame',
        title="How Conversation Topics Shifted During the Protest",
        labels={'count': 'Number of Posts', 'date': 'Date'}
    )
    st.plotly_chart(fig_narrative_timeline, use_container_width=True)

# === ENGAGEMENT ANALYSIS ===
st.header("🔥 Engagement & Influence")

col1, col2 = st.columns(2)

with col1:
    # Sentiment vs engagement scatter
    if 'engagement_weight' in df.columns:
        fig_scatter = px.scatter(
            df,
            x='sentiment_score',
            y='engagement_weight',
            color='sentiment',
            size='score',
            hover_data=['title', 'author'],
            title="Sentiment vs. Engagement (bubble size = score)",
            labels={'sentiment_score': 'Sentiment Score', 'engagement_weight': 'Engagement Weight'},
            color_discrete_map={
                'POSITIVE': '#00CC96',
                'NEUTRAL': '#FFA15A',
                'NEGATIVE': '#EF553B'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # Top engaged posts by sentiment
    if 'engagement_weight' in df.columns:
        top_engaged = df.nlargest(10, 'engagement_weight')[['title', 'sentiment', 'engagement_weight', 'score']]
        top_engaged['title'] = top_engaged['title'].str[:50] + '...'
        
        st.subheader("Most Influential Posts")
        st.dataframe(top_engaged, use_container_width=True)

# === KEYWORD ANALYSIS ===
st.header("🔑 Key Topics & Keywords")

if 'keywords' in df.columns:
    # Extract all keywords
    all_keywords = []
    for kw_str in df['keywords'].dropna():
        if kw_str:
            all_keywords.extend(kw_str.split(','))
    
    if all_keywords:
        keyword_counts = Counter(all_keywords).most_common(20)
        kw_df = pd.DataFrame(keyword_counts, columns=['Keyword', 'Frequency'])
        
        fig_keywords = px.bar(
            kw_df,
            x='Frequency',
            y='Keyword',
            orientation='h',
            title="Top 20 Keywords in Protest Discussions",
            color='Frequency',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_keywords, use_container_width=True)

# === COMPARATIVE ANALYSIS ===
st.header("⚖️ Comparative Insights")

col1, col2 = st.columns(2)

with col1:
    # Positive vs Negative posts characteristics
    st.subheader("Positive vs Negative Posts")
    
    comparison_data = []
    for sentiment in ['POSITIVE', 'NEGATIVE']:
        subset = df[df['sentiment'] == sentiment]
        if len(subset) > 0:
            comparison_data.append({
                'Sentiment': sentiment,
                'Count': len(subset),
                'Avg Score': subset['score'].mean(),
                'Avg Comments': subset['num_comments'].mean(),
                'Avg Engagement': subset['engagement_weight'].mean() if 'engagement_weight' in subset.columns else 0
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True)

with col2:
    # Subjectivity analysis
    if 'subjectivity' in df.columns:
        st.subheader("Subjectivity Distribution")
        st.markdown("*Higher = more opinion-based, Lower = more fact-based*")
        
        fig_subj = px.histogram(
            df,
            x='subjectivity',
            nbins=30,
            title="Are Posts Opinion-Based or Factual?",
            labels={'subjectivity': 'Subjectivity Score (0=Objective, 1=Subjective)'},
            color_discrete_sequence=['#AB63FA']
        )
        st.plotly_chart(fig_subj, use_container_width=True)

# === EXTREME POSTS ===
st.header("📌 Notable Posts")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Most Positive")
    top_pos = df.nlargest(3, 'sentiment_score')[['title', 'sentiment_score', 'score']]
    for idx, row in top_pos.iterrows():
        st.info(f"**Score: {row['sentiment_score']:.2f}** ({row['score']} upvotes)\n\n{row['title'][:100]}...")

with col2:
    st.subheader("Most Negative")
    top_neg = df.nsmallest(3, 'sentiment_score')[['title', 'sentiment_score', 'score']]
    for idx, row in top_neg.iterrows():
        st.error(f"**Score: {row['sentiment_score']:.2f}** ({row['score']} upvotes)\n\n{row['title'][:100]}...")

with col3:
    st.subheader("Most Engaged")
    if 'engagement_weight' in df.columns:
        top_eng = df.nlargest(3, 'engagement_weight')[['title', 'engagement_weight', 'score']]
        for idx, row in top_eng.iterrows():
            st.success(f"**Weight: {row['engagement_weight']:.2f}** ({row['score']} upvotes)\n\n{row['title'][:100]}...")

# === RAW DATA EXPLORER ===
st.header("🔍 Data Explorer")

with st.expander("View Raw Data"):
    # Column selector
    available_cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Select columns to display",
        available_cols,
        default=['created_utc', 'title', 'sentiment', 'sentiment_score', 
                 'dominant_emotion', 'narrative_frame', 'score', 'num_comments'][:len(available_cols)]
    )
    
    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset as CSV",
            data=csv,
            file_name=f"nepal_protest_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# === INSIGHTS & INTERPRETATION ===
st.header("💡 Key Insights")

with st.expander("Click to see automated insights"):
    insights = []
    
    # Sentiment trend
    if 'date' in df.columns and len(df) > 1:
        df_sorted = df.sort_values('date')
        first_week_sentiment = df_sorted.head(len(df)//2)['sentiment_score'].mean()
        second_week_sentiment = df_sorted.tail(len(df)//2)['sentiment_score'].mean()
        
        if second_week_sentiment > first_week_sentiment:
            insights.append(f"✅ Sentiment improved over time (from {first_week_sentiment:.2f} to {second_week_sentiment:.2f})")
        else:
            insights.append(f"⚠️ Sentiment declined over time (from {first_week_sentiment:.2f} to {second_week_sentiment:.2f})")
    
    # Dominant emotion
    if 'dominant_emotion' in df.columns:
        top_emotion = df['dominant_emotion'].mode()[0]
        emotion_pct = (df['dominant_emotion'] == top_emotion).sum() / len(df) * 100
        insights.append(f"😢 **{top_emotion.title()}** was the dominant emotion in {emotion_pct:.1f}% of posts")
    
    # Top narrative
    if 'narrative_frame' in df.columns:
        top_narrative = df['narrative_frame'].mode()[0]
        narrative_pct = (df['narrative_frame'] == top_narrative).sum() / len(df) * 100
        insights.append(f"📰 **{top_narrative.replace('_', ' ').title()}** was the most common narrative ({narrative_pct:.1f}% of posts)")
    
    # Engagement vs sentiment
    if 'engagement_weight' in df.columns and 'sentiment_score' in df.columns:
        corr = df[['engagement_weight', 'sentiment_score']].corr().iloc[0, 1]
        if abs(corr) > 0.3:
            if corr > 0:
                insights.append(f"🔥 Positive posts received MORE engagement (correlation: {corr:.2f})")
            else:
                insights.append(f"🔥 Negative posts received MORE engagement (correlation: {corr:.2f})")
    
    for insight in insights:
        st.markdown(f"- {insight}")

# Footer
st.markdown("---")
if 'processed_at' in df.columns:
    last_update = df['processed_at'].max()
    st.markdown(f"**Last updated:** {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("📊 Multi-dimensional sentiment analysis | Data from r/Nepal | Built with Streamlit & Airflow")