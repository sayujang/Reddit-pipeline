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
def load_data(view_mode='realtime', days_back=7):
    """
    Load data from Postgres based on view mode
    
    Args:
        view_mode: 'realtime', 'protest_phases', or 'all'
        days_back: For realtime mode only
    """
    engine = get_db_connection()
    if engine is None:
        return None
    
    try:
        if view_mode == 'realtime':
            query = f"""
            SELECT * FROM reddit_sentiment
            WHERE phase = 'realtime'
            AND processed_at >= NOW() - INTERVAL '{days_back} days'
            ORDER BY created_utc DESC
            """
        elif view_mode == 'protest_phases':
            query = """
            SELECT * FROM reddit_sentiment
            WHERE phase IN ('before', 'during', 'after')
            ORDER BY created_utc
            """
        else:  # all
            query = """
            SELECT * FROM reddit_sentiment
            ORDER BY created_utc DESC
            """
        
        df = pd.read_sql(query, engine)
        
        if len(df) == 0:
            return None
            
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
Analyze sentiment evolution, emotional intensity, narrative framing, and community engagement.
""")

# Check connection
if get_db_connection() is None:
    st.error("❌ Cannot connect to database")
    st.stop()

# ===== SIDEBAR WITH VIEW MODE =====
st.sidebar.header("📊 View Mode")
view_mode = st.sidebar.radio(
    "Select Data View",
    options=['protest_phases', 'realtime'],
    format_func=lambda x: {
        'protest_phases': '🔍 Protest Phase Analysis (Before/During/After)',
        'realtime': '🕒 Real-time Data (Daily Updates)'
    }[x]
)

st.sidebar.markdown("---")

# Conditional filters based on view mode
if view_mode == 'protest_phases':
    st.sidebar.header("⚙️ Phase Filters")
    
    phases_to_show = st.sidebar.multiselect(
        "Select Phases to Compare",
        options=['before', 'during', 'after'],
        default=['before', 'during', 'after'],
        format_func=lambda x: {
            'before': '📅 Before Protest',
            'during': '🔥 During Protest',
            'after': '📊 After Protest'
        }[x]
    )
    
    st.sidebar.info("📌 **Protest Phase Analysis**: Compare sentiment across three critical periods of the Gen Z movement")
    
    days_back = None  # Not used for protest phases
    
else:  # realtime
    st.sidebar.header("⚙️ Time Filters")
    days_back = st.sidebar.slider("Days of recent data", 1, 30, 7)
    phases_to_show = None  # Not used for realtime
    
    st.sidebar.info("📌 **Real-time Analysis**: Track current sentiment trends from daily Reddit scraping")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Content Filters")
min_score = st.sidebar.number_input("Minimum post score", value=0)
min_engagement = st.sidebar.slider("Minimum engagement weight", 0.0, 10.0, 0.0)

# Load data based on view mode
with st.spinner("Loading data..."):
    df = load_data(view_mode=view_mode, days_back=days_back)

if df is None or len(df) == 0:
    st.warning(f"⚠️ No data available for {view_mode} mode. Please run the appropriate pipeline first.")
    if view_mode == 'protest_phases':
        st.info("Run the **'process_protest_phase_data'** DAG from Airflow to process historical data")
    else:
        st.info("Run the **'etl_reddit_pipeline_realtime'** DAG from Airflow for daily updates")
    st.stop()

# Filter by selected phases (for protest mode)
if view_mode == 'protest_phases' and phases_to_show:
    df = df[df['phase'].isin(phases_to_show)]

# Apply content filters
df = df[df['score'] >= min_score]
if 'engagement_weight' in df.columns:
    df = df[df['engagement_weight'] >= min_engagement]

if len(df) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# ===== PHASE INDICATOR =====
if view_mode == 'protest_phases':
    st.info(f"📊 Analyzing **{len(df)} posts** across **{len(phases_to_show)} protest phase(s)**: {', '.join([p.title() for p in phases_to_show])}")
else:
    st.info(f"📊 Analyzing **{len(df)} posts** from the last **{days_back} days** of real-time data")

# === KEY METRICS ===
st.header("📊 Overview Metrics")

if view_mode == 'protest_phases':
    # Show metrics per phase
    phase_cols = st.columns(len(phases_to_show) if phases_to_show else 1)
    
    for idx, phase in enumerate(phases_to_show if phases_to_show else ['all']):
        phase_df = df[df['phase'] == phase] if view_mode == 'protest_phases' else df
        
        with phase_cols[idx]:
            st.subheader(f"{phase.title()} Phase")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Posts", len(phase_df))
            with col2:
                avg_sent = phase_df['sentiment_score'].mean()
                st.metric("Avg Sentiment", f"{avg_sent:.2f}")
            
            col3, col4 = st.columns(2)
            with col3:
                if 'dominant_emotion' in phase_df.columns:
                    top_emotion = phase_df['dominant_emotion'].mode()[0] if len(phase_df) > 0 else 'N/A'
                    st.metric("Top Emotion", top_emotion.title())
            with col4:
                total_comments = phase_df['num_comments'].sum()
                st.metric("Comments", f"{total_comments:,}")
else:
    # Standard metrics for realtime
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

# === SENTIMENT COMPARISON ACROSS PHASES ===
if view_mode == 'protest_phases' and len(phases_to_show) > 1:
    st.header("📊 Cross-Phase Sentiment Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average sentiment by phase
        phase_sentiment = df.groupby('phase')['sentiment_score'].mean().reindex(phases_to_show)
        
        fig_phase_sentiment = px.bar(
            x=[p.title() for p in phase_sentiment.index],
            y=phase_sentiment.values,
            title="Average Sentiment Score by Protest Phase",
            labels={'x': 'Phase', 'y': 'Avg Sentiment Score'},
            color=phase_sentiment.values,
            color_continuous_scale='RdYlGn'
        )
        fig_phase_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_phase_sentiment, use_container_width=True)
    
    with col2:
        # Sentiment distribution by phase
        phase_sent_dist = df.groupby(['phase', 'sentiment']).size().reset_index(name='count')
        phase_sent_dist['phase'] = phase_sent_dist['phase'].str.title()
        
        fig_phase_dist = px.bar(
            phase_sent_dist,
            x='phase',
            y='count',
            color='sentiment',
            title="Sentiment Distribution Across Phases",
            labels={'count': 'Number of Posts', 'phase': 'Phase'},
            color_discrete_map={
                'POSITIVE': '#00CC96',
                'NEUTRAL': '#FFA15A',
                'NEGATIVE': '#EF553B'
            },
            barmode='group'
        )
        st.plotly_chart(fig_phase_dist, use_container_width=True)

# === SENTIMENT EVOLUTION OVER TIME ===
st.header("📈 Sentiment Evolution Over Time")

df['date'] = df['created_utc'].dt.date

if view_mode == 'protest_phases':
    # Timeline with phase markers
    daily_sentiment = df.groupby(['date', 'phase']).agg({
        'sentiment_score': 'mean'
    }).reset_index()
    daily_sentiment['phase'] = daily_sentiment['phase'].str.title()
    
    fig_timeline = px.line(
        daily_sentiment,
        x='date',
        y='sentiment_score',
        color='phase',
        title="Sentiment Timeline Across Protest Phases",
        labels={'sentiment_score': 'Avg Sentiment Score', 'date': 'Date'},
        markers=True,
        color_discrete_map={
            'Before': '#636EFA',
            'During': '#EF553B',
            'After': '#00CC96'
        }
    )
    fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
else:
    # Weighted sentiment for realtime
    if 'weighted_sentiment' in df.columns:
        daily_weighted = df.groupby('date').agg({
            'weighted_sentiment': 'sum',
            'engagement_weight': 'sum'
        }).reset_index()
        daily_weighted['avg_weighted_sentiment'] = (
            daily_weighted['weighted_sentiment'] / daily_weighted['engagement_weight']
        )
        
        fig_evolution = go.Figure()
        
        fig_evolution.add_trace(go.Scatter(
            x=daily_weighted['date'],
            y=daily_weighted['avg_weighted_sentiment'],
            mode='lines+markers',
            name='Weighted Sentiment',
            line=dict(color='#636EFA', width=3),
            marker=dict(size=8)
        ))
        
        fig_evolution.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_evolution.update_layout(
            title="Engagement-Weighted Sentiment Over Time",
            xaxis_title="Date",
            yaxis_title="Weighted Sentiment Score",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True)

# === EMOTION ANALYSIS ===
st.header("😢😡😊 Emotional Intensity Analysis")

emotion_cols = ['emotion_anger', 'emotion_fear', 'emotion_joy', 
                'emotion_sadness', 'emotion_frustration', 'emotion_hope']

if all(col in df.columns for col in emotion_cols):
    if view_mode == 'protest_phases' and len(phases_to_show) > 1:
        # Compare emotions across phases
        st.subheader("Emotional Intensity Comparison Across Phases")
        
        phase_emotions = df.groupby('phase')[emotion_cols].mean()
        phase_emotions.index = phase_emotions.index.str.title()
        phase_emotions.columns = phase_emotions.columns.str.replace('emotion_', '').str.title()
        
        fig_phase_emotions = go.Figure()
        
        for phase in phase_emotions.index:
            fig_phase_emotions.add_trace(go.Scatterpolar(
                r=phase_emotions.loc[phase].values,
                theta=phase_emotions.columns,
                fill='toself',
                name=phase
            ))
        
        fig_phase_emotions.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Emotional Profile by Protest Phase (Radar Chart)",
            showlegend=True
        )
        st.plotly_chart(fig_phase_emotions, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average emotion intensity
        if view_mode == 'protest_phases':
            # Group by phase
            for phase in phases_to_show:
                phase_df = df[df['phase'] == phase]
                avg_emotions = phase_df[emotion_cols].mean()
                avg_emotions.index = avg_emotions.index.str.replace('emotion_', '').str.title()
                
                fig = px.bar(
                    x=avg_emotions.index,
                    y=avg_emotions.values,
                    title=f"Emotions in {phase.title()} Phase",
                    labels={'x': 'Emotion', 'y': 'Intensity'},
                    color=avg_emotions.values,
                    color_continuous_scale=['green', 'yellow', 'red']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
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
            if view_mode == 'protest_phases':
                emotion_by_phase = df.groupby(['phase', 'dominant_emotion']).size().reset_index(name='count')
                emotion_by_phase['phase'] = emotion_by_phase['phase'].str.title()
                emotion_by_phase['dominant_emotion'] = emotion_by_phase['dominant_emotion'].str.title()
                
                fig_emotions_phase = px.bar(
                    emotion_by_phase,
                    x='phase',
                    y='count',
                    color='dominant_emotion',
                    title="Dominant Emotions Across Phases",
                    labels={'count': 'Number of Posts'},
                    barmode='stack'
                )
                st.plotly_chart(fig_emotions_phase, use_container_width=True)
            else:
                emotion_counts = df['dominant_emotion'].value_counts()
                
                fig_dominant = px.pie(
                    values=emotion_counts.values,
                    names=[e.title() for e in emotion_counts.index],
                    title="Dominant Emotion Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_dominant, use_container_width=True)

# === NARRATIVE FRAMING ===
st.header("📰 Narrative Frame Analysis")

if 'narrative_frame' in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        if view_mode == 'protest_phases' and len(phases_to_show) > 1:
            # Narratives by phase
            narrative_by_phase = df.groupby(['phase', 'narrative_frame']).size().reset_index(name='count')
            narrative_by_phase['phase'] = narrative_by_phase['phase'].str.title()
            narrative_by_phase['narrative_frame'] = narrative_by_phase['narrative_frame'].str.replace('_', ' ').str.title()
            
            fig_narratives = px.bar(
                narrative_by_phase,
                x='phase',
                y='count',
                color='narrative_frame',
                title="Narrative Frames Across Protest Phases",
                labels={'count': 'Number of Posts'},
                barmode='stack'
            )
            st.plotly_chart(fig_narratives, use_container_width=True)
        else:
            narrative_counts = df['narrative_frame'].value_counts()
            narrative_labels = narrative_counts.index.str.replace('_', ' ').str.title()
            
            fig_narratives = px.bar(
                x=narrative_counts.values,
                y=narrative_labels,
                orientation='h',
                title="Narrative Frame Distribution",
                labels={'x': 'Number of Posts', 'y': 'Narrative Frame'},
                color=narrative_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_narratives, use_container_width=True)
    
    with col2:
        # Sentiment by narrative
        if view_mode == 'protest_phases':
            narrative_sentiment_phase = df.groupby(['phase', 'narrative_frame'])['sentiment_score'].mean().reset_index()
            narrative_sentiment_phase['phase'] = narrative_sentiment_phase['phase'].str.title()
            narrative_sentiment_phase['narrative_frame'] = narrative_sentiment_phase['narrative_frame'].str.replace('_', ' ').str.title()
            
            fig_narrative_sent = px.bar(
                narrative_sentiment_phase,
                x='narrative_frame',
                y='sentiment_score',
                color='phase',
                title="Sentiment by Narrative & Phase",
                labels={'sentiment_score': 'Avg Sentiment'},
                barmode='group'
            )
            fig_narrative_sent.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_narrative_sent, use_container_width=True)
        else:
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

# === KEYWORD ANALYSIS ===
st.header("🔑 Key Topics & Keywords")

if 'keywords' in df.columns:
    if view_mode == 'protest_phases' and len(phases_to_show) > 1:
        st.subheader("Keywords by Protest Phase")
        
        phase_cols = st.columns(len(phases_to_show))
        
        for idx, phase in enumerate(phases_to_show):
            phase_df = df[df['phase'] == phase]
            all_keywords = []
            for kw_str in phase_df['keywords'].dropna():
                if kw_str:
                    all_keywords.extend(kw_str.split(','))
            
            if all_keywords:
                keyword_counts = Counter(all_keywords).most_common(10)
                kw_df = pd.DataFrame(keyword_counts, columns=['Keyword', 'Frequency'])
                
                with phase_cols[idx]:
                    st.markdown(f"**{phase.title()} Phase**")
                    fig = px.bar(
                        kw_df,
                        x='Frequency',
                        y='Keyword',
                        orientation='h',
                        title=f"Top Keywords",
                        color='Frequency',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
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
                title="Top 20 Keywords",
                color='Frequency',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_keywords, use_container_width=True)

# === KEY INSIGHTS ===
st.header("💡 Key Insights")

with st.expander("Click to see automated insights"):
    insights = []
    
    if view_mode == 'protest_phases' and len(phases_to_show) > 1:
        # Cross-phase insights
        phase_sentiments = df.groupby('phase')['sentiment_score'].mean()
        
        most_positive_phase = phase_sentiments.idxmax()
        most_negative_phase = phase_sentiments.idxmin()
        
        insights.append(f"📊 **{most_positive_phase.title()}** phase had the most positive sentiment ({phase_sentiments[most_positive_phase]:.2f})")
        insights.append(f"📊 **{most_negative_phase.title()}** phase had the most negative sentiment ({phase_sentiments[most_negative_phase]:.2f})")
        
        sentiment_change = phase_sentiments['after'] - phase_sentiments['before'] if 'after' in phase_sentiments.index and 'before' in phase_sentiments.index else None
        if sentiment_change is not None:
            if sentiment_change > 0:
                insights.append(f"✅ Sentiment IMPROVED from before to after the protest (+{sentiment_change:.2f})")
            else:
                insights.append(f"⚠️ Sentiment DECLINED from before to after the protest ({sentiment_change:.2f})")
    else:
        # Realtime insights
        if 'date' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('date')
            first_half_sentiment = df_sorted.head(len(df)//2)['sentiment_score'].mean()
            second_half_sentiment = df_sorted.tail(len(df)//2)['sentiment_score'].mean()
            
            if second_half_sentiment > first_half_sentiment:
                insights.append(f"✅ Sentiment improved recently (from {first_half_sentiment:.2f} to {second_half_sentiment:.2f})")
            else:
                insights.append(f"⚠️ Sentiment declined recently (from {first_half_sentiment:.2f} to {second_half_sentiment:.2f})")
    
    # Common insights
    if 'dominant_emotion' in df.columns:
        top_emotion = df['dominant_emotion'].mode()[0]
        emotion_pct = (df['dominant_emotion'] == top_emotion).sum() / len(df) * 100
        insights.append(f"😢 **{top_emotion.title()}** was the dominant emotion in {emotion_pct:.1f}% of posts")
    
    if 'narrative_frame' in df.columns:
        top_narrative = df['narrative_frame'].mode()[0]
        narrative_pct = (df['narrative_frame'] == top_narrative).sum() / len(df) * 100
        insights.append(f"📰 **{top_narrative.replace('_', ' ').title()}** was the most common narrative ({narrative_pct:.1f}% of posts)")
    
    for insight in insights:
        st.markdown(f"- {insight}")

# === RAW DATA EXPLORER ===
st.header("🔍 Data Explorer")

with st.expander("View Raw Data"):
    available_cols = df.columns.tolist()
    default_cols = ['created_utc', 'title', 'sentiment', 'sentiment_score', 
                    'dominant_emotion', 'narrative_frame', 'score', 'num_comments', 'phase']
    default_cols = [col for col in default_cols if col in available_cols]
    
    selected_cols = st.multiselect(
        "Select columns to display",
        available_cols,
        default=default_cols
    )
    
    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=csv,
            file_name=f"nepal_protest_sentiment_{view_mode}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if 'processed_at' in df.columns:
        last_update = df['processed_at'].max()
        st.markdown(f"**Last updated:** {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.markdown(f"**View Mode:** {view_mode.replace('_', ' ').title()}")
with col3:
    if view_mode == 'protest_phases':
        st.markdown(f"**Phases Shown:** {', '.join([p.title() for p in phases_to_show])}")

st.markdown("📊 Multi-dimensional sentiment analysis | Data from r/Nepal | Built with Streamlit & Airflow")