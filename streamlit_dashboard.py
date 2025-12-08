import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import os
from datetime import datetime, timedelta
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="Nepal Subreddit Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Database connection
@st.cache_resource
def get_db_connection():
    """Create database connection to the same DB as Airflow"""
    # Use environment variable or default to postgres (Docker service name)
    db_host = os.getenv('POSTGRES_HOST', 'postgres')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = 'airflow_reddit'
    db_user = 'postgres'
    db_password = 'postgres'
    
    connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    
    try:
        engine = create_engine(connection_string)
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return engine
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.info(f"Trying to connect to: {db_host}:{db_port}/{db_name}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
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
        
        # Convert timestamp columns
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_datetime(df['created_utc'])
        if 'processed_at' in df.columns:
            df['processed_at'] = pd.to_datetime(df['processed_at'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def check_table_exists():
    """Check if reddit_sentiment table exists"""
    engine = get_db_connection()
    if engine is None:
        return False
    
    try:
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'reddit_sentiment'
        );
        """
        result = pd.read_sql(query, engine)
        return result.iloc[0, 0]
    except Exception as e:
        st.error(f"Error checking table: {str(e)}")
        return False

# Dashboard Title
st.title("🇳🇵 Nepal Subreddit Sentiment Analysis Dashboard")
st.markdown("Real-time sentiment analysis of r/Nepal posts and discussions")

# Check database connection
if get_db_connection() is None:
    st.error("❌ Cannot connect to database. Please ensure:")
    st.markdown("""
    1. Docker containers are running: `docker-compose ps`
    2. Postgres service is healthy
    3. Database 'airflow_reddit' exists
    4. Run the ETL pipeline at least once to create the table
    """)
    st.stop()

# Check if table exists
if not check_table_exists():
    st.warning("⚠️ Table 'reddit_sentiment' does not exist yet.")
    st.info("""
    The table will be created automatically when you run the ETL pipeline.
    
    To run the pipeline:
    1. Go to http://localhost:8080 (Airflow UI)
    2. Login with username: `admin`, password: `admin`
    3. Enable and trigger the `etl_reddit_pipeline` DAG
    4. Wait for all tasks to complete
    5. Refresh this dashboard
    """)
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
days_back = st.sidebar.slider("Days of data to display", 1, 30, 7)
min_score = st.sidebar.number_input("Minimum post score", value=0)

# Load data
try:
    with st.spinner("Loading data from database..."):
        df = load_data(days_back)
    
    if df is None or len(df) == 0:
        st.warning("No data available. Please run the ETL pipeline first.")
        st.info("""
        To generate data:
        1. Go to Airflow UI at http://localhost:8080
        2. Trigger the `etl_reddit_pipeline` DAG
        3. Wait for completion (~5-10 minutes)
        4. Refresh this page
        """)
        st.stop()
    
    # Apply filters
    df = df[df['score'] >= min_score]
    
    if len(df) == 0:
        st.warning("No data matches your filters. Try adjusting the filter values.")
        st.stop()
    
    # Key Metrics Row
    st.header("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", len(df))
    
    with col2:
        avg_sentiment = df['sentiment_score'].mean()
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    
    with col3:
        positive_pct = (df['sentiment'] == 'POSITIVE').sum() / len(df) * 100
        st.metric("Positive Posts", f"{positive_pct:.1f}%")
    
    with col4:
        total_engagement = df['score'].sum()
        st.metric("Total Engagement", f"{total_engagement:,}")
    
    # Sentiment Distribution
    st.header("📊 Sentiment Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        sentiment_counts = df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'POSITIVE': '#00CC96',
                'NEUTRAL': '#FFA15A',
                'NEGATIVE': '#EF553B'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Sentiment Count",
            labels={'x': 'Sentiment', 'y': 'Count'},
            color=sentiment_counts.index,
            color_discrete_map={
                'POSITIVE': '#00CC96',
                'NEUTRAL': '#FFA15A',
                'NEGATIVE': '#EF553B'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Sentiment Over Time
    st.header("📅 Sentiment Trends Over Time")
    
    # Group by date
    df['date'] = df['created_utc'].dt.date
    daily_sentiment = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
    
    fig_timeline = px.line(
        daily_sentiment,
        x='date',
        y='count',
        color='sentiment',
        title="Daily Sentiment Trends",
        color_discrete_map={
            'POSITIVE': '#00CC96',
            'NEUTRAL': '#FFA15A',
            'NEGATIVE': '#EF553B'
        }
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Sentiment Score Distribution
    st.header("📉 Sentiment Score Distribution")
    fig_hist = px.histogram(
        df,
        x='sentiment_score',
        nbins=50,
        title="Distribution of Sentiment Scores",
        labels={'sentiment_score': 'Sentiment Score', 'count': 'Frequency'},
        color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Top Posts by Engagement
    st.header("🔥 Top Posts by Engagement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Positive Posts")
        top_positive = df.nlargest(5, 'sentiment_score')[['title', 'sentiment_score', 'score', 'author']]
        st.dataframe(top_positive, use_container_width=True)
    
    with col2:
        st.subheader("Most Negative Posts")
        top_negative = df.nsmallest(5, 'sentiment_score')[['title', 'sentiment_score', 'score', 'author']]
        st.dataframe(top_negative, use_container_width=True)
    
    # Top Authors by Activity
    st.header("👥 Most Active Authors")
    author_stats = df.groupby('author').agg({
        'id': 'count',
        'score': 'sum',
        'sentiment_score': 'mean'
    }).reset_index()
    author_stats.columns = ['Author', 'Post Count', 'Total Score', 'Avg Sentiment']
    author_stats = author_stats.sort_values('Post Count', ascending=False).head(10)
    
    fig_authors = px.bar(
        author_stats,
        x='Author',
        y='Post Count',
        title="Top 10 Most Active Authors",
        color='Avg Sentiment',
        color_continuous_scale='RdYlGn',
        labels={'Avg Sentiment': 'Avg Sentiment Score'}
    )
    st.plotly_chart(fig_authors, use_container_width=True)
    
    # Word Cloud Data (Top Keywords)
    st.header("🔤 Common Topics")
    if 'combined_text' in df.columns:
        # Simple word frequency analysis
        all_text = ' '.join(df['combined_text'].astype(str))
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                     'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'}
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        word_freq = Counter(words).most_common(20)
        
        if word_freq:
            word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            fig_words = px.bar(
                word_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title="Top 20 Most Common Words"
            )
            st.plotly_chart(fig_words, use_container_width=True)
    
    # Raw Data Explorer
    st.header("🔍 Data Explorer")
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"reddit_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer with last update time
    st.markdown("---")
    if 'processed_at' in df.columns:
        last_update = df['processed_at'].max()
        st.markdown(f"**Last data update:** {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("Dashboard created with Streamlit | Data from r/Nepal")
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check the logs and ensure all services are running properly.")