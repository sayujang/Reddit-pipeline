Collecting workspace information# Project Summary & Documentation

## Project Overview

This is a **comprehensive data pipeline** for analyzing sentiment and emotional dynamics around Nepal's Gen Z protest movement through Reddit discourse. The system combines real-time data collection with historical analysis across three protest phases (before, during, after).

## Architecture

The project follows a modular ETL (Extract, Transform, Load) architecture:

```
Reddit Data → S3 (Raw) → Processing → S3 (Processed) → PostgreSQL → Streamlit Dashboard
```

## Key Components

### 1. **Data Extraction** (redditdag.py)
- Real-time daily extraction from r/Nepal subreddit using PRAW
- Historical data collection for protest phase analysis
- Two Apache Airflow DAGs:
  - `etl_reddit_pipeline_realtime`: Daily extraction and processing
  - `process_protest_phase_data`: Historical phase-based analysis

### 2. **ETL Pipelines**
- **reddit_etl.py**: Reddit API connection and post extraction
- **aws_etl.py**: S3 bucket operations
- **reddit_pipeline.py**: Basic extraction and transformation
- **reddit_processing_pipeline.py**: Advanced sentiment analysis pipeline

### 3. **Advanced Analysis** (reddit_processing_pipeline.py)
- **Sentiment Analysis**: Multi-class classification (POSITIVE/NEUTRAL/NEGATIVE) using DistilBERT
- **Emotion Detection**: Six dimensions tracked:
  - Anger, Fear, Joy, Sadness, Frustration, Hope
- **Narrative Framing**: Classifies posts into frames:
  - Anti-corruption, Youth empowerment, Government criticism, Media freedom, Reform advocacy, General discussion
- **Keyword Extraction**: Protest-related keyword identification
- **Engagement Weighting**: Logarithmic scoring based on post score and comments

### 4. **Data Storage**
- **S3**: Raw and processed CSV files organized by phase
- **PostgreSQL**: Structured data with phase and timestamp metadata
- **Columns tracked**: sentiment, sentiment_score, emotions, narrative_frame, keywords, engagement_weight, subjectivity

### 5. **Dashboard** (streamlit_dashboard.py)
Interactive visualizations including:
- Sentiment timeline across phases
- Emotion intensity analysis (radar charts, bar charts)
- Narrative frame distribution
- Keyword frequency analysis
- Cross-phase comparisons
- Automated insights generation

## Key Features

### Multi-Dimensional Analysis
- **Sentiment Scoring**: Combined with engagement metrics for weighted sentiment
- **Emotional Profiling**: Track emotional intensity across protest phases
- **Narrative Analysis**: Understand how different perspectives frame the protest
- **Subjectivity Scoring**: TextBlob-based objectivity measurement

### Batch Processing
- Processes up to 1,000 records per run with configurable batch sizes
- Handles remaining records with default neutral sentiment
- Graceful error handling with fallback values

### Phase-Based Comparison
```
Before Phase  →  During Phase  →  After Phase
(Sep 1-7)       (Sep 8-9)        (Sep 10+)
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | Apache Airflow 2.7.1 |
| Container | Docker |
| Data Processing | Pandas, NumPy |
| NLP | Transformers (HuggingFace), TextBlob |
| Cloud Storage | AWS S3 |
| Database | PostgreSQL |
| Visualization | Streamlit, Plotly |
| Language | Python 3.11 |

## Setup & Installation

### Prerequisites
- Docker & Docker Compose
- AWS credentials configured
- Reddit API credentials

### Configuration

Update config.conf:
```ini
[api_keys]
reddit_client_id = YOUR_CLIENT_ID
reddit_secret_key = YOUR_SECRET_KEY

[aws]
aws_access_key_id = YOUR_AWS_KEY
aws_secret_access_key = YOUR_AWS_SECRET
aws_region = us-east-1
aws_bucket_name = your-bucket-name

[database]
database_host = postgres
database_port = 5432
database_name = airflow_reddit
database_username = postgres
database_password = postgres
```

### Run with Docker
```bash
docker-compose up --build
```

Access:
- **Airflow UI**: `http://localhost:8080`
- **Streamlit Dashboard**: `http://localhost:8501`

## Data Flow

1. **Extract** → Reddit posts from r/Nepal
2. **Upload** → Store raw CSV in S3 `raw/` folder
3. **Process** → Clean data, run NLP models
4. **Enrich** → Add sentiment, emotions, narratives, keywords
5. **Store** → Save processed data to S3 & PostgreSQL
6. **Visualize** → Interactive dashboard analysis

## Key Insights Generated

- Sentiment trends across protest phases
- Emotional intensity profiles by phase
- Dominant narratives and framing patterns
- High-impact keywords and topics
- Engagement-weighted sentiment evolution
- Phase-to-phase sentiment changes

## Performance Considerations

- Batch processing: 8 items per batch
- Max records processed: 1,000 per run
- Remaining records: Assigned neutral sentiment
- Caching: 5-minute TTL on Streamlit data
- Timeout: 1 hour for phase processing

---