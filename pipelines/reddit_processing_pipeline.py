import pandas as pd
import boto3
import os
import sys
from io import StringIO
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
from datetime import datetime, timedelta
import warnings
import numpy as np
from textblob import TextBlob
import re

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY, AWS_REGION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_s3_client():
    """Create S3 client with credentials"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_ACCESS_KEY,
            region_name=AWS_REGION
        )
        return s3_client
    except Exception as e:
        logger.error(f"Error creating S3 client: {str(e)}")
        raise

def list_files_in_date_range(bucket_name, start_date, end_date):
    """
    List all CSV files in S3 raw/ folder within date range
    
    Args:
        bucket_name: S3 bucket name
        start_date: Start date in YYYYMMDD format (e.g., '20251101')
        end_date: End date in YYYYMMDD format (e.g., '20251115')
    
    Returns:
        List of file keys in S3
    """
    try:
        s3_client = get_s3_client()
        
        # List all files in raw/ folder
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='raw/reddit_'
        )
        
        if 'Contents' not in response:
            logger.warning("No files found in raw/ folder")
            return []
        
        # Filter files by date range
        files_in_range = []
        start_date_int = int(start_date)
        end_date_int = int(end_date)
        
        for obj in response['Contents']:
            file_key = obj['Key']
            # Extract date from filename: raw/reddit_20251208.csv -> 20251208
            filename = file_key.split('/')[-1]  # Get filename
            if filename.startswith('reddit_') and filename.endswith('.csv'):
                try:
                    date_str = filename.replace('reddit_', '').replace('.csv', '')
                    file_date = int(date_str)
                    
                    if start_date_int <= file_date <= end_date_int:
                        files_in_range.append(file_key)
                        logger.info(f"Found file in range: {file_key}")
                except ValueError:
                    # Skip files that don't match date pattern
                    continue
        
        logger.info(f"Found {len(files_in_range)} files between {start_date} and {end_date}")
        return sorted(files_in_range)
    
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise

def download_from_s3(bucket_name, file_key):
    """Download CSV file from S3 and return as DataFrame"""
    try:
        s3_client = get_s3_client()
        logger.info(f"Downloading {file_key} from bucket {bucket_name}")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        logger.info(f"Successfully loaded {len(df)} records from {file_key}")
        return df
    except Exception as e:
        logger.error(f"Error downloading {file_key}: {str(e)}")
        return None

def download_and_combine_files(bucket_name, file_keys):
    """
    Download multiple CSV files from S3 and combine them
    
    Args:
        bucket_name: S3 bucket name
        file_keys: List of file keys to download
    
    Returns:
        Combined DataFrame
    """
    all_dataframes = []
    
    for file_key in file_keys:
        df = download_from_s3(bucket_name, file_key)
        if df is not None and len(df) > 0:
            all_dataframes.append(df)
    
    if not all_dataframes:
        logger.error("No data loaded from any files")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Combined {len(all_dataframes)} files into {len(combined_df)} total records")
    
    return combined_df

def clean_reddit_data(df):
    """Clean and preprocess Reddit data"""
    try:
        logger.info("Starting data cleaning...")
        initial_count = len(df)
        df = df.drop_duplicates(subset=['id'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        df = df.fillna('')
        if 'edited' in df.columns:
            df['edited'] = df['edited'].apply(lambda x: False if x is False or x == 'False' or x == 0 else True)
        df['combined_text'] = df['title'].astype(str)
        df = df[df['combined_text'].str.strip() != '']
        df['combined_text'] = df['combined_text'].str[:500]
        
        logger.info(f"Cleaned data: {len(df)} records remaining")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def extract_keywords(text):
    """Extract important keywords related to protests and government"""
    protest_keywords = [
        'protest', 'demonstration', 'rally', 'movement', 'uprising',
        'corruption', 'government', 'youth', 'genz', 'gen z',
        'politics', 'minister', 'prime minister', 'parliament',
        'justice', 'reform', 'change', 'revolution', 'activism',
        'nepobabies', 'nepotism', 'accountability', 'transparency',
        'social media', 'ban', 'restriction', 'censorship'
    ]
    
    text_lower = text.lower()
    found_keywords = [kw for kw in protest_keywords if kw in text_lower]
    return found_keywords

def analyze_emotion(text):
    """Analyze emotional intensity and specific emotions"""
    emotions = {
        'anger': 0.0,
        'fear': 0.0,
        'joy': 0.0,
        'sadness': 0.0,
        'frustration': 0.0,
        'hope': 0.0
    }
    
    text_lower = text.lower()
    
    anger_words = ['angry', 'furious', 'outraged', 'disgusted', 'hate', 'corrupt', 
                   'shameful', 'betrayed', 'unfair', 'injustice', 'fraud', 'scam']
    emotions['anger'] = sum(1 for word in anger_words if word in text_lower) / 5.0
    
    fear_words = ['afraid', 'scared', 'worried', 'anxious', 'fear', 'threat', 
                  'danger', 'risk', 'unsafe', 'violence']
    emotions['fear'] = sum(1 for word in fear_words if word in text_lower) / 5.0
    
    joy_words = ['happy', 'joy', 'celebrate', 'proud', 'victory', 'success', 
                 'win', 'achievement', 'progress', 'hopeful']
    emotions['joy'] = sum(1 for word in joy_words if word in text_lower) / 5.0
    
    sadness_words = ['sad', 'disappointed', 'depressed', 'hopeless', 'tragic', 
                     'unfortunate', 'failed', 'loss', 'defeat']
    emotions['sadness'] = sum(1 for word in sadness_words if word in text_lower) / 5.0
    
    frustration_words = ['frustrated', 'tired', 'enough', 'fed up', 'sick of',
                        'why', 'when will', 'how long', 'still waiting']
    emotions['frustration'] = sum(1 for word in frustration_words if word in text_lower) / 5.0
    
    hope_words = ['hope', 'believe', 'change', 'together', 'future', 'better',
                 'movement', 'unity', 'strength', 'possible']
    emotions['hope'] = sum(1 for word in hope_words if word in text_lower) / 5.0
    
    emotions = {k: min(v, 1.0) for k, v in emotions.items()}
    
    return emotions

def detect_narrative_frame(text, keywords):
    """Detect which narrative frame the post belongs to"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['corrupt', 'nepotism', 'nepobabies', 'accountability']):
        return 'anti_corruption'
    elif any(word in text_lower for word in ['youth', 'gen z', 'genz', 'young people', 'students']):
        return 'youth_empowerment'
    elif any(word in text_lower for word in ['government', 'minister', 'parliament', 'politician']):
        return 'government_criticism'
    elif any(word in text_lower for word in ['ban', 'censorship', 'social media', 'restriction', 'tiktok']):
        return 'media_freedom'
    elif any(word in text_lower for word in ['reform', 'change', 'justice', 'revolution']):
        return 'reform_advocacy'
    else:
        return 'general_discussion'

def calculate_engagement_weight(row):
    """Calculate engagement weight based on score and comments"""
    score_weight = np.log1p(max(row['score'], 0))
    comment_weight = np.log1p(row['num_comments'])
    total_weight = score_weight + comment_weight
    return total_weight

def perform_enhanced_sentiment_analysis(df, text_column='combined_text'):
    """Enhanced sentiment analysis with multiple dimensions"""
    try:
        logger.info("Initializing enhanced sentiment analysis...")
        
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True,
            max_length=512
        )
        
        logger.info(f"Running enhanced sentiment analysis on {len(df)} records...")
        
        max_records = 1000  # Increased for batch processing
        if len(df) > max_records:
            logger.warning(f"Processing first {max_records} records")
            df_to_process = df.head(max_records).copy()
        else:
            df_to_process = df.copy()
        
        sentiments = []
        sentiment_scores = []
        emotions_list = []
        keywords_list = []
        narratives = []
        subjectivity_scores = []
        
        batch_size = 8
        total_batches = (len(df_to_process) + batch_size - 1) // batch_size
        
        for i in range(0, len(df_to_process), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = df_to_process[text_column].iloc[i:i+batch_size].tolist()
            batch_texts = [str(text)[:400] for text in batch_texts]
            
            try:
                results = sentiment_analyzer(batch_texts)
                
                for idx, (text, result) in enumerate(zip(batch_texts, results)):
                    label = result['label'].upper()
                    score = result['score']
                    
                    sentiments.append(label)
                    
                    if label == 'NEGATIVE':
                        sentiment_scores.append(-score)
                    else:
                        sentiment_scores.append(score)
                    
                    keywords = extract_keywords(text)
                    keywords_list.append(','.join(keywords) if keywords else '')
                    
                    emotions = analyze_emotion(text)
                    emotions_list.append(emotions)
                    
                    narrative = detect_narrative_frame(text, keywords)
                    narratives.append(narrative)
                    
                    try:
                        blob = TextBlob(text)
                        subjectivity_scores.append(blob.sentiment.subjectivity)
                    except:
                        subjectivity_scores.append(0.5)
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_num}: {str(e)}")
                for _ in range(len(batch_texts)):
                    sentiments.append('NEUTRAL')
                    sentiment_scores.append(0.0)
                    keywords_list.append('')
                    emotions_list.append({k: 0.0 for k in ['anger', 'fear', 'joy', 'sadness', 'frustration', 'hope']})
                    narratives.append('general_discussion')
                    subjectivity_scores.append(0.5)
            
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"Processed batch {batch_num}/{total_batches}")
        
        df_to_process['sentiment'] = sentiments
        df_to_process['sentiment_score'] = sentiment_scores
        df_to_process['keywords'] = keywords_list
        df_to_process['narrative_frame'] = narratives
        df_to_process['subjectivity'] = subjectivity_scores
        
        for emotion in ['anger', 'fear', 'joy', 'sadness', 'frustration', 'hope']:
            df_to_process[f'emotion_{emotion}'] = [e[emotion] for e in emotions_list]
        
        emotion_cols = [f'emotion_{e}' for e in ['anger', 'fear', 'joy', 'sadness', 'frustration', 'hope']]
        df_to_process['dominant_emotion'] = df_to_process[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')
        
        df_to_process['engagement_weight'] = df_to_process.apply(calculate_engagement_weight, axis=1)
        df_to_process['weighted_sentiment'] = df_to_process['sentiment_score'] * df_to_process['engagement_weight']
        
        if len(df) > max_records:
            remaining_df = df.iloc[max_records:].copy()
            remaining_df['sentiment'] = 'NEUTRAL'
            remaining_df['sentiment_score'] = 0.0
            remaining_df['keywords'] = ''
            remaining_df['narrative_frame'] = 'general_discussion'
            remaining_df['subjectivity'] = 0.5
            for emotion in ['anger', 'fear', 'joy', 'sadness', 'frustration', 'hope']:
                remaining_df[f'emotion_{emotion}'] = 0.0
            remaining_df['dominant_emotion'] = 'neutral'
            remaining_df['engagement_weight'] = 0.0
            remaining_df['weighted_sentiment'] = 0.0
            df_to_process = pd.concat([df_to_process, remaining_df], ignore_index=True)
        
        logger.info("Enhanced sentiment analysis completed!")
        logger.info(f"\nSentiment distribution:\n{df_to_process['sentiment'].value_counts()}")
        logger.info(f"\nNarrative distribution:\n{df_to_process['narrative_frame'].value_counts()}")
        logger.info(f"\nDominant emotion distribution:\n{df_to_process['dominant_emotion'].value_counts()}")
        
        return df_to_process
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        df['sentiment'] = 'NEUTRAL'
        df['sentiment_score'] = 0.0
        return df

def upload_to_s3(df, bucket_name, file_key):
    """Upload processed DataFrame to S3"""
    try:
        s3_client = get_s3_client()
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        logger.info(f"Uploading processed data to S3: {file_key}")
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=csv_buffer.getvalue()
        )
        logger.info("Successfully uploaded processed data to S3")
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise

def save_to_postgres(df, table_name='reddit_sentiment', phase='realtime'):
    """Save processed data to Postgres database with phase information"""
    try:
        from sqlalchemy import create_engine
        
        db_host = 'postgres'
        db_port = '5432'
        db_name = 'airflow_reddit'
        db_user = 'postgres'
        db_password = 'postgres'
        
        connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        
        logger.info("Connecting to Postgres database...")
        engine = create_engine(connection_string)
        
        df['processed_at'] = datetime.now()
        df['phase'] = phase
        
        logger.info(f"Saving {len(df)} records to Postgres table '{table_name}' with phase '{phase}'")
        df.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=100)
        
        logger.info("Successfully saved to Postgres!")
    except Exception as e:
        logger.error(f"Error saving to Postgres: {str(e)}")
        raise

def process_phase_date_range(phase_name, start_date, end_date, bucket_name, **kwargs):
    """
    Process all CSV files in a date range for a specific protest phase
    
    Args:
        phase_name: 'before', 'during', or 'after'
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        bucket_name: S3 bucket name
    """
    try:
        logger.info("=" * 60)
        logger.info(f"PROCESSING PHASE: {phase_name.upper()}")
        logger.info(f"Date Range: {start_date} to {end_date}")
        logger.info("=" * 60)
        
        # Step 1: List all files in date range
        logger.info("STEP 1: Finding files in date range...")
        file_keys = list_files_in_date_range(bucket_name, start_date, end_date)
        
        if not file_keys:
            logger.warning(f"No files found for phase {phase_name} in date range {start_date}-{end_date}")
            return f"No files found for phase {phase_name}"
        
        # Step 2: Download and combine all files
        logger.info("STEP 2: Downloading and combining files...")
        combined_df = download_and_combine_files(bucket_name, file_keys)
        
        if combined_df is None or len(combined_df) == 0:
            logger.warning(f"No data loaded for phase {phase_name}")
            return f"No data for phase {phase_name}"
        
        # Step 3: Clean data
        logger.info("STEP 3: Cleaning data...")
        df_cleaned = clean_reddit_data(combined_df)
        
        # Step 4: Sentiment analysis
        logger.info("STEP 4: Performing sentiment analysis...")
        df_with_sentiment = perform_enhanced_sentiment_analysis(df_cleaned)
        
        # Step 5: Save to S3 (processed folder)
        logger.info("STEP 5: Saving to S3...")
        processed_file_key = f'processed/{phase_name}_phase_{start_date}_{end_date}_processed.csv'
        upload_to_s3(df_with_sentiment, bucket_name, processed_file_key)
        
        # Step 6: Save to Postgres
        logger.info("STEP 6: Saving to Postgres...")
        save_to_postgres(df_with_sentiment, phase=phase_name)
        
        logger.info("=" * 60)
        logger.info(f"PHASE {phase_name.upper()} COMPLETED SUCCESSFULLY!")
        logger.info(f"Processed {len(df_with_sentiment)} records from {len(file_keys)} files")
        logger.info("=" * 60)
        
        return f"Processed {len(df_with_sentiment)} records for phase {phase_name}"
        
    except Exception as e:
        logger.error(f"Failed to process phase {phase_name}: {str(e)}")
        raise

def reddit_processing_pipeline(file_name, bucket_name, phase='realtime', **kwargs):
    """
    Single-file processing pipeline (for daily real-time data)
    
    Args:
        file_name: Name of the file in S3 (without .csv extension)
        bucket_name: S3 bucket name
        phase: 'realtime' for daily processing
    """
    try:
        raw_file_key = f'raw/{file_name}.csv'
        processed_file_key = f'processed/{file_name}_processed.csv'
        
        logger.info("=" * 50)
        logger.info(f"PROCESSING SINGLE FILE: {file_name}")
        logger.info(f"PHASE: {phase.upper()}")
        logger.info("=" * 50)
        
        logger.info("STEP 1: Downloading raw data from S3")
        df = download_from_s3(bucket_name, raw_file_key)
        
        if df is None or len(df) == 0:
            logger.warning(f"No data in file {file_name}")
            return f"No data in {file_name}"
        
        logger.info("STEP 2: Cleaning data")
        df_cleaned = clean_reddit_data(df)
        
        logger.info("STEP 3: Performing sentiment analysis")
        df_with_sentiment = perform_enhanced_sentiment_analysis(df_cleaned)
        
        logger.info("STEP 4: Saving to S3")
        upload_to_s3(df_with_sentiment, bucket_name, processed_file_key)
        
        logger.info("STEP 5: Saving to Postgres")
        save_to_postgres(df_with_sentiment, phase=phase)
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        return f"Processed {len(df_with_sentiment)} records"
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise