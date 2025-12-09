import pandas as pd
import boto3
import os
import sys
from io import StringIO
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging
from datetime import datetime
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

def download_from_s3(bucket_name, file_key):
    """Download CSV file from S3 and return as DataFrame"""
    try:
        s3_client = get_s3_client()
        logger.info(f"Downloading {file_key} from bucket {bucket_name}")
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        logger.info(f"Successfully loaded {len(df)} records from S3")
        return df
    except Exception as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        raise

def clean_reddit_data(df):
    """Clean and preprocess Reddit data"""
    try:
        logger.info("Starting data cleaning...")
        initial_count = len(df)
        df = df.drop_duplicates(subset=['id'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        df = df.fillna('')
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
    # Keywords related to Nepal's Gen Z protests
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
    """
    Analyze emotional intensity and specific emotions
    Returns: emotion scores for anger, fear, joy, sadness
    """
    # Simple rule-based emotion detection
    emotions = {
        'anger': 0.0,
        'fear': 0.0,
        'joy': 0.0,
        'sadness': 0.0,
        'frustration': 0.0,
        'hope': 0.0
    }
    
    text_lower = text.lower()
    
    # Anger indicators
    anger_words = ['angry', 'furious', 'outraged', 'disgusted', 'hate', 'corrupt', 
                   'shameful', 'betrayed', 'unfair', 'injustice', 'fraud', 'scam']
    emotions['anger'] = sum(1 for word in anger_words if word in text_lower) / 5.0
    
    # Fear indicators
    fear_words = ['afraid', 'scared', 'worried', 'anxious', 'fear', 'threat', 
                  'danger', 'risk', 'unsafe', 'violence']
    emotions['fear'] = sum(1 for word in fear_words if word in text_lower) / 5.0
    
    # Joy indicators
    joy_words = ['happy', 'joy', 'celebrate', 'proud', 'victory', 'success', 
                 'win', 'achievement', 'progress', 'hopeful']
    emotions['joy'] = sum(1 for word in joy_words if word in text_lower) / 5.0
    
    # Sadness indicators
    sadness_words = ['sad', 'disappointed', 'depressed', 'hopeless', 'tragic', 
                     'unfortunate', 'failed', 'loss', 'defeat']
    emotions['sadness'] = sum(1 for word in sadness_words if word in text_lower) / 5.0
    
    # Frustration indicators
    frustration_words = ['frustrated', 'tired', 'enough', 'fed up', 'sick of',
                        'why', 'when will', 'how long', 'still waiting']
    emotions['frustration'] = sum(1 for word in frustration_words if word in text_lower) / 5.0
    
    # Hope indicators
    hope_words = ['hope', 'believe', 'change', 'together', 'future', 'better',
                 'movement', 'unity', 'strength', 'possible']
    emotions['hope'] = sum(1 for word in hope_words if word in text_lower) / 5.0
    
    # Cap at 1.0
    emotions = {k: min(v, 1.0) for k, v in emotions.items()}
    
    return emotions

def detect_narrative_frame(text, keywords):
    """
    Detect which narrative frame the post belongs to
    """
    text_lower = text.lower()
    
    # Anti-corruption narrative
    if any(word in text_lower for word in ['corrupt', 'nepotism', 'nepobabies', 'accountability']):
        return 'anti_corruption'
    
    # Youth empowerment narrative
    elif any(word in text_lower for word in ['youth', 'gen z', 'genz', 'young people', 'students']):
        return 'youth_empowerment'
    
    # Government criticism narrative
    elif any(word in text_lower for word in ['government', 'minister', 'parliament', 'politician']):
        return 'government_criticism'
    
    # Social media/censorship narrative
    elif any(word in text_lower for word in ['ban', 'censorship', 'social media', 'restriction', 'tiktok']):
        return 'media_freedom'
    
    # Reform/change narrative
    elif any(word in text_lower for word in ['reform', 'change', 'justice', 'revolution']):
        return 'reform_advocacy'
    
    else:
        return 'general_discussion'

def calculate_engagement_weight(row):
    """
    Calculate engagement weight based on score and comments
    Higher engagement = more influential post
    """
    score_weight = np.log1p(max(row['score'], 0))  # log scale for score
    comment_weight = np.log1p(row['num_comments'])  # log scale for comments
    
    # Combined weight (normalized)
    total_weight = score_weight + comment_weight
    return total_weight

def perform_enhanced_sentiment_analysis(df, text_column='combined_text'):
    """
    Enhanced sentiment analysis with multiple dimensions:
    1. Basic sentiment (positive/negative/neutral)
    2. Emotion analysis (anger, fear, joy, sadness, frustration, hope)
    3. Narrative framing
    4. Keyword extraction
    5. Engagement weighting
    """
    try:
        logger.info("Initializing enhanced sentiment analysis...")
        
        # Initialize base sentiment analyzer
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True,
            max_length=512
        )
        
        logger.info(f"Running enhanced sentiment analysis on {len(df)} records...")
        
        # Limit processing if needed
        max_records = 500
        if len(df) > max_records:
            logger.warning(f"Processing first {max_records} records")
            df_to_process = df.head(max_records).copy()
        else:
            df_to_process = df.copy()
        
        # Initialize result columns
        sentiments = []
        sentiment_scores = []
        emotions_list = []
        keywords_list = []
        narratives = []
        subjectivity_scores = []
        
        batch_size = 4
        total_batches = (len(df_to_process) + batch_size - 1) // batch_size
        
        for i in range(0, len(df_to_process), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = df_to_process[text_column].iloc[i:i+batch_size].tolist()
            batch_texts = [str(text)[:400] for text in batch_texts]
            
            try:
                # Basic sentiment
                results = sentiment_analyzer(batch_texts)
                
                for idx, (text, result) in enumerate(zip(batch_texts, results)):
                    label = result['label'].upper()
                    score = result['score']
                    
                    sentiments.append(label)
                    
                    # Normalize score to -1 to 1
                    if label == 'NEGATIVE':
                        sentiment_scores.append(-score)
                    else:
                        sentiment_scores.append(score)
                    
                    # Extract keywords
                    keywords = extract_keywords(text)
                    keywords_list.append(','.join(keywords) if keywords else '')
                    
                    # Analyze emotions
                    emotions = analyze_emotion(text)
                    emotions_list.append(emotions)
                    
                    # Detect narrative
                    narrative = detect_narrative_frame(text, keywords)
                    narratives.append(narrative)
                    
                    # Calculate subjectivity using TextBlob
                    try:
                        blob = TextBlob(text)
                        subjectivity_scores.append(blob.sentiment.subjectivity)
                    except:
                        subjectivity_scores.append(0.5)
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_num}: {str(e)}")
                # Add neutral values for failed batch
                for _ in range(len(batch_texts)):
                    sentiments.append('NEUTRAL')
                    sentiment_scores.append(0.0)
                    keywords_list.append('')
                    emotions_list.append({k: 0.0 for k in ['anger', 'fear', 'joy', 'sadness', 'frustration', 'hope']})
                    narratives.append('general_discussion')
                    subjectivity_scores.append(0.5)
            
            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"Processed batch {batch_num}/{total_batches}")
        
        # Assign results
        df_to_process['sentiment'] = sentiments
        df_to_process['sentiment_score'] = sentiment_scores
        df_to_process['keywords'] = keywords_list
        df_to_process['narrative_frame'] = narratives
        df_to_process['subjectivity'] = subjectivity_scores
        
        # Unpack emotions into separate columns
        for emotion in ['anger', 'fear', 'joy', 'sadness', 'frustration', 'hope']:
            df_to_process[f'emotion_{emotion}'] = [e[emotion] for e in emotions_list]
        
        # Calculate dominant emotion
        emotion_cols = [f'emotion_{e}' for e in ['anger', 'fear', 'joy', 'sadness', 'frustration', 'hope']]
        df_to_process['dominant_emotion'] = df_to_process[emotion_cols].idxmax(axis=1).str.replace('emotion_', '')
        
        # Calculate engagement weight
        df_to_process['engagement_weight'] = df_to_process.apply(calculate_engagement_weight, axis=1)
        
        # Calculate weighted sentiment (more weight to highly engaged posts)
        df_to_process['weighted_sentiment'] = df_to_process['sentiment_score'] * df_to_process['engagement_weight']
        
        # Handle unprocessed records
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
        # Fallback
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

def save_to_postgres(df, table_name='reddit_sentiment'):
    """Save processed data to Postgres database"""
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
        
        logger.info(f"Saving {len(df)} records to Postgres table '{table_name}'")
        df.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=100)
        
        logger.info("Successfully saved to Postgres!")
    except Exception as e:
        logger.error(f"Error saving to Postgres: {str(e)}")
        raise

def reddit_processing_pipeline(file_name, bucket_name, **kwargs):
    """
    Enhanced processing pipeline with multi-dimensional sentiment analysis
    """
    try:
        raw_file_key = f'raw/{file_name}.csv'
        processed_file_key = f'processed/{file_name}_processed.csv'
        
        logger.info("=" * 50)
        logger.info("STEP 1: Downloading raw data from S3")
        logger.info("=" * 50)
        df = download_from_s3(bucket_name, raw_file_key)
        
        logger.info("=" * 50)
        logger.info("STEP 2: Cleaning data")
        logger.info("=" * 50)
        df_cleaned = clean_reddit_data(df)
        
        logger.info("=" * 50)
        logger.info("STEP 3: Performing enhanced sentiment analysis")
        logger.info("=" * 50)
        df_with_sentiment = perform_enhanced_sentiment_analysis(df_cleaned)
        
        logger.info("=" * 50)
        logger.info("STEP 4: Saving to S3")
        logger.info("=" * 50)
        upload_to_s3(df_with_sentiment, bucket_name, processed_file_key)
        
        logger.info("=" * 50)
        logger.info("STEP 5: Saving to Postgres")
        logger.info("=" * 50)
        save_to_postgres(df_with_sentiment)
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        return f"Processed {len(df_with_sentiment)} records"
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise