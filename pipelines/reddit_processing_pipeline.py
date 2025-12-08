import pandas as pd
import boto3
import os
import sys
from io import StringIO
from transformers import pipeline
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the root path to sys.path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY, AWS_REGION

# Set up logging
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
        
        # Read CSV into DataFrame
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
        
        # Remove duplicates based on post ID
        initial_count = len(df)
        df = df.drop_duplicates(subset=['id'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Fill NaN values
        df = df.fillna('')
        
        # Create a combined text column for sentiment analysis
        # Use only title since selftext is not in POST_FIELDS
        df['combined_text'] = df['title'].astype(str)
        
        # Remove empty texts
        df = df[df['combined_text'].str.strip() != '']
        
        # Truncate very long texts early to save processing time
        df['combined_text'] = df['combined_text'].str[:500]
        
        logger.info(f"Cleaned data: {len(df)} records remaining")
        return df
    
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def perform_sentiment_analysis(df, text_column='combined_text'):
    """
    Perform sentiment analysis on Reddit data using a lightweight model
    """
    try:
        logger.info("Initializing sentiment analysis model...")
        
        # Use a smaller, faster model for quicker processing
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,  # Force CPU to avoid GPU memory issues
            truncation=True,
            max_length=512
        )
        
        logger.info(f"Running sentiment analysis on {len(df)} records...")
        
        # Limit processing if dataset is very large
        max_records = 500  # Process max 500 records to prevent timeout
        if len(df) > max_records:
            logger.warning(f"Dataset has {len(df)} records. Processing only first {max_records} records.")
            df_to_process = df.head(max_records).copy()
        else:
            df_to_process = df.copy()
        
        # Process in smaller batches
        batch_size = 4  # Reduced from 32
        sentiments = []
        scores = []
        
        total_batches = (len(df_to_process) + batch_size - 1) // batch_size
        
        for i in range(0, len(df_to_process), batch_size):
            batch_num = i // batch_size + 1
            batch = df_to_process[text_column].iloc[i:i+batch_size].tolist()
            
            # Truncate long texts
            batch = [str(text)[:400] for text in batch]
            
            try:
                results = sentiment_analyzer(batch)
                
                for result in results:
                    label = result['label'].upper()
                    score = result['score']
                    
                    sentiments.append(label)
                    
                    # Normalize score to -1 to 1 range
                    if label == 'NEGATIVE':
                        scores.append(-score)
                    else:  # POSITIVE
                        scores.append(score)
                    
            except Exception as e:
                logger.warning(f"Error in batch {batch_num}/{total_batches}: {str(e)}")
                # Add neutral sentiment for failed batches
                sentiments.extend(['NEUTRAL'] * len(batch))
                scores.extend([0.0] * len(batch))
            
            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"Processed batch {batch_num}/{total_batches}")
        
        df_to_process['sentiment'] = sentiments
        df_to_process['sentiment_score'] = scores
        
        # If we limited the records, add back the unprocessed ones with neutral sentiment
        if len(df) > max_records:
            remaining_df = df.iloc[max_records:].copy()
            remaining_df['sentiment'] = 'NEUTRAL'
            remaining_df['sentiment_score'] = 0.0
            df_to_process = pd.concat([df_to_process, remaining_df], ignore_index=True)
        
        logger.info("Sentiment analysis completed!")
        logger.info(f"Sentiment distribution:\n{df_to_process['sentiment'].value_counts()}")
        
        return df_to_process
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        # Return dataframe with neutral sentiments as fallback
        df['sentiment'] = 'NEUTRAL'
        df['sentiment_score'] = 0.0
        return df

def upload_to_s3(df, bucket_name, file_key):
    """Upload processed DataFrame to S3"""
    try:
        s3_client = get_s3_client()
        
        # Convert DataFrame to CSV
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
        
        # Use the same Postgres connection as Airflow
        db_host = 'postgres'
        db_port = '5432'
        db_name = 'airflow_reddit'
        db_user = 'postgres'
        db_password = 'postgres'
        
        # Create connection string
        connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        
        logger.info("Connecting to Postgres database...")
        engine = create_engine(connection_string)
        
        # Add timestamp column
        df['processed_at'] = datetime.now()
        
        # Save to database (append mode to keep historical data)
        logger.info(f"Saving {len(df)} records to Postgres table '{table_name}'")
        df.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=100)
        
        logger.info("Successfully saved to Postgres!")
        
    except Exception as e:
        logger.error(f"Error saving to Postgres: {str(e)}")
        raise

def reddit_processing_pipeline(file_name, bucket_name, **kwargs):
    """
    Main processing pipeline that:
    1. Downloads raw data from S3
    2. Cleans the data
    3. Performs sentiment analysis
    4. Saves to S3 (processed) and Postgres
    """
    try:
        # Construct S3 file paths
        raw_file_key = f'raw/{file_name}.csv'
        processed_file_key = f'processed/{file_name}_processed.csv'
        
        # Step 1: Download raw data from S3
        logger.info("=" * 50)
        logger.info("STEP 1: Downloading raw data from S3")
        logger.info("=" * 50)
        df = download_from_s3(bucket_name, raw_file_key)
        
        # Step 2: Clean the data
        logger.info("=" * 50)
        logger.info("STEP 2: Cleaning data")
        logger.info("=" * 50)
        df_cleaned = clean_reddit_data(df)
        
        # Step 3: Perform sentiment analysis
        logger.info("=" * 50)
        logger.info("STEP 3: Performing sentiment analysis")
        logger.info("=" * 50)
        df_with_sentiment = perform_sentiment_analysis(df_cleaned)
        
        # Step 4: Save processed data to S3
        logger.info("=" * 50)
        logger.info("STEP 4: Saving to S3")
        logger.info("=" * 50)
        upload_to_s3(df_with_sentiment, bucket_name, processed_file_key)
        
        # Step 5: Save to Postgres
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