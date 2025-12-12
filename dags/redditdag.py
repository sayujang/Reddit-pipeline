from airflow import DAG
from datetime import datetime, timedelta
import os
import sys
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.aws_s3_pipeline import upload_s3_pipeline
from pipelines.reddit_pipeline import reddit_pipeline
from pipelines.reddit_processing_pipeline import reddit_processing_pipeline, process_phase_date_range
from utils.constants import AWS_BUCKET_NAME

default_args = {
    'owner': 'Sayuj Chapagain',
    'start_date': datetime(2025, 12, 7),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

# ==========================================
# DAG 1: Real-time ETL Pipeline (Daily)
# ==========================================
file_postfix = datetime.now().strftime("%Y%m%d")

dag_realtime = DAG(
    dag_id="etl_reddit_pipeline_realtime",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=['reddit', 'etl', 'realtime']
)

# Real-time extraction
extract_realtime = PythonOperator(
    task_id='reddit_extraction',
    python_callable=reddit_pipeline,
    op_kwargs={
        'file_name': f'reddit_{file_postfix}',
        'subreddit': 'Nepal',
        'time_filter': 'day',
        'limit': 100
    },
    dag=dag_realtime
)

# Upload to S3
upload_s3_realtime = PythonOperator(
    task_id="s3_upload",
    python_callable=upload_s3_pipeline,
    dag=dag_realtime
)

# Process real-time data
process_realtime = PythonOperator(
    task_id='process_reddit_data',
    python_callable=reddit_processing_pipeline,
    op_kwargs={
        'file_name': f'reddit_{file_postfix}',
        'bucket_name': AWS_BUCKET_NAME,
        'phase': 'realtime'
    },
    execution_timeout=timedelta(minutes=30),
    dag=dag_realtime
)

extract_realtime >> upload_s3_realtime >> process_realtime


# ==========================================
# DAG 2: Historical Protest Analysis (Manual/Once)
# Process date ranges for each protest phase
# ==========================================

dag_protest = DAG(
    dag_id="process_protest_phase_data",
    default_args=default_args,
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['reddit', 'protest', 'historical']
)

# IMPORTANT: DEFINE YOUR DATE RANGES HERE
# Adjust these dates based on when the protest actually occurred
PHASE_DATE_RANGES = {
    'before': {
        'start_date': '20250901',  # YYYYMMDD format
        'end_date': '20250907',    # Day before protest started
        'description': 'Before Gen Z Protest (Nov 1-15, 2025)'
    },
    'during': {
        'start_date': '20250908',  # Protest start date
        'end_date': '20250909',    # Protest end date
        'description': 'During Gen Z Protest (Nov 16-30, 2025)'
    },
    'after': {
        'start_date': '20250910',  # Day after protest ended
        'end_date': '20251031',    # Current date
        'description': 'After Gen Z Protest (Dec 1-10, 2025)'
    }
}

# Process BEFORE phase
process_before = PythonOperator(
    task_id='process_before_protest',
    python_callable=process_phase_date_range,
    op_kwargs={
        'phase_name': 'before',
        'start_date': PHASE_DATE_RANGES['before']['start_date'],
        'end_date': PHASE_DATE_RANGES['before']['end_date'],
        'bucket_name': AWS_BUCKET_NAME
    },
    execution_timeout=timedelta(hours=1),
    dag=dag_protest
)

# Process DURING phase
process_during = PythonOperator(
    task_id='process_during_protest',
    python_callable=process_phase_date_range,
    op_kwargs={
        'phase_name': 'during',
        'start_date': PHASE_DATE_RANGES['during']['start_date'],
        'end_date': PHASE_DATE_RANGES['during']['end_date'],
        'bucket_name': AWS_BUCKET_NAME
    },
    execution_timeout=timedelta(hours=1),
    dag=dag_protest
)

# Process AFTER phase
process_after = PythonOperator(
    task_id='process_after_protest',
    python_callable=process_phase_date_range,
    op_kwargs={
        'phase_name': 'after',
        'start_date': PHASE_DATE_RANGES['after']['start_date'],
        'end_date': PHASE_DATE_RANGES['after']['end_date'],
        'bucket_name': AWS_BUCKET_NAME
    },
    execution_timeout=timedelta(hours=1),
    dag=dag_protest
)

# Process all phases in parallel (independent tasks)
[process_before, process_during, process_after]