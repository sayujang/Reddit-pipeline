from airflow import DAG
from datetime import datetime, timedelta
import os
import sys
from airflow.operators.python import PythonOperator

# sys.path is a list of folder paths that Python searches through when you try to import something
# add the root path of the directory in the container as the first index to search
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.aws_s3_pipeline import upload_s3_pipeline
from pipelines.reddit_pipeline import reddit_pipeline
from pipelines.reddit_processing_pipeline import reddit_processing_pipeline
from utils.constants import AWS_BUCKET_NAME

default_args = {
    'owner': 'Sayuj Chapagain',
    'start_date': datetime(2025, 12, 7),
    'retries': 2,  # Retry failed tasks
    'retry_delay': timedelta(minutes=5)
}

# REMOVED: file_postfix = datetime.now().strftime("%Y%m%d")
# This caused issues because it's evaluated at DAG parse time, not execution time

# dag definition
dag = DAG(
    dag_id="etl_reddit_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    tags=['reddit', 'etl', 'pipeline'],
    dagrun_timeout=timedelta(hours=2),  # Added: Prevent hung DAG runs
    max_active_runs=3  # Added: Reasonable concurrency limit
)

# extraction task by calling reddit_pipeline function
extract = PythonOperator(
    task_id='reddit_extraction',
    python_callable=reddit_pipeline,
    op_kwargs={
        'file_name': 'reddit_{{ ds_nodash }}',  # FIXED: Uses Airflow template (e.g., reddit_20251209)
        'subreddit': 'Nepal',
        'time_filter': 'day',  # fetches from last day
        'limit': 1000  # INCREASED: From 100 to 1000 to capture more posts
    },
    execution_timeout=timedelta(minutes=15),  # Added: Timeout for extraction task
    dag=dag
)

# upload task to s3 bucket by calling upload_s3_pipeline
upload_s3 = PythonOperator(
    task_id="s3_upload",
    python_callable=upload_s3_pipeline,
    execution_timeout=timedelta(minutes=10),  # Added: Timeout for upload task
    dag=dag
)

# clean + sentiment analysis + save to postgres
process_data = PythonOperator(
    task_id='process_reddit_data',
    python_callable=reddit_processing_pipeline,
    op_kwargs={
        'file_name': 'reddit_{{ ds_nodash }}',  # FIXED: Uses Airflow template to match extraction file
        'bucket_name': AWS_BUCKET_NAME
    },
    execution_timeout=timedelta(minutes=30),
    dag=dag
)

# sets rule that extract task must occur before upload
extract >> upload_s3 >> process_data