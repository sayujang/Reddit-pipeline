from etls.aws_etl import connect_to_s3, create_bucket_if_not_exist, upload_to_s3
from utils.constants import AWS_BUCKET_NAME

#functions takes in task instance ti
def upload_s3_pipeline(ti):
    #get the file path of the saved csv file during extraction '/opt/airflow/data/output/reddit_20251207.csv'.
    file_path = ti.xcom_pull(task_ids='reddit_extraction', key='return_value')

    #create s3 session
    s3 = connect_to_s3()
    create_bucket_if_not_exist(s3, AWS_BUCKET_NAME)
    upload_to_s3(s3, file_path, AWS_BUCKET_NAME, file_path.split('/')[-1])