#use python3.9 as base image
FROM apache/airflow:2.7.1-python3.11 

#copies the requirement text into the opt/airflow directory inside image
COPY requirements_airflow.txt /opt/airflow/

#switch to root user to install system packages like gcc and python3-dev
USER root
RUN apt-get update && apt-get install -y gcc python3-dev
#Switch to airflow user
USER airflow

RUN pip install --no-cache-dir -r /opt/airflow/requirements_airflow.txt
#the resulting image is custom-airflow:2.7.1-python3.9

