import configparser
import os
import sys

# 1. Initialize the local config parser (for Airflow/Local use)
parser = configparser.ConfigParser()
parser.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/config.conf'))

# 2. helper function to safely get keys from Streamlit OR Local Config
def get_key(section, key_name, local_config_key=None):
    """
    Tries to get the key from:
    1. Streamlit Secrets (Production Dashboard)
    2. Environment Variables (Docker/CI)
    3. Local Config File (Local Development)
    """
    # Option A: Try Streamlit Secrets
    try:
        import streamlit as st
        # Check if secrets exist and specific section/key exists
        if hasattr(st, 'secrets') and section in st.secrets:
             # Use the key name exactly as you defined in TOML
             return st.secrets[section][key_name]
    except (ImportError, FileNotFoundError, KeyError, AttributeError):
        pass # Streamlit not installed or secrets not found, move to next option

    # Option B: Try Environment Variable
    # Example: looks for AWS_ACCESS_KEY_ID
    env_var = os.getenv(key_name.upper())
    if env_var:
        return env_var

    # Option C: Try Local Config file
    # If local_config_key is different (e.g., lowercase), use it, otherwise use key_name
    config_key = local_config_key if local_config_key else key_name
    try:
        return parser.get(section, config_key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return None

# ==========================================
# Load Variables using the smart helper
# ==========================================

# AWS Credentials
# matches [aws] in your TOML
AWS_ACCESS_KEY_ID = get_key('aws', 'AWS_ACCESS_KEY_ID', 'aws_access_key_id')
AWS_ACCESS_KEY = get_key('aws', 'AWS_ACCESS_KEY', 'aws_secret_access_key')
AWS_REGION = get_key('aws', 'AWS_REGION', 'aws_region')
AWS_BUCKET_NAME = get_key('aws', 'AWS_BUCKET_NAME', 'aws_bucket_name')

# Reddit Credentials
# matches [reddit] in your TOML
CLIENT_ID = get_key('reddit', 'CLIENT_ID', 'client_id')
SECRET = get_key('reddit', 'SECRET', 'secret')

# Output Paths (Keep these as local defaults)
OUTPUT_PATH = '/opt/airflow/data/output' if os.path.exists('/opt/airflow') else './data/output'

# List of fields to extract (Keep as is)
POST_FIELDS = (
    'id',
    'title',
    'score',
    'num_comments',
    'author',
    'created_utc',
    'url',
    'over_18',
    'edited',
    'spoiler',
    'stickied'
)