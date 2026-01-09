import sys
import os
import pandas as pd
import numpy as np
import pytest

# Add root directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.reddit_pipeline import transform_data

def test_transform_data_structure():
    """Test that transformation creates correct columns and types"""
    # 1. Create dummy raw data (mocking what comes from PRAW)
    raw_data = [{
        'id': '123',
        'title': 'Test Post',
        'score': 100.0, # Float, needs to become int
        'num_comments': 50.0,
        'author': 'test_user',
        'created_utc': 1700000000, # Unix timestamp
        'url': 'http://google.com',
        'over_18': True,
        'edited': False,
        'spoiler': False,
        'stickied': False
    }]
    
    df = pd.DataFrame(raw_data)
    
    # 2. Run your transformation function
    processed_df = transform_data(df)
    
    # 3. Assertions (The Test)
    assert 'created_utc' in processed_df.columns
    assert processed_df['score'].dtype == 'int64' or processed_df['score'].dtype == 'int32'
    assert isinstance(processed_df['created_utc'].iloc[0], pd.Timestamp)
    assert processed_df['title'].iloc[0] == 'Test Post'