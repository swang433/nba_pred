import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

#import file from the same dir
import features 

# Get processed data (runs feature engineering only once, then caches)
games = features.get_games()

print("🎯 Ready for modeling with engineered features!")
print(f"Dataset shape: {games.shape}")

# Show some example engineered features
feature_examples = ['pts_L3', 'FG_L5_home', 'FG_L5_away', 'HOT_STREAK_home', 
                   'rest_days_home', 'day_of_week', 'diff_L5']
available_features = [f for f in feature_examples if f in games.columns]
print(f"Example features: {available_features}")