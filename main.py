import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
# import file of functions from the same dir
import feature_funcs
import matplotlib.pyplot as plt

# read, sort chronologically, and dropna to ensure no extra rows being processed
games = pd.read_csv('nba_games/games.csv').sort_values('GAME_DATE_EST')
# print(f"Before cleaning: {len(games)} games")
games = games.dropna(subset=['AST_home', 'AST_away', 'FG_PCT_home', 'FG_PCT_away'])
# print(f"After cleaning: {len(games)} games")

#create a rolling average attr and split into train/test
games = feature_funcs.rolling_avgs(games)
print(f'created all rolling averages - {len(games)} rows')

# differential features
games['PT_DIFF_L5'] = games['PPG_L5_home'] - games['PPG_L5_away']
print('created point differential feature')
games['REB_DIFF_L5'] = games['RPG_L5_home'] - games['RPG_L5_away']
print('created rebound differential feature')
games['AST_DIFF_L5'] = games['APG_L5_home'] - games['APG_L5_away']
print('created assist differential feature')
games['FG_DIFF_L5'] = games['FG_L5_home'] - games['FG_L5_away']
print('created field goal PCT differential feature')

# clean and perform train/test split VERY IMPORTANT!!!
# print(f"Before dropna: {len(games)} games")
games = games.dropna()
# print(f"After dropna: {len(games)} games") 
train, test = feature_funcs.split(games, .8)

# Combine home and away features for training
x_train = train[['PPG_L5_home', 'PPG_L5_away',
                 'RPG_L5_home', 'RPG_L5_away',
                 'APG_L5_home', 'APG_L5_away',
                 'FG_L5_home', 'FG_L5_away',
                 'PT_DIFF_L5', 'REB_DIFF_L5', 
                 'AST_DIFF_L5', 'FG_DIFF_L5']]
y_train = train['HOME_TEAM_WINS']

# Combine home and away features for testing
x_test = test[['PPG_L5_home', 'PPG_L5_away',
                 'RPG_L5_home', 'RPG_L5_away',
                 'APG_L5_home', 'APG_L5_away',
                 'FG_L5_home', 'FG_L5_away',
                 'PT_DIFF_L5', 'REB_DIFF_L5', 
                 'AST_DIFF_L5', 'FG_DIFF_L5']]
y_test = test['HOME_TEAM_WINS']

# create model and feed train dataset/labels
model = XGBClassifier()
model.fit(x_train, y_train)

# create inferred labels and check accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print('Model accuracy: ' + str(100 * accuracy) + '%')
# print(model.feature_importances_)
feature_importance_dict = dict(zip(x_train.columns, model.feature_importances_))
print("\nFeature Importances:")
print("-" * 40)
for feature, importance in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature:20s}: {importance:.4f}")

# debug
# print()
# print('Debug:')
# print("NaN counts:")
# print(games[['PT_DIFF_L5', 'REB_DIFF_L5', 'AST_DIFF_L5']].isna().sum())
# print("\nSample of differentials:")
# print(games[['PPG_L5_home', 'PPG_L5_away', 'PT_DIFF_L5']].head(10))

# visualize
# Get importances and sort
importances = model.feature_importances_
features = x_train.columns
sorted_idx = importances.argsort()

# Create horizontal bar plot
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance for NBA Win Prediction (69.6% Accuracy)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()