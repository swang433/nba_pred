import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
#import file of functions from the same dir
import feature_funcs

#read and sort
games = pd.read_csv('nba_games/games.csv').sort_values('GAME_DATE_EST')

#create a rolling average attr and split into train/test
games = feature_funcs.rolling_avg(games, 'PTS', 5, 'PPG_L5')
print('created PPG rolling average')
games = feature_funcs.rolling_avg(games, 'REB', 5, 'RPG_L5')
print('created RPG rolling average')
games = feature_funcs.rolling_avg(games, 'AST', 5, 'APG_L5')
print('created APG rolling average')
games = feature_funcs.rolling_avg(games, 'FG_PCT', 5, 'FG_L5')
print('created FG rolling average')
train, test = feature_funcs.split(games, .8)

# Combine home and away features for training
x_train = train[['PPG_L5_home', 'PPG_L5_away',
                 'RPG_L5_home', 'RPG_L5_away',
                 'APG_L5_home', 'APG_L5_away',
                 'FG_L5_home', 'FG_L5_away',]]
y_train = train['HOME_TEAM_WINS']

# Combine home and away features for testing
x_test = test[['PPG_L5_home', 'PPG_L5_away',
                 'RPG_L5_home', 'RPG_L5_away',
                 'APG_L5_home', 'APG_L5_away',
                 'FG_L5_home', 'FG_L5_away',]]
y_test = test['HOME_TEAM_WINS']

#create model and feed train dataset/labels
feats = ['PPG_L5_home', 'PPG_L5_away',
                 'FG3_PCT_home', 'FG3_PCT_away',
                 'AST_home', 'AST_away']
model = XGBClassifier()
model.fit(x_train, y_train) 

#create inferred labels and check accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print('Model accuracy: ' + str(100 * accuracy) + '%')