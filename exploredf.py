import pandas as pd
import numpy as np

games = pd.read_csv('nba_games/games.csv')
pd.set_option('display.max_columns', None)

########## pt1
#1 home team score > 120
over_120 = games[games['PTS_home'] > 120]

#2 Lakers home games ID: 1610612747
lakers_home = games[games['HOME_TEAM_ID'] == 1610612747]

#3 games during the year 2020 or later
later_2020 = games[games['SEASON'] >= 2020]

#4 close games (within 3)
close_games = games[games['PTS_home'] - games['PTS_away'] <= 3]

#5 games where the away team wins
away_wins = games[games['HOME_TEAM_WINS'] == 0]


########## pt2
# 6. Calculate average points scored per team (home games only)
avg_pts_home = games.groupby('HOME_TEAM_ID')['PTS_home'].mean()
# 7. Count how many home games each team played
games_per_team = games.groupby('HOME_TEAM_ID').size()

# 8. Find the team with highest average FG% at home
best_shooting_team = games.groupby('FG_PCT_home')

# 9. Calculate win percentage for each team at home
home_win_pct = games.groupby('HOME_TEAM_ID')['HOME_TEAM_WINS'].mean()

# 10. Get total assists per team across all their games (home + away)
# Hint: You'll need to combine HOME and AWAY stats
home_ast = games.groupby('HOME_TEAM_ID')['AST_home'].sum()
away_ast = games.groupby('VISITOR_TEAM_ID')['AST_away'].sum()
total_ast = home_ast.add(away_ast, fill_value=0)

########## pt3
# 11. Calculate each team's average points in their last 3 home games
# Hint: groupby + rolling + shift

games['pts_L3'] = games.groupby('HOME_TEAM_ID')['PTS_home'].transform( #rolling -> mean -> shift
    lambda x: x.rolling(window=3).mean().shift(1)
)

# 12. Calculate each team's average FG% over last 5 gamess
#reorganize columns
home_L5 = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'FG_PCT_home']].copy()
home_L5.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'FG_PCT']
away_L5 = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'FG_PCT_away']].copy()
away_L5.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'FG_PCT']

#concatenate
all_L5 = pd.concat([home_L5, away_L5], ignore_index=True)
all_L5.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
all_L5['FG_L5'] = all_L5.groupby('TEAM_ID')['FG_PCT'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
)

#merge back with the original df 
games = games.merge(
    all_L5[['GAME_ID', 'TEAM_ID', 'FG_L5']], 
    left_on=['GAME_ID', 'HOME_TEAM_ID'], 
    right_on=['GAME_ID', 'TEAM_ID'], 
    how='left'
).rename(columns={'FG_L5': 'FG_L5_home'}).drop('TEAM_ID', axis=1)

games = games.merge(
    all_L5[['GAME_ID', 'TEAM_ID', 'FG_L5']], 
    left_on=['GAME_ID', 'VISITOR_TEAM_ID'], 
    right_on=['GAME_ID', 'TEAM_ID'], 
    how='left'
).rename(columns={'FG_L5': 'FG_L5_away'}).drop('TEAM_ID', axis=1)
#note: left on and right on are what common cols being merged

# 13. Create a 'hot streak' indicator: 1 if team scored 110+ in last 2 games
#combine home + away points scored
L2_home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'PTS_home']].copy()
L2_home.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'PTS']
L2_away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'PTS_away']].copy()
L2_away.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'PTS']

#concatenate home and away points into a separate dataframe
L2_all = pd.concat([L2_home, L2_away], ignore_index=True)
L2_all = L2_all.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
L2_all['hot_streak'] = L2_all.groupby('TEAM_ID')['PTS'].transform(
    lambda x: ((x.shift(1) >= 110) & (x.shift(2) >= 110)).astype(int)
)
print(L2_all)
#CONTINUE!!!! FIX ME!!!!!! :,D
#merge back

# 14. Calculate rolling standard deviation of points (last 5 games)
# This measures consistency
games['pts_std_L5'] = ...

# 15. Calculate days between games for each team
# Hint: groupby + diff
games['rest_days'] = ...

# 16. Create 'day_of_week' feature (0=Monday, 6=Sunday)
games['day_of_week'] = ...

# 17. Create 'month' feature (1-12)
games['month'] = ...

# 18. Flag back-to-back games (rest_days == 1)
games['is_b2b'] = ...

# 19. Calculate cumulative wins for each team
# Hint: groupby + cumsum
games['cumulative_wins'] = ...

# 20. Calculate season-to-date win percentage for each team
games['season_win_pct'] = ...

# 21. Track current win/loss streak for each team
# Hint: This is tricky! Think about how to count consecutive wins
games['win_streak'] = ...

# 22. Calculate head-to-head win record between two teams
# For each game, what's the home team's historical record vs this opponent?
games['h2h_wins'] = ...

# 23. Create 'strength of schedule' - average opponent win%
# For each game, what's the away team's season win%?
games['opp_win_pct'] = ...

# 24. Calculate point differential feature: home_pts_L5 - away_pts_L5
games['pt_diff_advantage'] = ...