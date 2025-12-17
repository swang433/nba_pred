import pandas as pd
import numpy as np

def get_processed_games():
    """Load and process NBA games data with all engineered features"""
    
    games = pd.read_csv('nba_games/games.csv')
    pd.set_option('display.max_columns', None)

    # Clean data at the source to prevent cascading issues
    print(f'Original games shape: {games.shape}')
    print(f'Duplicate GAME_IDs: {games["GAME_ID"].duplicated().sum()}')
    print(f'Missing PTS values: {games["PTS_home"].isna().sum() + games["PTS_away"].isna().sum()}')

    # Remove duplicates and missing values
    games = games.drop_duplicates(subset=['GAME_ID'], keep='first')
    games = games.dropna(subset=['PTS_home', 'PTS_away'])
    print(f'Cleaned games shape: {games.shape}')

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
    best_shooting_home_team = games.groupby('HOME_TEAM_ID').agg({'FG_PCT_home': 'mean'}).sort_values('FG_PCT_home', ascending=False).head(1)

    # 9. Calculate win percentage for each team at home
    home_win_pct = games.groupby('HOME_TEAM_ID')['HOME_TEAM_WINS'].mean()

    # 10. Get total assists per team across all their games (home + away)
    home_ast = games.groupby('HOME_TEAM_ID')['AST_home'].sum()
    away_ast = games.groupby('VISITOR_TEAM_ID')['AST_away'].sum()
    total_ast = home_ast.add(away_ast, fill_value=0)

    ########## pt3
    # 11. Calculate each team's average points in their last 3 HOME games
    games['pts_L3'] = games.groupby('HOME_TEAM_ID')['PTS_home'].transform(
        lambda x: x.rolling(window=3).mean().shift(1)
    )

    # 12. Calculate each team's average FG% over last 5 games
    home_L5 = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'FG_PCT_home']].copy()
    home_L5.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'FG_PCT']
    away_L5 = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'FG_PCT_away']].copy()
    away_L5.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'FG_PCT']

    all_L5 = pd.concat([home_L5, away_L5], ignore_index=True)
    all_L5 = all_L5.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
    all_L5['FG_L5'] = all_L5.groupby('TEAM_ID')['FG_PCT'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )

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

    # 13. Create a 'hot streak' indicator: 1 if team scored 110+ in last 2 games
    L2_home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'PTS_home']].copy()
    L2_home.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'PTS']
    L2_away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'PTS_away']].copy()
    L2_away.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'PTS']

    L2_all = pd.concat([L2_home, L2_away], ignore_index=True)
    L2_all = L2_all.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
    L2_all['hot_streak'] = L2_all.groupby('TEAM_ID')['PTS'].transform(
        lambda x: ((x.shift(1) >= 110) & (x.shift(2) >= 110)).astype(int)
    )

    games = games.merge(
        L2_all[['GAME_ID', 'TEAM_ID', 'hot_streak']], 
        left_on=['GAME_ID', 'HOME_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'hot_streak': 'HOT_STREAK_home'}).drop('TEAM_ID', axis=1)

    games = games.merge(
        L2_all[['GAME_ID', 'TEAM_ID', 'hot_streak']], 
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'hot_streak': 'HOT_STREAK_away'}).drop('TEAM_ID', axis=1)

    # 14. Calculate rolling standard deviation of points for each team (last 5 games)
    games = games.sort_values('GAME_DATE_EST')
    STD_home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'PTS_home']].copy()
    STD_home.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'PTS']
    STD_away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'PTS_away']].copy()
    STD_away.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID' ,'PTS']

    STD_all = pd.concat([STD_home, STD_away], ignore_index=True)
    STD_all = STD_all.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
    STD_all['PTS_STD_L5'] = STD_all.groupby('TEAM_ID')['PTS'].transform(
        lambda x: x.rolling(window=5, min_periods=2).std().shift(1)
    )

    games = games.merge(
        STD_all[['GAME_ID', 'TEAM_ID', 'PTS_STD_L5']], 
        left_on=['GAME_ID', 'HOME_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'PTS_STD_L5': 'PTS_STD_L5_home'}).drop('TEAM_ID', axis=1)

    games = games.merge(
        STD_all[['GAME_ID', 'TEAM_ID', 'PTS_STD_L5']], 
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'PTS_STD_L5': 'PTS_STD_L5_away'}).drop('TEAM_ID', axis=1)

    # 15. Calculate days between games for each team
    REST_home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID']].copy()
    REST_home.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID']
    REST_away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID']].copy()
    REST_away.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID']

    REST_all = pd.concat([REST_home, REST_away], ignore_index=True)
    REST_all = REST_all.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
    REST_all['GAME_DATE_EST'] = pd.to_datetime(REST_all['GAME_DATE_EST'])

    REST_all['rest_days'] = REST_all.groupby('TEAM_ID')['GAME_DATE_EST'].transform(
        lambda x: x.diff().dt.days
    )
    REST_all['b2b'] = (REST_all['rest_days'] == 1).astype('int')

    games = games.merge(
        REST_all[['GAME_ID', 'TEAM_ID', 'rest_days']], 
        left_on=['GAME_ID', 'HOME_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'rest_days': 'rest_days_home'}).drop('TEAM_ID', axis=1)

    games = games.merge(
        REST_all[['GAME_ID', 'TEAM_ID', 'rest_days']], 
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'rest_days': 'rest_days_away'}).drop('TEAM_ID', axis=1)

    # 16-18. Create date and back-to-back features
    games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
    games['day_of_week'] = games['GAME_DATE_EST'].dt.day_of_week
    games['month'] = games['GAME_DATE_EST'].dt.month

    games = games.merge(
        REST_all[['GAME_ID', 'TEAM_ID', 'b2b']],
        left_on=['GAME_ID', 'HOME_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'b2b': 'B2B_home'}).drop('TEAM_ID', axis=1)

    games = games.merge(
        REST_all[['GAME_ID', 'TEAM_ID', 'b2b']], 
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'b2b': 'B2B_away'}).drop('TEAM_ID', axis=1)
    
    games['B2B'] = ((games['B2B_home'] == 1) | (games['B2B_away'] == 1)).astype('int')

    # 19-21. Calculate cumulative wins and streaks
    games['AWAY_TEAM_WINS'] = (games['HOME_TEAM_WINS'] == 0).astype('int')

    CUM_home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'HOME_TEAM_WINS']].copy()
    CUM_home.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'WIN']
    CUM_away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'AWAY_TEAM_WINS']].copy()
    CUM_away.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', 'WIN']

    CUM_all = pd.concat([CUM_away, CUM_home], ignore_index=True)
    CUM_all = CUM_all.sort_values(['TEAM_ID', 'GAME_DATE_EST'])

    CUM_all['CUM_WINS'] = CUM_all.groupby('TEAM_ID')['WIN'].transform(
        lambda x: x.cumsum().shift(1)
    )

    games = games.merge(
        CUM_all[['GAME_ID', 'TEAM_ID', 'CUM_WINS']], 
        left_on=['GAME_ID', 'HOME_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'CUM_WINS': 'CUM_WINS_home'}).drop('TEAM_ID', axis=1)

    games = games.merge(
        CUM_all[['GAME_ID', 'TEAM_ID', 'CUM_WINS']], 
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'], 
        right_on=['GAME_ID', 'TEAM_ID'], 
        how='left'
    ).rename(columns={'CUM_WINS': 'CUM_WINS_away'}).drop('TEAM_ID', axis=1)

    # 22. Head-to-head win record
    games = games.sort_values(['GAME_DATE_EST', 'GAME_ID'])
    h2h = games.groupby(['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])['HOME_TEAM_WINS'].expanding().mean().groupby(['HOME_TEAM_ID', 'VISITOR_TEAM_ID']).shift(1)

    # 24. Point differential feature
    PTS_L5_home = games.groupby('HOME_TEAM_ID')['PTS_home'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )

    PTS_L5_away = games.groupby('VISITOR_TEAM_ID')['PTS_away'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )

    games['diff_L5'] = PTS_L5_home - PTS_L5_away

    print("✅ Feature engineering completed!")
    print(f"Final dataset shape: {games.shape}")
    
    return games

# Global cache variable
_cached_games = None

def get_games():
    """Get processed games data (cached after first call)"""
    global _cached_games
    if _cached_games is None:
        print("🏀 Loading and processing NBA data...")
        _cached_games = get_processed_games()
    else:
        print("📋 Using cached NBA data")
    return _cached_games