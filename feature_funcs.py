import pandas as pd
'''
calculates rolling average of "attr" with size "win_sz" 
main_games is the dataframe to be modified by adding a new column "new_col_name"
'''

def rolling_avgs(games):
    '''
    Create all rolling features in ONE pass to avoid merge explosions
    
    Create list to store all team-level rolling stats
    
    
    For each stat, create home and away versions
    '''
    team_stats = []
    stats_to_roll = {
        'PTS': ('PPG_L5', 5),
        'REB': ('RPG_L5', 5),
        'AST': ('APG_L5', 5),
        'FG_PCT': ('FG_L5', 5)
    }
    
    for stat, (col_name, window) in stats_to_roll.items():
        # Home games
        home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', stat + '_home']].copy()
        home.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', stat]
        
        # Away games
        away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', stat + '_away']].copy()
        away.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', stat]
        
        # Combine
        together = pd.concat([home, away], ignore_index=True)
        together = together.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
        
        # Calculate rolling average
        together[col_name] = together.groupby('TEAM_ID')[stat].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        team_stats.append(together[['GAME_ID', 'TEAM_ID', col_name]])
    
    # Merge all team stats at once
    all_team_stats = team_stats[0]
    for df in team_stats[1:]:
        all_team_stats = all_team_stats.merge(df, on=['GAME_ID', 'TEAM_ID'], how='outer')
    
    # Now merge back to main games ONCE for home, ONCE for away
    games = games.merge(
        all_team_stats,
        left_on=['GAME_ID', 'HOME_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_home')
    ).drop('TEAM_ID', axis=1)
    
    # Rename home columns
    for stat, (col_name, _) in stats_to_roll.items():
        games = games.rename(columns={col_name: col_name + '_home'})
    
    games = games.merge(
        all_team_stats,
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_away')
    ).drop('TEAM_ID', axis=1)
    
    # Rename away columns
    for stat, (col_name, _) in stats_to_roll.items():
        games = games.rename(columns={col_name: col_name + '_away'})
    
    return games

def split(main_games, split_ratio): 
    split_index = int(len(main_games) * split_ratio)
    train = main_games.iloc[:split_index]
    test = main_games.iloc[split_index:]
    return train, test