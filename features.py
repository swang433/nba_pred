import pandas as pd

#read and sort
games = pd.read_csv('nba_games/games.csv').sort_values('GAME_DATE_EST')

'''
calculates rolling average of "attr" with size "win_sz" 
main_games is the dataframe to be modified by adding a new column "new_col_name"
'''
def rolling_avg(main_games, attr, win_sz, new_col_name): 
    #extract necessary columns for home and away stats 
    home = main_games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', attr + '_home']].copy()
    home.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', attr]
    away = main_games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', attr + '_away']].copy()
    away.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', attr]
    
    #merge and calculate rolling averages
    together = pd.concat([home, away], ignore_index=True)
    together = together.sort_values(['TEAM_ID', 'GAME_DATE_EST'])
    together[new_col_name] = together.groupby('TEAM_ID')[attr].transform(
        lambda x: x.rolling(window=win_sz, min_periods=1).mean().shift(1)
    )
    
    # Merge back to games df for home and away teams
    together_home = together[['GAME_ID', 'TEAM_ID', new_col_name]].copy()
    together_home.columns = ['GAME_ID', 'HOME_TEAM_ID', new_col_name + '_home']
    together_away = together[['GAME_ID', 'TEAM_ID', new_col_name]].copy()
    together_away.columns = ['GAME_ID', 'VISITOR_TEAM_ID', new_col_name + '_away']
    
    main_games = main_games.merge(
        together_home, 
        on=['GAME_ID', 'HOME_TEAM_ID'], 
        how='left')
    
    main_games = main_games.merge(
        together_away,
        on=['GAME_ID', 'VISITOR_TEAM_ID'],
        how='left')
    
    return main_games
    
#rolling average points scored for last 5 games
sample = rolling_avg(games, 'PTS', 5, 'PTS_L5')
print(sample.columns)
# print(games.columns)