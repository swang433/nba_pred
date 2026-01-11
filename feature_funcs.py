import pandas as pd
'''
calculates rolling average of "attr" with size "win_sz" 
main_games is the dataframe to be modified by adding a new column "new_col_name"
'''
def rolling_avg(main_games, attr, win_sz, new_col_name): 
    # Reset index to avoid merge issues
    main_games = main_games.reset_index(drop=True)
    
    #extract necessary columns for home and away stats 
    home = main_games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', attr + '_home']].copy()
    home.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', attr]
    away = main_games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', attr + '_away']].copy()
    away.columns = ['GAME_ID', 'GAME_DATE_EST', 'TEAM_ID', attr]
    
    #calculate rolling averages
    together = pd.concat([home, away], ignore_index=True)
    together = together.sort_values(['TEAM_ID', 'GAME_DATE_EST']).reset_index(drop=True)
    together[new_col_name] = together.groupby('TEAM_ID')[attr].transform(
        lambda x: x.rolling(window=win_sz, min_periods=1).mean().shift(1)
    )
    
    #drop duplicates to avoid memory leak
    together_home = together[['GAME_ID', 'TEAM_ID', new_col_name]].copy().drop_duplicates()
    together_away = together[['GAME_ID', 'TEAM_ID', new_col_name]].copy().drop_duplicates()
    
    #merge back to the main dataframe with suffixes to control naming conflicts
    main_games = main_games.merge(
        together_home, 
        left_on=['GAME_ID', 'HOME_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_home'))
    main_games = main_games.drop('TEAM_ID', axis=1)
    main_games = main_games.rename(columns={new_col_name: new_col_name + '_home'})
    
    main_games = main_games.merge(
        together_away,
        left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_away'))
    main_games = main_games.drop('TEAM_ID', axis=1)
    main_games = main_games.rename(columns={new_col_name: new_col_name + '_away'})
    
    return main_games

def split(main_games, split_ratio): 
    split_index = int(len(main_games) * split_ratio)
    train = main_games.iloc[:split_index]
    test = main_games.iloc[split_index:]
    return train, test