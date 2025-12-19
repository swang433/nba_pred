import pandas as pd
def split(main_games, split_ratio): 
    split_index = int(len(main_games) * split_ratio)
    test = main_games.iloc[:split_index]
    train = main_games.iloc[split_index:]
    return train, test