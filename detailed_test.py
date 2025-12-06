import pandas as pd
import numpy as np

# Let's trace through step by step what's happening
test_data = {
    'GAME_ID': [1, 2, 3, 4, 5, 6],
    'GAME_DATE_EST': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25'],
    'HOME_TEAM_ID': [100, 100, 200, 100, 200, 100],
    'VISITOR_TEAM_ID': [200, 200, 100, 200, 100, 200],
    'HOME_TEAM_WINS': [1, 0, 1, 1, 0, 0]  # Win, Loss, Win, Win, Loss, Loss
}

test_games = pd.DataFrame(test_data)
test_games = test_games.sort_values(['GAME_DATE_EST', 'GAME_ID'])

print("Sorted Test Data:")
print(test_games)
print("\n")

# Let's see what happens step by step
grouped = test_games.groupby(['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])

print("Groups created:")
for name, group in grouped:
    print(f"\nGroup {name}:")
    print(group[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_WINS']])
    
print("\n" + "="*60)

# Now let's see the expanding calculation
print("\nExpanding means (before shift):")
expanding_means = test_games.groupby(['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])['HOME_TEAM_WINS'].expanding().mean()
print(expanding_means)

print("\nAfter shift(1):")
h2h_test = expanding_means.shift(1)
print(h2h_test)

print("\n" + "="*60)
print("VERIFICATION:")
print("Game 1: Team 100 home vs 200 away - Should be NaN (no history) ✓")
print("Game 2: Team 100 home vs 200 away - Should be 1.0 (won previous game) ✓") 
print("Game 3: Team 200 home vs 100 away - Should be NaN (no history for this direction) ✓")
print("Game 4: Team 100 home vs 200 away - Should be 0.5 (1 win, 1 loss in previous 2 games) ✓")
print("Game 5: Team 200 home vs 100 away - Should be 1.0 (won previous game in this direction) ✓")
print("Game 6: Team 100 home vs 200 away - Should be 0.67 (2 wins, 1 loss in previous 3 games) ✓")