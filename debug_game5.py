import pandas as pd

# Let me trace the specific issue with Game 5
test_data = {
    'GAME_ID': [1, 2, 3, 4, 5, 6],
    'GAME_DATE_EST': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25'],
    'HOME_TEAM_ID': [100, 100, 200, 100, 200, 100],
    'VISITOR_TEAM_ID': [200, 200, 100, 200, 100, 200],
    'HOME_TEAM_WINS': [1, 0, 1, 1, 0, 0]
}

test_games = pd.DataFrame(test_data)
test_games = test_games.sort_values(['GAME_DATE_EST', 'GAME_ID'])

print("Focus on Team 200 home vs Team 100 away:")
team_200_vs_100 = test_games[(test_games['HOME_TEAM_ID'] == 200) & (test_games['VISITOR_TEAM_ID'] == 100)]
print(team_200_vs_100[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_WINS']])

print("\nGame 3: Team 200 wins (HOME_TEAM_WINS = 1)")
print("Game 5: Team 200 loses (HOME_TEAM_WINS = 0)")

print("\nExpanding mean calculation for this group:")
print("After Game 3: 1/1 = 1.0")
print("After Game 5: (1+0)/2 = 0.5")

print("\nWith shift(1):")
print("Game 3 gets: NaN (no prior history)")
print("Game 5 gets: 1.0 (the expanding mean after Game 3)")

print("\nSo Game 5 result should be 1.0, not 0.5!")
print("Let me check the actual output again...")

h2h = test_games.groupby(['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])['HOME_TEAM_WINS'].expanding().mean().shift(1)
team_200_results = h2h[(200, 100)]
print(f"\nActual results for Team 200 vs 100: {team_200_results}")

# Check which games these correspond to
print("\nGame index mapping:")
team_200_games = test_games[(test_games['HOME_TEAM_ID'] == 200) & (test_games['VISITOR_TEAM_ID'] == 100)]
for idx, row in team_200_games.iterrows():
    print(f"Index {idx}: Game {row['GAME_ID']}, Result: {h2h.loc[(200, 100), idx] if (200, 100, idx) in h2h.index else 'Not found'}")