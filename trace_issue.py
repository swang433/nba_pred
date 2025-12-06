import pandas as pd

test_data = {
    'GAME_ID': [1, 2, 3, 4, 5, 6],
    'GAME_DATE_EST': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25'],
    'HOME_TEAM_ID': [100, 100, 200, 100, 200, 100],
    'VISITOR_TEAM_ID': [200, 200, 100, 200, 100, 200],
    'HOME_TEAM_WINS': [1, 0, 1, 1, 0, 0]
}

test_games = pd.DataFrame(test_data)
test_games = test_games.sort_values(['GAME_DATE_EST', 'GAME_ID'])

print("Let me trace step by step what pandas is actually doing:")
print("\nOriginal index after sort:")
print(test_games.reset_index(drop=True))

# The issue might be the index - let me check
h2h_no_shift = test_games.groupby(['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])['HOME_TEAM_WINS'].expanding().mean()
print(f"\nExpanding means (no shift):\n{h2h_no_shift}")

h2h_with_shift = h2h_no_shift.shift(1)
print(f"\nWith shift(1):\n{h2h_with_shift}")

print("\nProblem identified:")
print("The issue is that when we do .shift(1) on a grouped result,")
print("it shifts within the ENTIRE series, not within each group!")
print("This is why the values don't align correctly with the games.")

print(f"\nTo fix this, we need to shift within each group:")
h2h_correct = test_games.groupby(['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])['HOME_TEAM_WINS'].expanding().mean().groupby(['HOME_TEAM_ID', 'VISITOR_TEAM_ID']).shift(1)
print(f"Correct approach:\n{h2h_correct}")

print("\nNow let's verify:")
test_games_copy = test_games.copy()
test_games_copy['h2h_correct'] = h2h_correct.values

for i, row in test_games_copy.iterrows():
    h2h_val = f"{row['h2h_correct']:.3f}" if pd.notna(row['h2h_correct']) else "NaN"
    print(f"Game {row['GAME_ID']}: Team {row['HOME_TEAM_ID']} vs {row['VISITOR_TEAM_ID']} → H2H rate: {h2h_val}")