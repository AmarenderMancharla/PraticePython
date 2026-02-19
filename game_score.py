# game_score.py

# Input Collection
player_name = input("Enter player's name: ")

# Numeric Input Processing
games_played = int(input("Enter number of games played: "))

# Score Data Entry
total_score = int(input("Enter total score achieved: "))

# Computation
average_score = total_score / games_played

# Output Display
print("\nPlayer:", player_name)
print("\nGames Played:", games_played)
print("\nTotal Score:", total_score)
print("\nAverage Score:", average_score)
