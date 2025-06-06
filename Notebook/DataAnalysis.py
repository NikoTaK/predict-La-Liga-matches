import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

df_train = pd.read_csv('Data/SP1_all_seasons.csv')
df_test = pd.read_csv('Data/Matches24-25.csv')

print("Full history shape:", df_train.shape)
print("2024-2025 season shape:", df_test.shape)

overall_missing_percentage = df_train.isnull().sum().sum() / (df_train.shape[0] * df_train.shape[1]) * 100
print(f"Overall percentage of missing values in all seasons: {overall_missing_percentage:.2f}%")

df_train['FTR'].value_counts().plot(kind='bar')
plt.title('Distribution of Match Results')
plt.show()

dist = df_train['FTR'].value_counts(normalize=True) * 100
print('Percentage of results:')
print(f"Home Wins: {dist.get('H', 0):.2f}%")
print(f"Draws: {dist.get('D', 0):.2f}%")
print(f"Away Wins: {dist.get('A', 0):.2f}%")

plt.figure(figsize=(8, 5))
total_home_goals = df_train['FTHG'].sum()
total_away_goals = df_train['FTAG'].sum()
plt.bar(['Home Goals', 'Away Goals'], [total_home_goals, total_away_goals], color=['blue', 'orange'])
plt.title('Total Goals Scored (Home vs Away)')
plt.ylabel('Total Goals')
plt.show()

df_train['GoalDiff'] = df_train['FTHG'] - df_train['FTAG']


le = LabelEncoder()
teams = pd.concat([
    df_train['HomeTeam'], df_train['AwayTeam'],
    df_test['HomeTeam'], df_test['AwayTeam']
])
le.fit(teams)
df_train['HomeTeam_enc'] = le.transform(df_train['HomeTeam'])
df_train['AwayTeam_enc'] = le.transform(df_train['AwayTeam'])

print(df_train[['GoalDiff', 'HomeTeam_enc', 'AwayTeam_enc']].head())

print(df_train.head())