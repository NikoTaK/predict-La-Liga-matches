import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv('Data/SP1_all_seasons.csv')
df_test = pd.read_csv('Data/Matches24-25.csv')

print("Full history shape:", df_train.shape)
print("2024-2025 season shape:", df_test.shape)

(df_train.isnull().sum()/ df_train.shape[0]).to_csv('Data/LaLiga_Matches_nulls.csv')
print(df_test.isnull().sum()/ df_test.shape[0])

df_train['FTR'].value_counts().plot(kind='bar')
plt.title('Distribution of Match Results')
plt.show()

num_cols = ['FTHG', 'FTAG']
df_train[num_cols].hist(bins=20, figsize=(10,4))
plt.suptitle('Goal Distributions')
plt.show()

df_train['GoalDiff'] = df_train['FTHG'] - df_train['FTAG']

le = LabelEncoder()
df_train['HomeTeam_enc'] = le.fit_transform(df_train['HomeTeam'])
df_train['AwayTeam_enc'] = le.fit_transform(df_train['AwayTeam'])

print(df_train[['GoalDiff', 'HomeTeam_enc', 'AwayTeam_enc']].head())

print(df_train.head())