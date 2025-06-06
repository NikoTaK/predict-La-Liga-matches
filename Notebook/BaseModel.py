import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

DF_PATH = 'Data/SP1_all_seasons.csv'

cols = [
    'HomeTeam', 'AwayTeam',
    'B365H', 'B365D', 'B365A',
    'BWH', 'BWD', 'BWA',
    'IWH', 'IWD', 'IWA',
    'PSH', 'PSD', 'PSA',
    'FTR'
]

matches = pd.read_csv(DF_PATH, low_memory=False)

matches = matches[cols].dropna()

team_encoder = LabelEncoder()
team_encoder.fit(pd.concat([matches['HomeTeam'], matches['AwayTeam']]))
matches['HomeTeam_enc'] = team_encoder.transform(matches['HomeTeam'])
matches['AwayTeam_enc'] = team_encoder.transform(matches['AwayTeam'])

print(matches.head())

result_encoder = LabelEncoder()
y = result_encoder.fit_transform(matches['FTR'])

feature_cols = [
    'HomeTeam_enc', 'AwayTeam_enc',
    'B365H', 'B365D', 'B365A',
    'BWH', 'BWD', 'BWA',
    'IWH', 'IWD', 'IWA',
    'PSH', 'PSD', 'PSA'
]
X = matches[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
weight_dict = {i: w for i, w in enumerate(class_weights)}
draw_idx = result_encoder.transform(['D'])[0]
weight_dict[draw_idx] *= 1.5

model = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight=weight_dict)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, preds, target_names=result_encoder.classes_))