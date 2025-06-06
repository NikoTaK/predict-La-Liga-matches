import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

main_df = pd.read_csv('Data/SP1_all_seasons.csv', dayfirst=True, parse_dates=['Date'])

recent_df = pd.read_csv('Data/Matches24-25.csv', dayfirst=True, parse_dates=['Date'])
recent_df['season'] = '2024-25'

missing_cols = set(main_df.columns) - set(recent_df.columns)
for col in missing_cols:
    recent_df[col] = np.nan
recent_df = recent_df[main_df.columns]

full_df = pd.concat([main_df, recent_df], ignore_index=True)
full_df = full_df.sort_values('Date').reset_index(drop=True)

full_df = full_df.dropna(subset=['HomeTeam','AwayTeam','FTR','B365H','B365D','B365A'])

teams = pd.unique(full_df[['HomeTeam','AwayTeam']].values.ravel())
track = {t:{'gf':[],'ga':[],'pts':[]} for t in teams}

stats = {
    'home_gf_avg5':[], 'home_ga_avg5':[], 'away_gf_avg5':[], 'away_ga_avg5':[],
    'home_form5':[], 'away_form5':[]
}

map_home = {'H':3,'D':1,'A':0}
map_away = {'A':3,'D':1,'H':0}

for _, row in full_df.iterrows():
    h, a = row['HomeTeam'], row['AwayTeam']
    hs, as_ = track[h], track[a]
    stats['home_gf_avg5'].append(np.mean(hs['gf'][-5:]) if hs['gf'] else 0)
    stats['home_ga_avg5'].append(np.mean(hs['ga'][-5:]) if hs['ga'] else 0)
    stats['away_gf_avg5'].append(np.mean(as_['gf'][-5:]) if as_['gf'] else 0)
    stats['away_ga_avg5'].append(np.mean(as_['ga'][-5:]) if as_['ga'] else 0)
    stats['home_form5'].append(np.mean(hs['pts'][-5:]) if hs['pts'] else 0)
    stats['away_form5'].append(np.mean(as_['pts'][-5:]) if as_['pts'] else 0)

    hs['gf'].append(row['FTHG'])
    hs['ga'].append(row['FTAG'])
    as_['gf'].append(row['FTAG'])
    as_['ga'].append(row['FTHG'])
    hs['pts'].append(map_home[row['FTR']])
    as_['pts'].append(map_away[row['FTR']])

for key,val in stats.items():
    full_df[key] = val

full_df['month'] = full_df['Date'].dt.month
full_df['year'] = full_df['Date'].dt.year
full_df['home_implied'] = 1 / full_df['B365H']
full_df['draw_implied'] = 1 / full_df['B365D']
full_df['away_implied'] = 1 / full_df['B365A']
full_df['odds_ratio'] = full_df['B365H'] / full_df['B365A']
full_df['draw_ratio'] = full_df['B365D'] / ((full_df['B365H'] + full_df['B365A']) / 2)

team_le = LabelEncoder()
team_le.fit(pd.concat([full_df['HomeTeam'], full_df['AwayTeam']]))
full_df['HomeTeam_enc'] = team_le.transform(full_df['HomeTeam'])
full_df['AwayTeam_enc'] = team_le.transform(full_df['AwayTeam'])

result_le = LabelEncoder()
y = result_le.fit_transform(full_df['FTR'])

feature_cols = ['HomeTeam_enc','AwayTeam_enc','month','year','home_gf_avg5','home_ga_avg5',
                'away_gf_avg5','away_ga_avg5','home_form5','away_form5','home_implied',
                'draw_implied','away_implied','odds_ratio','draw_ratio']
X = full_df[feature_cols]

train_mask = full_df['season'] != '2024-25'
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False))
])

param_grid = {
    'model__n_estimators':[100,200],
    'model__max_depth':[3,5],
    'model__learning_rate':[0.1,0.05]
}

gs = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='accuracy', n_jobs=-1, verbose=1)

gs.fit(X_train, y_train)
print('Best params:', gs.best_params_)

best_model = gs.best_estimator_

preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)
print('Accuracy:', accuracy_score(y_test, preds))
print('Log Loss:', log_loss(y_test, probs))
print(classification_report(y_test, preds, target_names=result_le.classes_))