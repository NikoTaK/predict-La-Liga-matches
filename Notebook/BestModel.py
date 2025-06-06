import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

DATA_PATH = "Data/SP1_all_seasons.csv"

cols = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "B365H",
    "B365D",
    "B365A",
]

df = pd.read_csv(DATA_PATH, usecols=cols, dayfirst=True, parse_dates=["Date"])

df = df.dropna()

df = df.sort_values("Date").reset_index(drop=True)

teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
track = {t: {"gf": [], "ga": [], "pts": []} for t in teams}

home_gf_avg5, home_ga_avg5, away_gf_avg5, away_ga_avg5 = [], [], [], []
home_form5, away_form5 = [], []

point_map_home = {"H": 3, "D": 1, "A": 0}
point_map_away = {"A": 3, "D": 1, "H": 0}

for _, row in df.iterrows():
    h, a = row["HomeTeam"], row["AwayTeam"]
    hs, as_ = track[h], track[a]

    home_gf_avg5.append(np.mean(hs["gf"][-5:]) if hs["gf"] else 0)
    home_ga_avg5.append(np.mean(hs["ga"][-5:]) if hs["ga"] else 0)
    away_gf_avg5.append(np.mean(as_["gf"][-5:]) if as_["gf"] else 0)
    away_ga_avg5.append(np.mean(as_["ga"][-5:]) if as_["ga"] else 0)
    home_form5.append(np.mean(hs["pts"][-5:]) if hs["pts"] else 0)
    away_form5.append(np.mean(as_["pts"][-5:]) if as_["pts"] else 0)

    hs["gf"].append(row["FTHG"])
    hs["ga"].append(row["FTAG"])
    as_["gf"].append(row["FTAG"])
    as_["ga"].append(row["FTHG"])

    hs["pts"].append(point_map_home[row["FTR"]])
    as_["pts"].append(point_map_away[row["FTR"]])

df["home_gf_avg5"] = home_gf_avg5

df["home_ga_avg5"] = home_ga_avg5

df["away_gf_avg5"] = away_gf_avg5

df["away_ga_avg5"] = away_ga_avg5

df["home_form5"] = home_form5

df["away_form5"] = away_form5

df["month"] = df["Date"].dt.month

df["year"] = df["Date"].dt.year

df["home_implied"] = 1 / df["B365H"]

df["draw_implied"] = 1 / df["B365D"]

df["away_implied"] = 1 / df["B365A"]

df["odds_ratio"] = df["B365H"] / df["B365A"]

df["draw_ratio"] = df["B365D"] / ((df["B365H"] + df["B365A"]) / 2)

team_le = LabelEncoder()
team_le.fit(pd.concat([df["HomeTeam"], df["AwayTeam"]]))

df["HomeTeam_enc"] = team_le.transform(df["HomeTeam"])

df["AwayTeam_enc"] = team_le.transform(df["AwayTeam"])

result_le = LabelEncoder()
y = result_le.fit_transform(df["FTR"])

feature_cols = [
    "HomeTeam_enc",
    "AwayTeam_enc",
    "month",
    "year",
    "home_gf_avg5",
    "home_ga_avg5",
    "away_gf_avg5",
    "away_ga_avg5",
    "home_form5",
    "away_form5",
    "home_implied",
    "draw_implied",
    "away_implied",
    "odds_ratio",
    "draw_ratio",
]

X = df[feature_cols]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer()),
        ("smote", SMOTE(random_state=42)),
        (
            "model",
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
            ),
        ),
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.1, 0.05],
}

tscv = TimeSeriesSplit(n_splits=3)

gs = GridSearchCV(
    pipeline, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1, verbose=1
)

gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)

best_model = gs.best_estimator_

preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Log Loss:", log_loss(y_test, probs))
print(classification_report(y_test, preds, target_names=result_le.classes_))