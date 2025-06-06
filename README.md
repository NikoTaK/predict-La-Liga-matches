# Predict La Liga Matches

This project aims to analyze and predict the outcomes of Spanish La Liga football matches using historical data and machine learning models. The workflow includes data analysis, feature engineering, and model training to forecast match results for the 2024-2025 season.

## Project Structure

- `Notebook/DataAnalysis.py`: Exploratory data analysis and preprocessing of match data.
- `Model/BestModel_imp.py`: Feature engineering and model training using XGBoost and SMOTE for class balancing.
- `Data/`: Contains historical match data and the 2024-2025 season matches.

## Data

- `Data/SP1_all_seasons.csv`: Historical La Liga match data (multiple seasons).
- `Data/Matches24-25.csv`: Match data for the 2024-2025 season (for prediction/testing).

## Setup

1. **Clone the repository**
2. **With your environment activated Install dependencies**:
   ```bash
   pip freeze > requirements.txt
   ```
3. **Ensure data files are in the `Data/` directory**

## Usage

### 1. Data Analysis
Run the exploratory data analysis script to understand the data distribution and missing values:
```bash
python Notebook/DataAnalysis.py
```
This script:
- Prints dataset shapes and missing value percentages
- Visualizes match result distributions
- Shows total goals scored by home and away teams
- Encodes team names for modeling

### 2. Model Training & Prediction
Train the model and predict match outcomes for the 2024-2025 season:
```bash
python Model/BestModel_imp.py
```
This script:
- Performs feature engineering (recent form, averages, odds ratios, etc.)
- Handles missing values and class imbalance
- Trains an XGBoost classifier with hyperparameter tuning
- Evaluates model performance and prints classification metrics
- Displays feature importances

## Key Features
- **Feature Engineering**: Uses rolling averages, form, and betting odds to enhance predictive power.
- **Class Imbalance Handling**: Applies SMOTE to balance match result classes.
- **Model Selection**: Uses XGBoost with grid search for optimal parameters.
- **Evaluation**: Reports accuracy, log loss, and detailed classification metrics.

## Results
- The model outputs accuracy, log loss, and a classification report for the 2024-2025 season predictions.
- Feature importances are displayed to interpret model decisions.

## Requirements
- Python 3.7+
- pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost

## License
This project is for educational and research purposes.
