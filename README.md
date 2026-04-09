# Telecom Churn Prediction with ML
Churner prediction using binary classifiers in python, alongside PowerBI for report and A/B testing.

## Steps

1. Data loading and cleaning; initial exploration with Pandas and Numpy
2. Kprototypes clustering dropping most of service-related features; use of StandardScaler for numericals; custom gamma for weighted categoricals (?); Elbow plot and Silhouette score; Scatterplot with Factorial Analysis for Mixed Data (FAMD); Spider/Radar chart for profiling
3. Kaplan-Meier Curve for main categorical features survivability over months of tenure
4. Feature correlation and COX PH for spotting columns of highest HR
5. Auto ranking and choice of classification model with pipelines and GridSearchCV tuning; recall and f1 as main metrics for retention strat
6. Apply best model to recently joined customers and divise retention and winback measures (
7. PowerBI viz with slicers; report of losses to churn, distribution across categoricals and highest Hazard Ratio assessed; prediction panel with costs of strategies and imminent losses (emphasize emergency for top clusters)

Classification Models:
1. Tuned XGBoost
2. Reg cutoff BRFC
3. Vanilla EasyEnsembleClassifier
4. ADASYN Tuned reg cutoff RFC
5. Vanilla RFC
