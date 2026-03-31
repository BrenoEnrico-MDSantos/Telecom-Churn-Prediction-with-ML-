import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as IbmPipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

path = "C:/Users/Acer/Downloads/original_customer_data.csv"
df = pd.read_csv(path)

recently_joined = df[(df['Customer_Status'] == 'Joined')]  # .dropna(axis= 'columns', how= 'all')
stay_and_churn = df[(df['Customer_Status'] != 'Joined')]  # .dropna(axis= 'columns', how= 'all')

data = stay_and_churn.drop(['Customer_ID', 'Churn_Category', 'Churn_Reason'], axis=1)
numerical_columns = stay_and_churn.select_dtypes(include=['int64', 'float64']).columns.to_list()
ohe_columns = ['Gender', 'Married', 'Phone_Service', 'Multiple_Lines', 'Internet_Service', 'Internet_Type',
               'Online_Security', 'Online_Backup', 'Device_Protection_Plan', 'Premium_Support',
               'Streaming_TV', 'Streaming_Movies', 'Streaming_Music', 'Unlimited_Data', 'Contract',
               'Paperless_Billing', 'Payment_Method']
ordinal_columns = ['Value_Deal']
te_columns = ['State']

data['Customer_Status'] = data['Customer_Status'].map({'Stayed': 0, 'Churned': 1})

X = data.drop('Customer_Status', axis=1)
y = data['Customer_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor_rf = ColumnTransformer(transformers=[
    ('cat', TargetEncoder(smooth=1.0), te_columns),
    ('onehot', OneHotEncoder(), ohe_columns),
    ('ord', OrdinalEncoder(categories=[['No', 'Deal 1', 'Deal 2', 'Deal 3', 'Deal 4', 'Deal 5']]), ordinal_columns),
    ('num', 'passthrough', numerical_columns)
])

# Storing in a dict the column of model names alongside the pipelines that hold encoding and clf
pipelines = {
    "RandomForestClassifier": Pipeline(steps= [
        ('encodings', preprocessor_rf),
        ('rfc', RandomForestClassifier(random_state = 42))]),
    "OversampledRFC": IbmPipeline(steps = [
        ('encodings', preprocessor_rf),
        ('clf', ADASYN(sampling_strategy= 'auto', random_state = 42)),
        #('smote', SMOTE(random_state = 42)),
        ('overfc', RandomForestClassifier(random_state = 42))]),
    "EasyEnsemblerClassifier": IbmPipeline(steps = [
        ('encodings', preprocessor_rf),
        #('adasyn', ADASYN(sampling_strategy= 'auto', random_state = 42)),
        #('smote', SMOTE(random_state = 42)),
        #('smotetomek', SMOTETomek(random_state = 42)),
        ('eec', EasyEnsembleClassifier(random_state = 42))]),
    "BalancedRF": Pipeline(steps = [
        ('encodings', preprocessor_rf),
        ('brf', BalancedRandomForestClassifier(random_state = 42))]),
    "XGBoost": Pipeline(steps = [
        ('xgb', XGBClassifier(enable_categorical= True, random_state = 42))])
}

param_grid = {
    'XGBoost': {
        'xgb__n_estimators' : [100],
        'xgb__max_depth' : [None, 3],
        'xgb__learning_rate' : [0.01, 0.1, 0.2],
        'xgb__gamma' : [0.1],
        'xgb__alpha' : [0],
        'xgb__lambda' : [1],
        'xgb__scale_pos_weight' : [(y == 0).sum() / (y == 1).sum(), 1]
    },
    'RandomForestClassifier': {},
    'OversampledRFC' : {},
    'EasyEnsemblerClassifier' : {},
    'BalancedRF' : {}
}

y_preds = {}
y_preds_proba = {}

grid_ranking = []

recall_churn_scorer = make_scorer(recall_score, pos_label = 0)

for name, pipe in pipelines.items():
    gs = GridSearchCV(estimator= pipelines[name],
                      param_grid= param_grid[name],
                      scoring = {'AUC': 'roc_auc', 'F1': 'f1', 'Recall_Churn': recall_churn_scorer},
                      refit = 'Recall_Churn',
                      cv = 5,
                      verbose= 1)
    gs.fit(X_train, y_train)

    grid_ranking.append({
        'model_name': name,
        'best_estimator': gs.best_estimator_,
        'AUC': gs.cv_results_['mean_test_AUC'][gs.best_index_],
        'F1' : gs.cv_results_['mean_test_F1'][gs.best_index_],
        'recall_churn': gs.cv_results_['mean_test_Recall_Churn'][gs.best_index_]
    })

    model_ranking = pd.DataFrame(grid_ranking)
    best_model = model_ranking.sort_values(by = ['F1', 'recall_churn'], ascending = False)

    print(f"\n{name} trained!")

    print(f"Best model: {best_model['model_name']}")
    print(best_model)

    y_pred = gs.predict(X_test)
    y_preds['predicted'] = y_pred
    y_pred_proba = gs.predict_proba(X_test)[:, 1]
    y_preds_proba['predicted'] = y_pred_proba
    AUC = roc_auc_score(y_test, y_pred_proba)

    #print(f"Confusion Matrix for {name}")
    #print(confusion_matrix(y_test, y_pred))
    #print("\nClassification Report for {}:".format(name))
    #print(classification_report(y_test, y_pred))
    #print("Area Under Curve for {}: {:.2f}".format(name, AUC))
