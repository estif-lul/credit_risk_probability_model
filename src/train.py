import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow

df = pd.read_csv('data/processed/processed_customers.csv')
X = df.drop(columns=['is_high_risk'])
y = df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,  test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a classification model using various performance metrics.
    Parameters:
        model: A trained classification model with `predict` and `predict_proba` methods.
        X_test (array-like): Test feature data.
        y_test (array-like): True labels for the test data.
    Returns:
        dict: A dictionary containing the following evaluation metrics:
            - 'accuracy': Accuracy of the model.
            - 'precision': Precision score.
            - 'recall': Recall score.
            - 'f1_score': F1 score.
            - 'roc_auc': Area under the ROC curve (AUC).
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

#MLflow setup
mlflow.set_experiment('credit-risk-modle')

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        input_example = X_train.iloc[:1]

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.set_tag("model_type", name)
        mlflow.sklearn.log_model(model, name = "model", input_example=input_example)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        
        # Register the model in the Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=name)
    print(f"{name} logged with metrics: {metrics}")




