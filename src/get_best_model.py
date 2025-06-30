from mlflow.tracking import MlflowClient
import mlflow

def GetBestModel():
    """
    Retrieves the best trained model from the MLflow experiment named 'credit-risk-modle' based on the highest ROC AUC score.
    Returns:
        model: The scikit-learn model object loaded from the best MLflow run.
    Raises:
        Exception: If the experiment does not exist or no runs are found.
    """
    
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name('credit-risk-modle')
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=['metrics.roc_auc DESC'],
            max_results=1
        )

        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        return model
    except Exception as e:
        raise Exception(f"Failed to retrieve the best model: {str(e)}")