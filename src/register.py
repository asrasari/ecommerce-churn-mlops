"""Register the best run's model and walk it Staging -> Production."""
import mlflow
from mlflow.tracking import MlflowClient

from src import config


def pick_best(runs, metric=config.PRIMARY_METRIC):
    """Return the run with the highest value of `metric`."""
    if not runs:
        raise ValueError("no runs to choose from")
    return max(runs, key=lambda r: r.data.metrics.get(metric, float("-inf")))


def find_best_run(client, experiment_name=config.EXPERIMENT_NAME,
                  metric=config.PRIMARY_METRIC):
    """Search an experiment's runs (that logged a model) for the best metric."""
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=200,
    )
    # keep only runs that actually logged the primary metric
    runs = [r for r in runs if metric in r.data.metrics]
    return pick_best(runs, metric)


def main():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()

    best = find_best_run(client)
    auc = best.data.metrics[config.PRIMARY_METRIC]
    model_uri = f"runs:/{best.info.run_id}/model"
    print(f"Best run {best.info.run_id} {config.PRIMARY_METRIC}={auc:.4f}")

    # register a new version
    mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    print(f"Registered version {mv.version}")

    # Staging
    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME, version=mv.version, stage="Staging"
    )
    print(f"Version {mv.version} -> Staging")

    # validation gate -> Production
    if auc >= config.AUC_THRESHOLD:
        client.transition_model_version_stage(
            name=config.REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Version {mv.version} passed gate (AUC>={config.AUC_THRESHOLD}) -> Production")
    else:
        print(f"Version {mv.version} held in Staging (AUC<{config.AUC_THRESHOLD})")


if __name__ == "__main__":
    main()
