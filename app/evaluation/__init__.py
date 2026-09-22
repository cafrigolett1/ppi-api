"""Offline evaluation: metrics, an API-driven runner, and MLflow tracking.

Used by ``notebooks/benchmark.ipynb``, not by the service. Its dependencies
(``mlflow``, ``pandas``) live in the ``notebook`` dependency group and the
package is excluded from the Docker images, so importing anything here from
``app.main`` or ``app.routes`` would break the container build.

Submodules are not imported eagerly: ``tracking`` imports mlflow, which is
slow and optional.
"""

__all__ = ["metrics", "runner", "tracking"]
