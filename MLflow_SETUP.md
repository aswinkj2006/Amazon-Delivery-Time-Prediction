MLflow setup and tracking guide

Local quickstart
1. Create and activate a Python venv (optional but recommended):
   python -m venv .venv
   .venv\Scripts\Activate.ps1

2. Install dependencies:
   pip install -r requirements.txt

3. Start MLflow tracking UI (local file store):
   mlflow ui --backend-store-uri file:./mlruns --port 5000

4. Run training script which logs to MLflow:
   python train.py

5. Open http://127.0.0.1:5000 to inspect runs, parameters, metrics and model artifacts.

Advanced: Remote tracking server and artifact store
- Use a SQL database (Postgres/MySQL) for the backend store and S3 for artifacts. See MLflow docs for configuration.

Notes
- The training script uses mlflow.sklearn.log_model to store the model artifact per run.
- Ensure network/firewall allows the chosen port when using remote servers.
