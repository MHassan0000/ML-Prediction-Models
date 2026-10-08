"""
Modal Deployment – Heart Disease Prediction App
================================================
modal deploy modal/modal_heart_disease.py        # permanent
modal serve  modal/modal_heart_disease.py        # ephemeral dev
"""

from pathlib import Path
import modal

app = modal.App("heart-disease-predictor")

ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Build image: install deps AND bake app files in at build time ─
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit==1.45.1",
        "scikit-learn==1.9.1",
        "pandas==2.2.3",
        "numpy==2.2.4",
        "scipy==1.14.1",
        "seaborn==0.13.2",
        "matplotlib==3.9.2",
        "joblib==1.4.2",
        "threadpoolctl==3.5.0",
        "narwhals>=2.0.1",
    )
    .add_local_file(str(ROOT_DIR / "heart_disease_app.py"),               "/app/heart_disease_app.py")
    .add_local_file(str(ROOT_DIR / "models" / "heart_disease_svc_model.pkl"), "/app/models/heart_disease_svc_model.pkl")
    .add_local_file(str(ROOT_DIR / "models" / "heart_disease_le_cp.pkl"),     "/app/models/heart_disease_le_cp.pkl")
    .add_local_file(str(ROOT_DIR / "data" / "heart.csv"),                  "/app/data/heart.csv")
    .add_local_dir(str(ROOT_DIR / ".streamlit"),                            "/app/.streamlit")
)

# ── Web endpoint ──────────────────────────────────────────────────
@app.function(
    image=image,
    timeout=300,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=8501, startup_timeout=60)
def run():
    import subprocess, sys
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "/app/heart_disease_app.py",
        "--server.port",               "8501",
        "--server.address",            "0.0.0.0",
        "--server.headless",           "true",
        "--server.enableCORS",         "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats",  "false",
    ])
