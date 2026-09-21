![Logo](static/images/light.png)

# AemroVision: Automated Brain Tumor MRI Classification

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-945DD6.svg?logo=dvc&logoColor=white)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Deep Learning system engineered for automated detection and multi-class classification of brain tumors from Magnetic Resonance Imaging (MRI) scans. The system accurately classifies brain scans into four clinical categories:

* **Glioma Tumor**
* **Meningioma Tumor**
* **Pituitary Tumor**
* **No Tumor (Healthy Control)**

Designed with robust MLOps practices, **AemroVision** implements modular pipeline architecture, automated data ingestion, transfer learning, experiment tracking with MLflow/DagsHub, containerization with Docker, and CI/CD deployment on AWS via GitHub Actions.

---

## 📌 Table of Contents

1. [Key Features](#-key-features)
2. [Model Architecture & Methodology](#-model-architecture--methodology)
3. [Project Directory Structure](#-project-directory-structure)
4. [Tech Stack](#-tech-stack)
5. [Getting Started & Installation](#-getting-started--installation)
6. [Pipeline Execution & Workflow](#-pipeline-execution--workflow)
7. [Web Application & API Usage](#-web-application--api-usage)
8. [Experiment Tracking with MLflow & DagsHub](#-experiment-tracking-with-mlflow--dagshub)
9. [Docker Containerization](#-docker-containerization)
10. [CI/CD & Cloud Deployment](#-cicd--cloud-deployment)
11. [License & Acknowledgements](#-license--acknowledgements)

---

## 🚀 Key Features

* **Modular 4-Stage MLOps Pipeline:** Strict separation of concerns across Data Ingestion, Base Model Preparation, Model Training, and Model Evaluation.
* **Transfer Learning Backbone:** Pre-trained **VGG16** architecture initialized with ImageNet weights, customized with dense classification layers and dropout for feature stability.
* **Configuration & Schema Driven:** Centralized configuration management using `config.yaml`, `params.yaml`, and strongly typed Python dataclasses (`config_entity.py`).
* **Experiment Tracking:** Real-time metrics logging, parameter tracking, and model artifact versioning using **MLflow** integrated with **DagsHub**.
* **High-Performance Microservice API:** Built with **FastAPI** and Uvicorn, serving both an interactive web UI and a REST API supporting base64-encoded image payloads.
* **Containerized Deployment:** Dockerized runtime environment ensuring consistency across development, staging, and production environments.
* **Automated CI/CD Pipeline:** GitHub Actions workflow managing linting, Docker image builds, Amazon ECR publishing, and automatic deployment to AWS EC2 via a self-hosted runner.

---

## 🧠 Model Architecture & Methodology

```
MRI Image (224x224x3)
        │
        ▼
VGG16 Convolutional Backbone (Frozen ImageNet Weights)
  ├── Block 1 (Conv2D + MaxPool)
  ├── Block 2 (Conv2D + MaxPool)
  ├── Block 3 (Conv2D + MaxPool)
  ├── Block 4 (Conv2D + MaxPool)
  └── Block 5 (Conv2D + MaxPool)
        │
        ▼
Flatten Layer (Output: 25,088 features)
        │
        ▼
Dense Layer (4 Units, Softmax Activation)
        │
        ▼
Output Class: [Glioma | Meningioma | No Tumor | Pituitary]
```

* **Input Resolution:** `224 x 224 x 3` (RGB)
* **Optimization:** Adam optimizer (`lr=0.0001`)
* **Loss Function:** Categorical Crossentropy
* **Data Augmentation:** Rotation, shear, zoom, width/height shifts, and horizontal flips.

---

## 📂 Project Directory Structure

```text
.
├── .github/
│   └── workflows/
│       └── main.yaml              # CI/CD workflow for ECR and EC2 deployment
├── config/
│   └── config.yaml                # Pipeline artifacts and directory configurations
├── model/
│   └── model.keras                # Production-ready trained model artifact
├── research/
│   ├── 01_data_ingestion.ipynb    # Prototyping data ingestion
│   └── trials.ipynb               # Experimental scratchpad
├── src/
│   └── cnnClassifier/
│       ├── components/            # Core business logic for each stage
│       │   ├── data_ingestion.py
│       │   ├── prepare_base_model.py
│       │   ├── model_training.py
│       │   └── model_evaluation.py
│       ├── config/
│       │   └── configuration.py   # Configuration Manager mapping YAMLs to Entities
│       ├── constants/             # Constant file paths (config.yaml, params.yaml)
│       ├── entity/
│       │   └── config_entity.py   # Strongly-typed Dataclasses
│       ├── pipeline/              # Stage execution scripts and inference logic
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_prepare_base_model.py
│       │   ├── stage_03_model_training.py
│       │   ├── stage_04_model_evaluation.py
│       │   └── prediction.py      # Standalone inference pipeline
│       └── utils/
│           └── common.py          # Helper utilities (file I/O, base64 encoding/decoding)
├── static/                        # Frontend UI assets (CSS, JS, images)
├── templates/
│   └── index.html                 # Web dashboard UI
├── .dockerignore                  # Docker exclusion rules
├── .gitignore                     # Git tracking exclusions
├── app.py                         # FastAPI web microservice
├── Dockerfile                     # Container build recipe
├── dvc.yaml                       # Data Version Control pipeline specification
├── main.py                        # Sequential pipeline orchestrator
├── params.yaml                    # Hyperparameters configuration
├── requirements.txt               # Pinned project dependencies
└── setup.py                       # Local package installer
```

---

## 🛠 Tech Stack

* **Machine Learning & Vision:** TensorFlow 2.x, Keras, NumPy, SciPy, Pillow, Matplotlib, Seaborn
* **API & Backend:** FastAPI, Uvicorn, Pydantic, Python-Box
* **Data & Pipeline Versioning:** DVC (Data Version Control), PyYAML
* **Experiment Management:** MLflow, DagsHub
* **DevOps & Cloud:** Docker, GitHub Actions, AWS ECR (Elastic Container Registry), AWS EC2

---

## 💻 Getting Started & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mebre7/AemroVision-Brain-Tumor-Classifier.git
cd AemroVision-Brain-Tumor-Classifier
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 🔄 Pipeline Execution & Workflow

The pipeline is organized sequentially across 4 distinct stages:

```bash
# Execute the complete end-to-end training and evaluation pipeline:
python main.py
```

Each stage can also be executed independently:
1. **Data Ingestion:** Downloads and extracts the MRI dataset.
   ```bash
   python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
   ```
2. **Base Model Preparation:** Downloads VGG16, creates custom classification layers, compiles and saves the architecture.
   ```bash
   python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
   ```
3. **Model Training:** Applies data augmentation, trains the head on MRI training batches, and outputs `model.keras`.
   ```bash
   python src/cnnClassifier/pipeline/stage_03_model_training.py
   ```
4. **Model Evaluation & MLflow Logging:** Evaluates validation accuracy/loss, writes `metrics.json`, and logs results to DagsHub MLflow.
   ```bash
   python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
   ```

---

## 🌐 Web Application & API Usage

Start the local FastAPI web server:
```bash
python app.py
```
Open your browser at **`http://localhost:8080`** (or configured port) to access the interactive web dashboard.

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web interface dashboard. |
| `POST` | `/predict` | Accepts a JSON payload with a base64-encoded image string and returns classification results. |

#### Sample Prediction Request (`POST /predict`)
```json
{
  "image": "<base64_encoded_mri_string>"
}
```

#### Sample Response
```json
{
  "status": "success",
  "data": [
    {
      "prediction": "Pituitary Tumor",
      "confidence": 98.42
    }
  ]
}
```

---

## 📊 Experiment Tracking with MLflow & DagsHub

All training parameters (`learning_rate`, `epochs`, `batch_size`, `image_size`) and evaluation metrics (`loss`, `accuracy`) are tracked centrally on DagsHub.

Set your tracking credentials:
```bash
# Set environment variables
export MLFLOW_TRACKING_URI="https://dagshub.com/mebratucheka7/AemroVision-Brain-Tumor-Classifier.mlflow"
export MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
export MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>"
```

View the live dashboard and parameter comparisons directly on your DagsHub repository under the **Experiments** tab.

---

## 🐳 Docker Containerization

To run the application inside an isolated Docker container:

```bash
# 1. Build the Docker image
docker build -t aemrovision:latest .

# 2. Run the container
docker run -d -p 8080:8080 --name aemro_app aemrovision:latest

# 3. Access in browser
http://localhost:8080
```

---

## ☁️ CI/CD & Cloud Deployment

The repository includes a continuous deployment workflow powered by **GitHub Actions**:

1. **Continuous Integration (CI):** Triggers on push to `main`, validating syntax and code integrity.
2. **Continuous Delivery (CD):** Authenticates with AWS, builds the production Docker container, and publishes it to **Amazon ECR**.
3. **Continuous Deployment:** Connects to an **AWS EC2 Self-Hosted Runner**, pulls the updated image from ECR, and re-launches the container seamlessly.

### Required GitHub Repository Secrets
* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_REGION`
* `AWS_ECR_LOGIN_URI`
* `ECR_REPOSITORY_NAME`

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

* **Mebratu Cheka**
* **GitHub:** [@mebre7](https://github.com/mebre7)
* **Email:** mebratucheka7@gmail.com