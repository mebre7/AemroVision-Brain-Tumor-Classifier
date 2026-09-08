
![Logo](static/images/light.png)
# AemroVision: Automated Brain Tumor MRI Classification

An end-to-end production-grade computer vision system designed to classify brain MRI scans into four distinct categories: **Glioma, Meningioma, Pituitary, and No Tumor**. This project emphasizes modular software engineering, clean configuration-driven architecture, and robust deployment workflows.

## 🚀 Key Features

* **Modular Pipeline Architecture:** Separated components for data ingestion, validation, model training, and evaluation.
* **Transfer Learning Backbone:** Utilizes pre-trained convolutional neural networks optimized for high-accuracy medical image classification.
* **Configuration-Driven:** Centralized parameter management via YAML configuration files.
* **REST API Integration:** Built-in utilities for handling base64-encoded image payloads for seamless web and client application integration.

## 📂 Project Structure

```text
├── config/                  # Configuration YAML files
├── research/                # Jupyter notebooks for prototyping and experimentation
├── src/
│   └── cnnClassifier/       # Core source package
│       ├── components/      # Pipeline stages (Ingestion, Training, Evaluation)
│       ├── config/          # Configuration management logic
│       ├── entity/          # Data entity schemas
│       ├── pipeline/        # Execution pipeline scripts
│       └── utils/           # Utility functions (image encoding/decoding, file I/O)
├── templates/               # Frontend web interface templates
├── main.py                  # Main orchestration script
├── requirements.txt         # Project dependencies
├── setup.py                 # Package installation configuration
└── dvc.yaml                 # Data Version Control pipeline configuration
```