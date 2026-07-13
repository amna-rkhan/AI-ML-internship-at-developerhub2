# DevelopersHub AI/ML Internship - Phase 2 Tasks

This repository contains my submissions for Phase 2 of the AI/ML Engineering Internship at DevelopersHub Corporation.

---

## Task 1: News Topic Classifier Using BERT

*   **Objective:** Fine-tune a transformer model to classify headlines into four categories (World, Sports, Business, Sci/Tech).
*   **Dataset:** [AG News Dataset](https://huggingface.co/datasets/fancyzhx/ag_news) via Hugging Face.
*   **Approach:** 
    *   Used `distilbert-base-uncased` for fast and lightweight training.
    *   Tokenized inputs to a max length of 128.
    *   Fine-tuned the model for 2 epochs using Hugging Face `Trainer`.
    *   Deployed an interactive web UI using **Gradio**.
*   **Results:** Obtained ~90% accuracy and F1-score on the evaluation split.

---

## Task 2: End-to-End ML Pipeline (Scikit-learn)

*   **Objective:** Build a robust, production-ready pipeline to predict customer churn.
*   **Dataset:** Kaggle Telco Customer Churn Dataset.
*   **Approach:**
    *   Built a unified `scikit-learn` Pipeline.
    *   Imputed and standardized numerical features using `StandardScaler`.
    *   Encoded categorical features using `OneHotEncoder`.
    *   Tuned Logistic Regression and Random Forest models with `GridSearchCV`.
    *   Exported the complete best model to `churn_pipeline.joblib`.
*   **Results:** Generated optimized precision and recall metrics for high-risk customer churn.

---

## Task 3: Multimodal Housing Price Prediction

*   **Objective:** Predict continuous house prices using both structured attributes and house images.
*   **Dataset:** [Ahmed Elgazzar's Houses Dataset](https://github.com/emanhamed/Houses-dataset).
*   **Approach:**
    *   **Tabular Branch (MLP):** Processes features like area, bedrooms, bathrooms, and geographical zip code.
    *   **Image Branch (CNN):** Extracts visual features from frontal house photos using 2D convolutions.
    *   **Fusion:** Concatenated both vector representations into a final dense regressor layer.
*   **Results:** Measured performance using MAE and RMSE, showing that combining image features with tabular data drastically improves prediction accuracy.
