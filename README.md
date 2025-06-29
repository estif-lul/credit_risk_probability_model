# Credit Risk Probability Model

This project provides a machine learning-based solution for predicting the probability of credit risk. It is designed to help financial institutions assess the likelihood of default for loan applicants, enabling better risk management and decision-making.

## Table of Contents
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data](#data)
- [Modeling Approach](#modeling-approach)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)

## Credit Scoring Business Understanding

### Basel II Accord: The Need for Interpretability

The Basel II Capital Accord highlights the necessity for accurate and transparent risk quantification to maintain financial system stability. Credit scoring models must be both predictive and interpretable, allowing financial institutions to justify credit decisions to regulators. Techniques such as Weight of Evidence (WoE) and Information Value (IV) support explainability while effectively quantifying risk.

### Necessity of a Proxy Variable

In the absence of an explicit "default" indicator in the dataset, a proxy variable is constructed using Recency, Frequency, and Monetary (RFM) metrics. This binary target assumes that customers with low transaction activity and value are more likely to default. While this enables modeling without ground-truth labels, it introduces the risk of misclassification. Therefore, thorough documentation and regular recalibration of the proxy are essential.

### Model Complexity vs. Regulatory Simplicity

There is a trade-off between model interpretability and predictive performance. Logistic regression models with WoE encoding offer simplicity, transparency, and ease of audit, making them suitable for regulatory requirements. More complex models, such as gradient boosting machines (GBM), may provide higher accuracy but reduce interpretability. In regulated environments, model choice should balance explainability for audits and performance for internal risk management.

## Features

- Data preprocessing and feature engineering for credit datasets
- Multiple machine learning models (e.g., Logistic Regression, Random Forest, XGBoost)
- Model evaluation and comparison
- Probability scoring for individual applicants
- Configurable pipeline for experimentation

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/credit_risk_probability_model.git
    cd credit_risk_probability_model
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset in CSV format.
2. Edit the configuration file as needed.
3. Run the training script:
    ```bash
    python train.py --config config.yaml
    ```
4. Predict credit risk probabilities:
    ```bash
    python predict.py --input applicant_data.csv --model saved_model.pkl
    ```

## Project Structure

```
credit_risk_probability_model/
├── data/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluation.py
├── config.yaml
├── requirements.txt
├── train.py
├── predict.py
└── README.md
```

## Data

The model expects a tabular dataset with features such as:

- Applicant demographics (age, income, employment status)
- Credit history (previous defaults, credit score)
- Loan details (amount, term, purpose)

**Note:** Ensure data privacy and compliance with relevant regulations.

## Modeling Approach

- Data cleaning and imputation
- Feature selection and transformation
- Model training with cross-validation
- Hyperparameter tuning
- Probability calibration

## Evaluation Metrics

- Area Under ROC Curve (AUC-ROC)
- Precision, Recall, F1-score
- Brier Score
- Confusion Matrix

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.