
# **A Machine Learning Approach to Detecting Anomalous Banking Transactions**

## Overview

This project focuses on identifying anomalous banking transactions by analyzing transaction behavior, customer behavior, security behavior, and behavioral patterns using the Bank Transaction Dataset for Fraud Detection by Vala Khorasani. Since the dataset does not contain historical fraud labels, unsupervised anomaly detection techniques are applied to identify transactions that deviate from normal customer behavior.

## Project Workflow

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Encoding and Scaling
6. Isolation Forest
7. Local Outlier Factor (LOF)
8. Model Comparison
9. Conclusion

## Problem Statement

Financial institutions process thousands of transactions daily, making it challenging for fraud analysts to manually identify suspicious activity. In the absence of labeled fraud data, detecting unusual transaction behaviors requires methods that can identify patterns that deviate from normal customer activity.

**Business Objective:**

Develop an anomaly detection system that identifies transactions exhibiting unusual behavioral patterns, allowing fraud analysts to prioritize high-risk transactions for further investigation.

## Business Impact

The proposed anomaly detection pipeline enables financial institutions to prioritize transactions exhibiting unusual behavioral patterns for manual review. By combining exploratory data analysis, feature engineering, and multiple anomaly detection algorithms, the system provides broader detection coverage by capturing different types of anomalous transaction behavior without requiring historical fraud labels.

## Dataset

The dataset used in this project is the **Bank Transaction Dataset for Fraud Detection** by Vala Khorasani, available on Kaggle.

**Dataset Source:** [Bank Transaction Dataset for Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)

## Dataset Summary

- Transactions: **2,512**
- Customer Accounts: **495**
- Missing Values: **None**
- Duplicate Records: **None**

### Key Features

The following feature descriptions are adapted from the dataset documentation provided:

- **TransactionID**: Unique alphanumeric identifier for each transaction.
- **AccountID**: Unique identifier for each account, with multiple transactions per account.
- **TransactionAmount**: Monetary value of each transaction, ranging from small everyday expenses to larger purchases.
- **TransactionDate**: Timestamp of each transaction, capturing date and time.
- **TransactionType**: Categorical field indicating 'Credit' or 'Debit' transactions.
- **Location**: Geographic location of the transaction, represented by U.S. city names.
- **DeviceID**: Alphanumeric identifier for devices used to perform the transaction.
- **IP Address**: IPv4 address associated with the transaction, with occasional changes for some accounts.
- **MerchantID**: Unique identifier for merchants, showing preferred and outlier merchants for each account.
- **AccountBalance**: Balance in the account post-transaction, with logical correlations based on transaction type and amount.
- **PreviousTransactionDate**: Timestamp of the last transaction for the account, aiding in calculating transaction frequency.
- **Channel**: Channel through which the transaction was performed (e.g., Online, ATM, Branch).
- **CustomerAge**: Age of the account holder, with logical groupings based on occupation.
- **CustomerOccupation**: Occupation of the account holder (e.g., Doctor, Engineer, Student, Retired), reflecting income patterns.
- **TransactionDuration**: Duration of the transaction in seconds, varying by transaction type.
- **LoginAttempts**: Number of login attempts before the transaction, with higher values indicating potential anomalies.

### Data Cleaning

- Checked for missing values and duplicate records
- Converted date columns to datetime format

### Feature Encoding and Scaling

- One-hot encoded categorical variables
- Applied RobustScaler to numerical features


### Feature Engineering

Several behavioral features were engineered from the original dataset, including:

- Device change indicator
- IP address change indicator
- Transaction hour
- Day of week
- Weekend transaction flag
- Time since previous transaction
- Amount-to-balance ratio
- High-value transaction flag

## Exploratory Data Analysis (EDA)

EDA was performed to understand behavioral patterns and identify factors that may indicate unusual transaction activity.

The exploratory data analysis examined four key areas:
- Transaction behavior patterns
- Customer account activity
- Device and IP address changes
- Unusual transaction amounts and frequencies

Key findings:

The exploratory data analysis revealed that customer transactions generally exhibit consistent behavioral patterns across transaction, customer, security, and account activity. Most transactions were debit transactions with moderate transaction amounts, while transaction channels (Online, ATM, and Branch) were used at relatively similar frequencies. Although a small number of high-value transactions were identified as statistical outliers, the majority of transactions fell within the typical spending range.

Customer behavior showed that most accounts performed approximately five transactions, accessed their accounts from multiple locations and IP addresses, and utilized around five different devices on average. Frequent changes in devices and IP addresses were observed for a small subset of accounts, indicating behavioral patterns that may warrant closer examination when combined with other risk indicators.

From a security perspective, login attempts remained consistently low across all transaction channels, suggesting that unsuccessful login activity was generally uncommon. However, locations differed in their average transaction amounts, with several cities exhibiting relatively higher-value transactions that may require additional monitoring.

Overall, the analysis established a baseline of normal customer behavior while identifying behavioral characteristics that distinguish potentially anomalous transactions. These insights guided the feature engineering process and provided a strong foundation for applying unsupervised anomaly detection using Isolation Forest and Local Outlier Factor (LOF).

## Methodology

Two unsupervised machine learning algorithms were implemented to identify anomalous banking transactions.

| Model | Purpose | Key Parameters |
|--------|---------|----------------|
| **Isolation Forest** | Identifies anomalies by randomly isolating observations that differ significantly from normal transaction behavior. | `contamination = 0.05`<br>`n_estimators = 100` |
| **Local Outlier Factor (LOF)** | Detects anomalies by measuring the local density of a transaction relative to its neighboring observations. | `contamination = 0.05`<br>`n_neighbors = 20` |

## Results

The models were evaluated by comparing their anomaly predictions to identify both shared and model-specific anomaly detections.

### Key Results

- Isolation Forest detected **126** potential anomalies.
- Local Outlier Factor detected **126** potential anomalies.
- **43** transactions were identified by both models.
- **83** transactions were uniquely detected by Isolation Forest.
- **83** transactions were uniquely detected by Local Outlier Factor.
- The overall agreement between the two models was **93.39%**.

### Key observations:
- Transactions detected by both models represent the highest-confidence anomalies because they were independently identified by two different algorithms.
- Model-specific anomalies highlight that different algorithms capture different types of unusual behavioral patterns.
- Combining multiple anomaly detection approaches provides broader anomaly coverage.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

### Machine Learning
- Isolation Forest
- Local Outlier Factor (LOF)

## Future Improvements

- Evaluate additional anomaly detection algorithms such as One-Class SVM and DBSCAN.
- Incorporate labeled fraud data to build supervised classification models.
- Deploy the anomaly detection pipeline as a real-time monitoring system.


## Repository Structure

```text
financial-transaction-anomaly-detection/
│
├── README.md                    # Project documentation 
├── bank_transactions_data_2.csv # Dataset
├── ml_anomaly_detection.ipynb   # End-to-end project notebook
└── requirements.txt             # Python dependencies
```