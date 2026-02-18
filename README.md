# Give Me Some Credit Project
![Image](https://github.com/user-attachments/assets/38d6754a-4447-4c52-85ec-dab9dfd6cafc)
This is an end-to-end machine learning project using the classic Kaggle "Give Me Some Credit" dataset.  
The goal was to predict the probability of serious delinquency (default) in the next two years, using real-world, highly imbalanced tabular data.

I chose this dataset because I wanted hands-on experience with typical challenges in credit risk and fintech problems:
- severe class imbalance (~6–7% positive class)
- noisy real-world data (missing values, sentinels, outliers)
- need for both ranking power and probability calibration

## Project Overview

The project follows a complete ML pipeline:

1. **Data Preprocessing**
<img width="989" height="590" alt="Image" src="https://github.com/user-attachments/assets/2bc78bdc-7b16-4242-bada-99ddb37ccfe9" />
   - Handled sentinel values (96/98/99) as privacy/error flags → replaced with NaN  
   - Zero-income cases → flagged + imputed with non-zero median  
   - Missing dependents → conservative fill with 0  
   - Heavy-tailed features → log1p transformation + robust capping (0 to 99th percentile)  
   - PCA on correlated delinquency variables to reduce multicollinearity  

2. **Exploratory Data Analysis (EDA)**  
   - Distributions, correlations, target interactions  
   - Key insight: serious delinquency (especially 90+ days) and high unsecured utilization dominate risk signals  

3. **Modeling & Evaluation**  
<img width="806" height="700" alt="Image" src="https://github.com/user-attachments/assets/bce0dde2-9c05-4583-b68b-205ac8ad513a" />
   - Baseline: Logistic Regression (class-weighted vs SMOTE)  
   - Advanced: LightGBM & XGBoost (weighted vs SMOTE)  
   - Metrics: PR-AUC (main focus), F1-score, Brier score (calibration), probability histograms  
   - Interpretability: coefficients (logistic) + SHAP (boosting)  
   - Final submission: Kaggle leaderboard evaluation  

## Key Results

- **Best model**: LightGBM with class weighting (`is_unbalance=True`)  
  - Validation PR-AUC: **0.3997**  
  - Kaggle Private AUC: **0.8582** (competitive for this dataset — top ~15–20% historically)  

- **Core insight** from all top models (consistent across logistic, LightGBM, XGBoost):  
  Default risk is mainly a **behavioral and payment-discipline problem**:  
  - **Serious past delinquency** (especially 90+ days late) is the biggest red flag  
  - **Extreme current unsecured credit utilization** (maxed-out cards/lines) acts as a strong tipping point  
  - **Higher age** is consistently protective  

- **Imbalance handling lesson**:  
  Class weighting preserved real risk signals and generalized better (higher Kaggle AUC).  
  SMOTE sometimes improved calibration locally (lower Brier score) but hurt ranking on unseen data.

## Repository Structure
GiveMeSomeCredit/
├── data_preprocessing.ipynb       # Cleaning, imputation, capping, PCA
├── EDA.ipynb                      # Distributions, correlations, insights
├── modeling.ipynb                 # Logistic baseline + LightGBM/XGBoost + SHAP + Kaggle submissions
├── requirements.txt               # Dependencies
├── images/                        # Plots, SHAP summaries, probability histograms
└── README.md                      # This file

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Seungukeeee/GiveMeSomeCredit.git
   cd GiveMeSomeCredit
2. Install dependencies:
   pip install -r requirements.txt
3. Download the files(you can download cs-training and cs-test)
4. Run notebooks in order:
  data_preprocessing.ipynb → EDA.ipynb → modeling.ipynb

## What I Learned

Careful preprocessing (sentinels, capping, PCA) is crucial — small early decisions greatly affect final performance.
Class weighting is usually more reliable than SMOTE in highly imbalanced real-world data.
Different models can "see" the same problem differently (linear vs non-linear), but top ones converged on similar core signals.
Validation metrics can be misleading — external test sets (like Kaggle private LB) are the true judge.

This project was a very meaningful experience for me.
It helped me understand how to handle messy, imbalanced tabular data and turn technical steps into a clear story about real-world credit risk.
I’m now excited to explore more datasets in similar domains (fraud detection, customer churn, insurance claims) and try advanced techniques like ensembles, better calibration, and custom business thresholds.
Feel free to ⭐ or fork the repo if you find it useful!
