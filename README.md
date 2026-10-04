# mathsscorepredict
# O-Level Mathematics Score Prediction

## 1. Project Overview

This project develops a machine learning regression solution to predict students' O-Level Mathematics examination scores.

The objective is to help the school identify students who may be weaker in Mathematics **before the examination**, so that targeted academic support can be provided.

The project evaluates three regression models:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regression

The models are compared using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The best-performing model is selected based on validation performance and then evaluated on an unseen test dataset.

---

## 2. Dataset

The project uses:

```text
data/regression_bonus_practice_data.csv
```

The target variable is:

```text
final_test
```

`final_test` represents the student's Mathematics examination score.

The dataset contains student-related characteristics including:

- Number of siblings
- Direct admission status
- CCA
- Learning style
- Gender
- Tuition
- Age
- Study hours per week
- Attendance rate
- Sleep time
- Wake time
- Mode of transport
- Bag colour
- Male/female student counts

---

## 3. Project Structure

```text
regression/
│
├── README.md
├── requirements.txt
├── data/
│   └── regression_bonus_practice_data.csv
│
├── eda.ipynb
│
└── src/
    ├── config.yaml
    ├── data_preparation.py
    ├── model_training.py
    └── main.py
```

### File descriptions

| File                                      | Purpose                                             |
| ----------------------------------------- | --------------------------------------------------- |
| `README.md`                               | Project documentation                               |
| `requirements.txt`                        | Python dependencies                                 |
| `data/regression_bonus_practice_data.csv` | Input dataset                                       |
| `eda.ipynb`                               | Exploratory Data Analysis                           |
| `src/config.yaml`                         | Configuration and model parameters                  |
| `src/data_preparation.py`                 | Data loading, feature engineering and preprocessing |
| `src/model_training.py`                   | Model training, tuning and evaluation               |
| `src/main.py`                             | Main application/orchestration script               |

---

## 4. Exploratory Data Analysis

The `eda.ipynb` notebook performs:

- Dataset shape and structure analysis
- Data type inspection
- Duplicate checking
- Missing-value analysis
- Target-variable distribution
- Correlation analysis
- Numerical feature analysis
- Categorical feature analysis
- Outlier analysis
- Sleep-duration feature engineering
- Feature-selection justification

### Important feature engineering

The dataset contains:

```text
sleep_time
wake_time
```

These are transformed into:

```text
sleep_duration_hours
```

For example:

```text
Sleep: 22:00
Wake : 06:00
Duration: 8 hours
```

This numerical feature is more useful for regression than treating the original clock-time values as ordinary categories.

---

## 5. Data Preparation

The `DataPreparation` class performs the following steps.

### 5.1 Load data

The CSV file is loaded using Pandas.

### 5.2 Remove identifier fields

The following fields are excluded:

```text
index
student_id
```

These fields identify records/students but do not represent meaningful academic characteristics.

### 5.3 Feature engineering

The following feature is created:

```text
sleep_duration_hours
```

### 5.4 Missing values

Numerical features:

```text
Median imputation
```

Categorical features:

```text
Most-frequent imputation
```

### 5.5 Numerical preprocessing

Numerical variables are standardised using:

```text
StandardScaler
```

### 5.6 Categorical preprocessing

Categorical variables are transformed using:

```text
OneHotEncoder(handle_unknown="ignore")
```

All preprocessing is implemented inside the machine-learning pipeline to avoid data leakage.

---

## 6. Features Used

### Numerical features

```text
number_of_siblings
n_male
n_female
age
hours_per_week
attendance_rate
sleep_duration_hours
```

### Categorical features

```text
direct_admission
CCA
learning_style
gender
tuition
mode_of_transport
bag_color
```

### Target

```text
final_test
```

---

## 7. Machine Learning Models

### 7.1 Linear Regression

Linear Regression is used as the baseline model.

It assumes that the target score can be approximately represented as a linear combination of the input features.

Advantages:

- Simple
- Fast
- Easy to interpret
- Useful as a baseline

---

### 7.2 Ridge Regression

Ridge Regression extends Linear Regression by adding L2 regularisation.

It is useful when:

- There are many encoded categorical features
- Features may be correlated
- The model needs to control coefficient magnitude

The regularisation parameter `alpha` is tuned using GridSearchCV.

---

### 7.3 Random Forest Regression

Random Forest is an ensemble of decision trees.

It is suitable because student performance may depend on non-linear relationships and interactions between features.

For example:

```text
attendance + tuition + study hours
```

may have a combined effect that is not purely linear.

Random Forest can capture these relationships without requiring them to be manually specified.

---

## 8. Model Training Strategy

The dataset is divided into:

```text
80% Training
10% Validation
10% Test
```

The training data is used for model fitting and hyperparameter tuning.

The validation set is used to compare the models.

The test set remains unseen until the final evaluation.

This helps provide a fair estimate of how the selected model performs on new students.

---

## 9. Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation is used.

### Linear Regression

```yaml
regressor__fit_intercept:
  - true
  - false
```

### Ridge Regression

```yaml
regressor__alpha:
  - 0.1
  - 1
  - 10
  - 100
  - 1000

regressor__fit_intercept:
  - true
  - false
```

### Random Forest

The configuration contains values for:

```yaml
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

The final hyperparameters are selected using cross-validation.

---

## 10. Model Evaluation

Three metrics are used.

### Mean Absolute Error — MAE

MAE measures the average absolute difference between the predicted and actual score.

```text
Lower is better
```

For example, an MAE of 5.5 means the prediction is approximately 5.5 marks away from the actual score on average.

---

### Root Mean Squared Error — RMSE

RMSE is the square root of the mean squared prediction error.

```text
Lower is better
```

RMSE gives greater importance to larger prediction errors.

This is useful for the school because large prediction errors could cause students needing support to be missed.

---

### R² Score

R² measures how much of the variation in examination scores is explained by the model.

```text
Higher is better
```

An R² of 0.70 means the model explains approximately 70% of the variation in the target scores.

---

## 11. Overall Model Comparison

The models are compared using the validation dataset.

The expected output has the following format:

```text
MODEL COMPARISON - VALIDATION SET

Model                 CV_R2    Validation_MAE    Validation_RMSE    Validation_R2
Linear Regression     ...
Ridge                 ...
Random Forest         ...
```

The primary selection criterion is:

```text
Lowest Validation MAE
```

RMSE and R² are used as supporting evaluation metrics.

This approach is appropriate because the objective is to predict students' actual scores as accurately as possible.

---

## 12. Final Model Evaluation

After selecting the best model:

1. The selected model is retrained using the training + validation datasets.
2. The model is evaluated once using the unseen test dataset.
3. Final MAE, RMSE and R² are reported.

Example output:

```text
FINAL TEST PERFORMANCE

Test MAE : ...
Test RMSE: ...
Test R2  : ...
```

The test dataset is not used during model selection.

---

## 13. Identifying Weaker Students

The final regression model predicts each student's expected Mathematics score.

The school can then define an appropriate academic-support threshold.

For example:

```text
Predicted score < school-defined threshold
        ↓
Potentially weaker student
        ↓
Additional academic support
```

The threshold should be determined by the school's academic criteria rather than arbitrarily selecting a value.

Possible support actions include:

- Additional Mathematics lessons
- Targeted revision
- Practice papers
- Individual teacher consultation
- Additional tuition/support sessions

The model should be treated as a **decision-support tool**, rather than the sole basis for determining a student's academic ability.

---

## 14. How to Run the Project

### Step 1 — Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run EDA

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
eda.ipynb
```

Run the notebook cells to generate the analysis and charts.

### Step 4 — Run the machine-learning pipeline

From the project root:

```bash
python src/main.py
```

---

## 15. Expected Workflow

```text
                  O-Level Dataset
                         |
                         v
                Data Preparation
                         |
             +-----------+-----------+
             |                       |
             v                       v
      Feature Engineering       Missing Values
             |                       |
             +-----------+-----------+
                         |
                         v
                 Preprocessing
                         |
                         v
              Train / Validation / Test
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
     Linear           Ridge       Random Forest
   Regression       Regression      Regression
          |              |              |
          +--------------+--------------+
                         |
                         v
                  Model Comparison
                         |
                         v
                  Best Model Selected
                         |
                         v
                  Final Test Set
                         |
                         v
              Student Score Prediction
                         |
                         v
             Identify Students Needing
                   Additional Support
```

---

## 16. Conclusion

This project demonstrates an end-to-end regression workflow for predicting O-Level Mathematics scores.

The approach includes:

- Exploratory Data Analysis
- Feature engineering
- Missing-value handling
- Categorical encoding
- Feature scaling
- Multiple regression models
- Hyperparameter tuning
- Cross-validation
- Model comparison
- Final test evaluation

The final model should be selected based on empirical performance rather than simply choosing the most complex algorithm.

The selected model can support the school in identifying students who may benefit from additional Mathematics support before the O-Level examination.

