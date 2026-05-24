# Airline Passenger Satisfaction Analysis - Advanced Classification Models

## Overview
This Jupyter notebook presents a comprehensive comparison of multiple machine learning models for predicting airline passenger satisfaction. The analysis includes logistic regression, decision trees, and random forest models, with hyperparameter tuning to optimize performance. The goal is to identify the most effective model for classifying customer satisfaction based on various service quality metrics and travel characteristics.

## Dataset
The analysis uses airline passenger satisfaction survey data (`Invistico_Airline.csv`) containing information from over 129,000 passengers across multiple flight parameters and service dimensions.

### Key Features (22 total)
**Target Variable**:
- **satisfaction**: satisfied/dissatisfied

**Demographic & Travel Information**:
- **Customer Type**: Loyal customer vs. disloyal customer
- **Age**: Passenger age
- **Type of Travel**: Business travel vs. Personal travel
- **Class**: Travel class (Business, Eco, Eco Plus)
- **Flight Distance**: Distance of the flight

**Service Ratings** (0-5 scale):
- Seat comfort
- Departure/Arrival time convenience
- Food and drink
- Gate location
- Inflight wifi service
- Inflight entertainment
- Online support
- Ease of Online booking
- On-board service
- Leg room service
- Baggage handling
- Checkin service
- Cleanliness
- Online boarding

**Delay Metrics**:
- Departure Delay (minutes)
- Arrival Delay (minutes)

### Dataset Statistics
- **Total observations**: 129,880 rows
- **Missing values**: 393 missing values in Arrival Delay (removed)
- **Final dataset**: 129,487 complete observations
- **Class distribution**: Balanced (satisfied: 71,087, dissatisfied: 58,793)

## Data Preprocessing

### Data Cleaning
1. **Missing Value Handling**: Dropped rows with missing Arrival Delay data
2. **Data Type Conversion**: Converted 'Inflight entertainment' to float
3. **Categorical Encoding**:
   - Target encoding: satisfaction mapped to binary (satisfied=1, dissatisfied=0)
   - One-hot encoding for: Customer Type, Type of Travel, Class
4. **Data Splitting**:
   - Logistic Regression: 70% train, 30% test
   - Decision Tree & Random Forest: 75% train, 25% test

## Model Analysis

### 1. Logistic Regression (Baseline Model)

**Features Used**: Inflight entertainment (single predictor)

**Model Parameters**:
- Intercept: -3.1936
- Coefficient: 0.9975

**Performance Metrics**:
| Metric | Score |
|--------|-------|
| Accuracy | 80.15% |
| Precision | 81.61% |
| Recall | 82.15% |
| F1 Score | 81.88% |

**Interpretation**: Simple model showing positive relationship between inflight entertainment and satisfaction, but limited by single-feature approach.

### 2. Decision Tree Model

**Data Preparation**:
- Converted Class to numeric (Business=3, Eco Plus=2, Eco=1)
- One-hot encoded categorical variables
- Removed satisfaction from features

**Model Parameters** (after hyperparameter tuning):
- max_depth: 15
- min_samples_leaf: 3
- random_state: 0

**Hyperparameter Tuning Grid**:
- max_depth: [1-20, 30, 40, 50]
- min_samples_leaf: [2-10, 15, 20, 50]
- Scoring metrics: f1, recall, accuracy, precision
- Cross-validation: 5-fold

**Performance Metrics**:
| Metric | Score |
|--------|-------|
| Accuracy | 93.54% |
| Precision | 94.29% |
| Recall | 93.90% |
| F1 Score | 94.09% |

**Key Decision Tree Features** (top splits):
1. Inflight entertainment (≤ 3.5)
2. Seat comfort (≤ 3.5)
3. Ease of Online booking (≤ 3.5)
4. Customer Type

### 3. Random Forest Model

**Data Preparation**:
- One-hot encoded all categorical variables (Customer Type, Type of Travel, Class)
- 26 total features after encoding

**Model Parameters** (optimal after grid search):
- n_estimators: 50
- max_depth: 50
- min_samples_leaf: 1
- min_samples_split: 0.001
- max_features: "sqrt"
- max_samples: 0.9
- random_state: 0

**Training Strategy**:
- Training set: 75% of data
- Validation set: 25% of training data (custom split for tuning)
- Test set: 25% of original data
- Grid search with 32 parameter combinations
- 1 fold (custom validation split)

**Performance Metrics**:
| Metric | Score |
|--------|-------|
| Accuracy | 94.25% |
| Precision | 95.01% |
| Recall | 94.45% |
| F1 Score | 94.73% |

## Model Comparison

| Model | F1 Score | Recall | Precision | Accuracy |
|-------|----------|--------|-----------|----------|
| Tuned Random Forest | **0.947** | **0.945** | **0.950** | **0.942** |
| Tuned Decision Tree | 0.945 | 0.936 | 0.955 | 0.941 |
| Tuned Logistic Regression | 0.819 | 0.822 | 0.816 | 0.802 |

### Key Insights:
1. **Random Forest** achieves the best overall performance, with highest F1 score (0.947) and balanced precision/recall
2. **Decision Tree** performs nearly as well but slightly lower recall
3. **Logistic Regression** significantly underperforms due to using only one feature
4. All tree-based models dramatically outperform the simple logistic regression

## Visualizations

### Decision Tree Visualization
- Tree structure showing top 2 levels
- Key split variables identified
- Gini impurity and sample distribution at each node

### Confusion Matrices
- Decision Tree: 93.5% accuracy on test set
- Logistic Regression: 80.2% accuracy with single feature

## Technical Implementation

### Libraries Used
```python
# Data manipulation
import numpy as np
import pandas as pd

# Preprocessing & Modeling
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
```

### Key Code Sections

**Decision Tree Hyperparameter Tuning**:
```python
tree_para = {
    'max_depth': [1,2,3,...,50],
    'min_samples_leaf': [2,3,4,...,50]
}
tuned_decision_tree = DecisionTreeClassifier(random_state=0)
clf = GridSearchCV(tuned_decision_tree, tree_para, 
                   scoring=scoring, cv=5, refit="f1")
clf.fit(X_train_dtree, y_train_dtree)
```

**Random Forest Grid Search**:
```python
cv_params = {
    'n_estimators': [50,100],
    'max_depth': [10,50],
    'min_samples_leaf': [0.5,1],
    'min_samples_split': [0.001,0.01],
    'max_features': ["sqrt"],
    'max_samples': [.5,.9]
}
custom_split = PredefinedSplit(split_index)
rf = RandomForestClassifier(random_state=0)
rf_val = GridSearchCV(rf, cv_params, cv=custom_split, refit='f1')
```

## Business Insights

### Key Drivers of Satisfaction
1. **Inflight Entertainment** - Primary driver (top split in decision tree)
2. **Seat Comfort** - Secondary important factor
3. **Ease of Online Booking** - Significant predictor
4. **Customer Type** - Loyal vs. disloyal customers have different satisfaction patterns

### Model Performance Implications
- **Random Forest** provides the most reliable predictions (94.7% F1 score)
- Ensemble method reduces overfitting compared to single decision tree
- Tree-based models effectively capture non-linear relationships in service ratings

### Operational Recommendations
1. **Focus on Inflight Entertainment**: Highest impact on satisfaction
2. **Improve Seat Comfort**: Second most important factor
3. **Streamline Online Booking**: User experience matters significantly
4. **Segment by Customer Type**: Different strategies for loyal vs. disloyal customers

## Model Limitations
- Data represents single airline only
- No temporal information for trend analysis
- Self-reported survey data may have bias
- Delay data has 393 missing values (removed)

## Future Improvements
1. **Feature Engineering**:
   - Create interaction terms between service categories
   - Develop composite service quality scores
   - Engineer delay-related features (total delay, on-time flag)

2. **Advanced Modeling**:
   - Implement XGBoost/LightGBM for comparison
   - Test neural network architectures
   - Apply feature selection to reduce dimensionality

3. **Hyperparameter Optimization**:
   - Expand grid search for Random Forest
   - Implement Bayesian optimization
   - Test different validation strategies

4. **Business Applications**:
   - Deploy best model for real-time prediction
   - Create passenger satisfaction dashboard
   - Segment-based satisfaction prediction

## How to Run

### Prerequisites
- Python 3.x
- Required packages: numpy, pandas, scikit-learn, matplotlib, seaborn, pickle

### Installation
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

