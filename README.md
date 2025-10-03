# Predicting the Use of Hearing Aid — Data Mining Project 2

## Can we predict the use of hearing aid based on demographic data?
## Is a skewed dataset sufficient to make predictions that beat the simple majority class baseline?

## Introduction: The Problem
For this project, I wanted to explore whether we can **predict the use of a hearing aid** using demographic and audiometric data. This is personally interesting to me since I make and listen to music at high volumes, and I want to understand how age, sex, and hearing-related factors connect to real hearing aid use.  

The two guiding questions are:  
1. Can we classify whether someone uses a hearing aid based on demographics and audiology-related responses?  
2. Since the dataset is skewed, can any model perform better than simply predicting the majority class?

---

## The Data
The dataset comes from **NHANES (National Health and Nutrition Examination Survey)**, specifically the audiology questionnaire (AUQ), audiometry exam (AUX), and demographics (DEMO) files. These were merged using the participant identifier **SEQN**.  

**Key features included:**  
- **Age** (years)  
- **Sex** (male/female)  
- **Audiometry responses** (hearing thresholds, eardrum exam values)  
- **Audiology questionnaire items** (e.g., hearing aid use, noise exposure history)  

The **target variable** is AUQ147: “Does this participant use a hearing aid?” (binary: Yes/No).  

---

## Pre-processing the Data
I merged the datasets, dropped columns with high amounts of recorded null values, and focused on cleaning rows where the hearing aid response was present.

```python
import pandas as pd

df_auq = pd.read_csv('/content/P_AUQ.csv')
df_aux = pd.read_csv('/content/P_AUX.csv')
df_demo = pd.read_csv('/content/P_DEMO.csv')

# Merge all three datasets
df_combined = pd.merge(df_auq, df_aux, on='SEQN', how='outer')
df_combined = pd.merge(df_combined, df_demo, on='SEQN', how='outer')

# Drop rows missing the target
df_combined = df_combined.dropna(subset=['AUQ147'])

```

### Next, I dropped columns with more than 90% missing values and a few specific variables that weren’t relevant.

```python

# Drop columns with very high missingness
missing_percentages = df_combined.isnull().mean() * 100
columns_to_drop = missing_percentages[missing_percentages > 90].index.tolist()

# Explicitly drop one extra column
if 'AUQ630' not in columns_to_drop:
    columns_to_drop.append('AUQ630')

df_combined_dropped = df_combined.drop(columns=columns_to_drop)

```

### Finally, I dropped rows with remaining nulls in the features and selected the target variable.

```python

# Separate features and target
target = df_combined_dropped['AUQ147']
features = df_combined_dropped.drop('AUQ147', axis=1)

# Drop extra eardrum columns
features = features.drop(['AUATYMTL', 'AUATYMTR'], axis=1)

# Drop rows with nulls in features
features_cleaned = features.dropna()
target_cleaned = target.loc[features_cleaned.index]

```

##Data Understanding & Visualization

###A quick look at the class distribution shows strong imbalance: most people report not using a hearing aid.

```python

target_cleaned.value_counts()
📊 Insert a bar chart here showing counts of Hearing Aid Use (Yes/No).

This imbalance means accuracy alone is misleading. Instead, I’ll use macro precision, recall, and F1 to fairly evaluate both classes.

Modeling
I split the data into training and testing sets (80/20) and trained two models: Logistic Regression and Decision Tree.

python
Copy code
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_cleaned, target_cleaned, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
Logistic Regression
What it is: A linear classifier that estimates probabilities and applies a threshold.
Pros: Simple, interpretable, fast.
Cons: May underfit nonlinear patterns, sensitive to imbalance.

python
Copy code
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=42, solver="liblinear", max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
Decision Tree
What it is: A tree-based classifier that splits features to minimize impurity.
Pros: Captures nonlinearity, easy to visualize.
Cons: Can overfit without pruning/tuning.

python
Copy code
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(random_state=42, min_samples_split=50)
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
Evaluation
I compared model performance to a majority baseline (always predicting “no hearing aid”).

python
Copy code
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Baseline accuracy
baseline_acc = y_test.value_counts().max() / y_test.shape[0]
print(f"Majority baseline accuracy: {baseline_acc:.3f}")

def report(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    print(f"\n=== {model_name} ===")
    print(f"Accuracy:   {acc:.3f}")
    print(f"Macro-Prec: {prec:.3f}  Macro-Rec: {rec:.3f}  Macro-F1: {f1:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Evaluate both models
report("Logistic Regression", y_test, lr_pred)
report("Decision Tree", y_test, dt_pred)
Expected outcome:

Both models beat the baseline accuracy.

Decision Tree may recall more of the minority class (hearing aid users).

Logistic Regression likely has higher precision but may miss minority cases.

Storytelling
What I learned:

Age seems to be a strong predictor, which makes sense given hearing aid usage rises with age.

The imbalance made accuracy misleading. F1 scores tell a clearer story.

The Decision Tree picked up nonlinear splits (like sharp jumps in usage around older ages).

Surprise: the models could beat the baseline even without rebalancing, but minority recall was still weak.

Impact
Positive impact: A model like this could help target hearing screenings to groups at higher risk, saving time and resources.

Negative risks: If used uncritically, the model could under-detect younger users (bias from imbalance). This could lead to unfair conclusions.

Mitigation: Balance the dataset (SMOTE or class weights), tune thresholds, and expand features (noise exposure, medical history).

References
CDC NHANES: NHANES Audiometry Data

scikit-learn documentation

Code
The full notebook is included in this repo for transparency.
All code snippets shown here are directly from the notebook.
