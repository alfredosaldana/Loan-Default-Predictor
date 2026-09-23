# 🏦 Loan Default Predictor
## Credit Risk Modeling with the Home Credit Dataset

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

> **"In God we trust. All others must bring data."**
> — W. Edwards Deming

---

## 👋 Welcome!

This README is your **complete guide** to understanding, running, and learning from this project.
It is intentionally long and detailed. Read it carefully — it will save you hours of confusion later.

By the end of this project you will have built a real machine learning system that a bank could
use to assess whether a loan applicant is likely to repay or default. More importantly, you will
understand **why** every single step is done the way it is.

Do not just run the code. **Read the explanations. Ask questions. Break things and fix them.**
That is how you actually learn data science.

---

## 📋 Table of Contents

1. [What Is This Project?](#-what-is-this-project)
2. [The Business Problem — Why Do Banks Care?](#-the-business-problem--why-do-banks-care)
3. [What Is Machine Learning? A Plain-English Primer](#-what-is-machine-learning-a-plain-english-primer)
4. [The Dataset — Home Credit Default Risk](#-the-dataset--home-credit-default-risk)
5. [The Data Science Lifecycle](#-the-data-science-lifecycle)
6. [Key Concepts Explained](#-key-concepts-explained)
7. [The Three Models We Use](#-the-three-models-we-use)
8. [Project Structure](#-project-structure)
9. [The Five Stages — Learning Roadmap](#-the-five-stages--learning-roadmap)
10. [Setup & Installation](#-setup--installation)
11. [How to Download the Data](#-how-to-download-the-data)
12. [Running the Web Application](#-running-the-web-application)
13. [Regulatory and Ethical Considerations](#️-regulatory-and-ethical-considerations)
14. [Glossary of Terms](#-glossary-of-terms)
15. [Further Reading](#-further-reading)
16. [FAQ](#-frequently-asked-questions)

---

## 🎯 What Is This Project?

This project is a complete, end-to-end **credit risk modeling system**. We take real loan
application data from 307,511 applicants, clean it, analyze it, build a machine learning model
on top of it, evaluate the model rigorously, explain its predictions, and finally deploy it
as an interactive web application.

**In short:** given everything we know about a person (their income, age, education, credit
history, family status, and more), can we predict whether they will repay their loan or default?

This is one of the oldest and most economically important applications of data science.
Banks have been building credit scoring models since the 1950s. What has changed is the
sophistication of the algorithms and the richness of the data available.

### What you will build

By completing all five stages of this project, you will have:

- A **trained XGBoost classifier** that predicts default probability for any new applicant
- A full **exploratory data analysis** with visualizations explaining patterns in the data
- A **preprocessing pipeline** that handles missing data, encodes categories, and scales features
- **Model comparison results** across three different algorithms
- **SHAP explanations** showing why the model made each prediction
- A **Streamlit web application** where anyone can enter applicant data and get a prediction
- A complete **GitHub repository** to show in your portfolio

---

## 💼 The Business Problem — Why Do Banks Care?

### The Cost of Getting It Wrong

Imagine you work at a bank. Every day, hundreds of people apply for loans. Each application
represents a decision: approve or deny? If you approve too many risky borrowers, you lose money
when they default. If you deny too many good borrowers, you lose the interest income you would
have earned, and you damage your reputation.

Let us put numbers on this. Suppose a bank issues a loan of $10,000:

- If the borrower **repays** → the bank earns interest, perhaps $1,500 over the life of the loan
- If the borrower **defaults** → the bank loses the remaining principal, perhaps $8,000, plus
  collection costs, plus legal fees

A single default can erase the profit from five or six successful loans. Getting this prediction
right is therefore not an academic exercise — it is directly tied to the financial health of the
institution and, by extension, to the savings of millions of depositors.

### The Scale Problem

A large bank might receive tens of thousands of loan applications every month. No team of human
analysts could review each one in depth. Machine learning allows banks to:

1. **Score every application automatically** in milliseconds
2. **Be consistent** — the model applies the same logic to every applicant, every time
3. **Identify non-obvious patterns** that humans would miss across hundreds of variables

### The Regulatory Problem

Lending is one of the most regulated industries in the world. In the United States, the
**Equal Credit Opportunity Act (ECOA)** and **Regulation B** prohibit discrimination based on
race, color, religion, national origin, sex, marital status, or age.

This creates a tension: a powerful black-box model might be highly accurate but might also
discriminate against protected groups in ways that are invisible. Regulators require banks to
be able to **explain** their credit decisions. That is why model interpretability — the ability
to say *why* a prediction was made — is not optional. It is a legal requirement.

This is one of the main reasons we use **SHAP values** in this project: they allow us to
produce a human-readable explanation for every single prediction.

---

## 🤖 What Is Machine Learning? A Plain-English Primer

If you are new to machine learning, this section is for you. If you are already comfortable
with the concept, feel free to skip ahead.

### The Traditional Programming Approach

In traditional programming, you write explicit rules:

```
IF income > 50000 AND credit_score > 700 AND age > 25 THEN approve
ELSE deny
```

The problem is that reality is far more complex. Which rule wins if income is high but credit
score is low? What about someone who is young but has a perfect payment history? Writing all
these rules by hand is impossible at scale.

### The Machine Learning Approach

Instead of writing rules, you show the computer thousands of examples:

```
"This person had income=45000, score=0.68, age=32 → they REPAID"
"This person had income=70000, score=0.39, age=24 → they DEFAULTED"
"This person had income=28000, score=0.72, age=55 → they REPAID"
... (307,511 more examples)
```

The algorithm **learns the rules itself** by finding patterns in the data. It discovers that
certain combinations of features are associated with default, even if no human explicitly
programmed that relationship.

### An Analogy: Learning to Recognize Cats

Imagine teaching a child what a cat is. You do not write a formal definition. Instead, you
show them thousands of pictures: "This is a cat. This is also a cat. This is NOT a cat — that's
a dog." After enough examples, the child can recognize cats they have never seen before.

Machine learning works the same way. We show the algorithm thousands of loan outcomes,
and it learns to recognize which patterns are associated with default.

### The Goal: Generalization

Here is the most important concept in machine learning: we do not want a model that memorizes
the training data. We want a model that **generalizes** — that performs well on new applicants
it has never seen before.

This is why we always split our data into a **training set** and a **test set**, and we evaluate
the model only on the test set. The test set simulates future, unseen applicants.

---

## 📊 The Dataset — Home Credit Default Risk

### Origin and Context

This dataset comes from a real company: **Home Credit Group**, an international consumer
finance provider operating in 9 countries. They serve customers who have limited or no credit
history — people who are often underserved by traditional banks.

Home Credit released this dataset as part of a public Kaggle competition in 2018.
Over 7,000 teams competed to build the best default prediction model. The competition
attracted professional data scientists from around the world, meaning this is a dataset
with real-world complexity and genuine stakes.

### The Files

The dataset consists of multiple files, each representing a different source of information:

| File | Rows | Description |
|------|------|-------------|
| `application_train.csv` | 307,511 | Main training data — one row per application, includes TARGET |
| `application_test.csv` | 48,744 | Same format, but TARGET is hidden (Kaggle submission) |
| `bureau.csv` | 1,716,428 | Credit records from other financial institutions |
| `bureau_balance.csv` | 27,299,925 | Monthly balance history of those credits |
| `previous_application.csv` | 1,670,214 | Previous loan applications at Home Credit |
| `installments_payments.csv` | 13,605,401 | Payment history for previous loans |

In this project we primarily use `application_train.csv` to keep things manageable and to
focus on core concepts. In a real production project you would join all these tables together.

### The Target Variable

The column we want to predict is called `TARGET`:

- `TARGET = 0` → the client **repaid** the loan on time
- `TARGET = 1` → the client **defaulted** — they failed to meet repayment obligations

In the training set, about **91.9% of clients repaid** and only **8.1% defaulted**.
This is a critically important fact that shapes everything we do in this project.

### Key Features

The file has 122 columns. Here are the most important ones:

**External Credit Scores**

| Column | Description | Why It Matters |
|--------|-------------|----------------|
| `EXT_SOURCE_1` | Credit score from external source 1 | Independent creditworthiness rating |
| `EXT_SOURCE_2` | Credit score from external source 2 | Usually the single most predictive feature |
| `EXT_SOURCE_3` | Credit score from external source 3 | Additional independent signal |

These scores come from credit bureaus — independent agencies that track people's credit behavior
across all financial institutions. A low score means the person has a history of late payments
or defaults with other lenders.

**Financial Information**

| Column | Description | Why It Matters |
|--------|-------------|----------------|
| `AMT_INCOME_TOTAL` | Annual income | Higher income = more capacity to repay |
| `AMT_CREDIT` | Loan amount requested | Larger loans relative to income = higher risk |
| `AMT_ANNUITY` | Annual repayment amount | Monthly burden on the borrower |
| `AMT_GOODS_PRICE` | Price of goods being financed | Ratio to credit reveals down payment |

**Demographics**

| Column | Description |
|--------|-------------|
| `DAYS_BIRTH` | Age in days, stored as a **negative** number |
| `DAYS_EMPLOYED` | Employment length in days (negative = currently employed) |
| `CODE_GENDER` | M or F |
| `NAME_EDUCATION_TYPE` | Highest education level achieved |
| `NAME_INCOME_TYPE` | Source of income (Working, Pensioner, etc.) |
| `NAME_FAMILY_STATUS` | Marital status |
| `FLAG_OWN_CAR` | Owns a car? (Y/N) |
| `FLAG_OWN_REALTY` | Owns real estate? (Y/N) |

> **Note on negative days:** The dataset stores time-based values as negative numbers
> representing days before the application date. `DAYS_BIRTH = -12000` means the person was
> born 12,000 days before applying — approximately 32.9 years old. We convert these to
> positive years during preprocessing.

### What Does "Default" Actually Mean?

In banking, **default** is a formal legal term. A borrower is in default when they have failed
to make required payments for a defined period — typically 90+ days past due.

At that point, the bank may send the account to collections, report the default to credit bureaus
(damaging the borrower's credit score), pursue legal action, or write off the loan as a loss.

In the Home Credit dataset, `TARGET = 1` means the client experienced "payment difficulties" —
they were late on payments by more than X days on at least one of the first Y installments.

---

## 🔄 The Data Science Lifecycle

This project follows a standard data science workflow. Understanding this lifecycle is as
important as understanding any individual technique:

```
┌───────────────────────────────────────────────────────────────────────┐
│                     DATA SCIENCE LIFECYCLE                            │
│                                                                       │
│  1. Problem        2. Data        3. EDA         4. Preprocessing     │
│   Definition      Collection     (Explore)        (Clean & Prepare)   │
│      │                │              │                   │            │
│      ▼                ▼              ▼                   ▼            │
│  "What are we     "Where does   "What patterns     "Fix missing       │
│   predicting?"     data come     exist? What        values, encode    │
│                    from?"        is unusual?"       categories"       │
│                                                                       │
│  5. Feature        6. Modeling    7. Evaluation    8. Deployment      │
│  Engineering       (Train)        (Test)           (Serve)            │
│      │                │              │                   │            │
│      ▼                ▼              ▼                   ▼            │
│  "Create new       "Fit ML        "How well does   "Make it           │
│   informative      algorithms     it work on       available to       │
│   features"        to data"       new data?"       real users"        │
└───────────────────────────────────────────────────────────────────────┘
```

| Stage Notebook | Lifecycle Steps |
|----------------|-----------------|
| Stage 1 — Setup & Data Loading | Problem Definition, Data Collection |
| Stage 2 — EDA | Exploratory Data Analysis |
| Stage 3 — Preprocessing & Features | Preprocessing, Feature Engineering |
| Stage 4 — Models & Evaluation | Modeling, Evaluation |
| Stage 5 — SHAP & Export | Evaluation (continued), Deployment |

---

## 💡 Key Concepts Explained

This section explains every major concept you will encounter in the notebooks.
Read these carefully before running the code. Return here when something is unclear.

---

### Classification vs. Regression

Machine learning problems fall into two broad categories:

**Regression** — the output is a continuous number.
- Example: "What will this house sell for?" → $345,000
- Example: "How many units will we sell?" → 12,400

**Classification** — the output is a category.
- Example: "Is this email spam?" → Spam / Not Spam
- Example: "Will this patient develop diabetes?" → Yes / No
- Example: "Will this borrower default?" → Default / Repay ← **this project**

Ours is a **binary classification** problem — binary because there are exactly two possible
outcomes. More specifically, we predict the **probability** of default. A score of 0.85 means
"85% chance of defaulting" — far more useful for a bank than a simple yes/no.

---

### Features and the Target Variable

In machine learning we use specific vocabulary:

- **Features** (also: predictors, inputs) — the columns used to make predictions.
  Examples: income, age, education, credit score.

- **Target Variable** (also: label, output) — what we are trying to predict.
  In our project: `TARGET` (0 = repaid, 1 = defaulted).

Think of it like baking:
- **Features** = ingredients (flour, eggs, sugar, butter)
- **Target** = the dish you want to produce (a cake)
- **Model** = the recipe that transforms ingredients into the dish

The model learns the "recipe" from thousands of past examples.

---

### Training Set vs. Test Set

We always split our data into at least two parts:

**Training Set (80% of data)**
The model learns from these examples during training.
Think of it as the textbook and practice problems before an exam.

**Test Set (20% of data)**
Held back during training. Used only for final evaluation.
Think of it as the actual exam — the questions should be new to the student.

> **Why this matters so much:**
> If we evaluated the model on the same data it learned from, we would be measuring
> memorization, not learning. A model that memorizes training data but fails on new data
> is called **overfitted** — and it is completely useless in practice, because every new
> loan applicant is "new data."

**Overfitting analogy:**
A student memorizes every answer from last year's exam. They score 100% on that exam.
But this year's exam has different questions, and they fail. They memorized, not learned.
An overfitted model does exactly the same thing.

**Stratified Split:**
Because our dataset is imbalanced (~8% defaults), we use a **stratified split** to ensure
both train and test sets have the same proportion of defaults:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% goes to test
    random_state=42,    # seed for reproducibility
    stratify=y          # maintain class proportions in both splits
)
```

---

### Class Imbalance

About 91.9% of loans were repaid and only 8.1% defaulted. This is **class imbalance**.

**Why is it a problem?**

A naive model that always predicts "repaid" — no matter what — would be 91.9% accurate.
But it would never flag a single defaulter. The bank would approve every risky applicant.
Accuracy is therefore a completely misleading metric for this problem.

**Three strategies to handle imbalance:**

**Strategy 1 — SMOTE (Synthetic Minority Oversampling Technique)**

SMOTE creates new, *synthetic* examples of the minority class by interpolating between
existing minority examples. It does not just copy existing defaulters — it creates plausible
new ones that fill the "space" around them.

```
Existing defaulter A: income=30k, ext_score=0.40
Existing defaulter B: income=35k, ext_score=0.45
Synthetic new point:  income=32k, ext_score=0.42  ← halfway between A and B
```

Applied only to training data. The test set must remain imbalanced to reflect reality.

**Strategy 2 — Class Weights**

Tell the model that mistakes on the minority class cost more.
In Random Forest: `class_weight="balanced"`.
The model then penalizes missing a defaulter (False Negative) more heavily.

**Strategy 3 — Threshold Adjustment**

By default we predict "defaulter" if probability > 0.5. Lowering this threshold to 0.3
flags more applicants as risky — catching more real defaulters at the cost of more false alarms.
Banks choose this threshold based on their risk appetite.

---

### Evaluation Metrics

Because accuracy is misleading for imbalanced problems, we use richer metrics.

**The Confusion Matrix**

Every prediction falls into one of four categories:

```
                      PREDICTED
                  Repaid    Defaulted
ACTUAL  Repaid  |   TN    |    FP   |
        Default |   FN    |    TP   |
```

- **True Positive (TP):** Predicted default, actually defaulted ✅
- **True Negative (TN):** Predicted repaid, actually repaid ✅
- **False Positive (FP):** Predicted default, actually repaid ❌ (wrongly denied a good customer)
- **False Negative (FN):** Predicted repaid, actually defaulted ❌ (approved a borrower who defaults)

**Which error is more expensive for a bank?**

A False Negative costs the bank the full outstanding loan amount.
A False Positive costs only one lost customer's interest income.
Banks therefore tune their models to minimize False Negatives — even if it means more
False Positives (conservative lending).

**Precision**
Of all predicted defaulters, what fraction actually defaulted?

```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity)**
Of all actual defaulters, what fraction did we catch?

```
Recall = TP / (TP + FN)
```

High recall is critical for banks. Missing a real defaulter is the most expensive mistake.

**F1-Score**
The harmonic mean of Precision and Recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**AUC-ROC (Area Under the ROC Curve)**

The ROC curve plots True Positive Rate against False Positive Rate at every threshold.

- **AUC = 0.50** → no better than random (coin flip)
- **AUC = 0.70** → reasonable, the model has real predictive power
- **AUC = 0.76** → our XGBoost result — strong for this difficult problem
- **AUC = 1.00** → perfect model (unrealistic in practice)

> **Intuition for AUC:** If you randomly pick one actual defaulter and one actual repayer,
> AUC is the probability that the model scores the defaulter higher.
> AUC = 0.76 means in 76 out of 100 such random pairs, the model correctly ranks the defaulter.

**Precision-Recall Tradeoff**

There is always a tradeoff between precision and recall. Making the model more aggressive
(lower threshold) catches more defaulters (higher recall) but also wrongly flags more
good applicants (lower precision). The bank must choose where on this curve to operate
based on the relative costs of each type of error.

---

### Model Interpretability and SHAP

**Why do we need interpretability?**

A model can be perfectly accurate but still useless if nobody understands it:
- A loan officer must explain to an applicant why they were denied
- A regulator must verify the model does not discriminate
- Management must trust the system before acting on its recommendations

**What is SHAP?**

SHAP stands for **SHapley Additive exPlanations**. It comes from cooperative game theory —
specifically from Lloyd Shapley's 1953 Nobel Prize-winning concept about fairly distributing
credit among players who cooperate to produce an outcome.

Applied to machine learning, SHAP answers: "How much did each feature contribute to *this
specific* prediction?"

**The restaurant bill analogy**

Three friends go to dinner. The total bill is $90. But the question is: how much did each
person's food order contribute to the total? SHAP fairly distributes the "blame" (or credit)
by considering every possible combination of people ordering:

```
Alice alone   → $30
Bob alone     → $40
Carol alone   → $20
Alice + Bob   → $60  (savings of $10 vs. ordering separately)
...
```

In machine learning, the "players" are features and the "bill" is the model's prediction.
SHAP fairly allocates the prediction among all features.

**Types of SHAP output:**

**Summary Plot (Beeswarm):** Global view — shows impact of all features across all predictions.
Each dot = one applicant. X-axis = SHAP value. Red = high feature value, Blue = low.

**Bar Plot:** Simpler global view — shows mean absolute SHAP value per feature.

**Waterfall Plot:** Individual prediction — shows exactly how each feature pushed the
prediction up or down from the baseline for one specific applicant:

```
Baseline (average applicant):         8.1% default probability
  EXT_SOURCE_2 is very low (0.25):  +18.3%  ← big red flag
  Age is young (22 years):           + 5.1%  ← additional risk
  Employment only 6 months:          + 3.8%  ← unstable income
  Income is $65,000 (above avg):     − 4.2%  ← reduces risk
  Owns real estate:                  − 2.1%  ← reduces risk
  ─────────────────────────────────────────
Final prediction:                     29.0% default probability
```

This is exactly what you would show a regulator or include in a rejection letter.

---

## 🤖 The Three Models We Use

### Model 1: Logistic Regression (Baseline)

Logistic Regression is our starting point — the simplest classification model.
Despite its name, it is a classification algorithm, not a regression one.

**How it works:**

The model learns a linear combination of features, then squashes the result through
a sigmoid function to get a probability between 0 and 1:

```
raw_score   = w₁×income + w₂×age + w₃×credit_score + ... + bias
probability = 1 / (1 + e^(−raw_score))
```

Training finds the weights (w₁, w₂, w₃...) that make predictions as accurate as possible.

**Strengths:** Fast, interpretable, good baseline, works well for linear relationships.

**Weaknesses:** Cannot automatically capture complex non-linear relationships or interactions
between features. Often underperforms tree-based methods on tabular data.

**Rule of thumb:** Always start here. If it works well enough, you do not need complexity.

---

### Model 2: Random Forest

A Random Forest combines many decision trees into one powerful ensemble model.

**How a Decision Tree works:**

A tree splits the data through a series of yes/no questions:

```
Is EXT_SOURCE_2 < 0.40?
├── YES → Is AGE < 27?
│         ├── YES → 65% default probability
│         └── NO  → 32% default probability
└── NO  → Is INCOME > $60,000?
          ├── YES → 5% default probability
          └── NO  → 18% default probability
```

Single trees tend to overfit — they memorize training data and fail on new data.

**The Forest solution — Bagging:**

1. Take a random sample of training data (with replacement) → Bootstrap
2. Train a tree on that sample, using only a random subset of features
3. Repeat 100+ times to get 100+ diverse trees
4. Final prediction = average of all trees → Aggregating

**Why does averaging help?**
Each tree overfits in a different direction (because it saw different data).
When you average many such models, their individual errors cancel out.
This is the "wisdom of crowds" applied to machine learning.

**Strengths:** Handles non-linearity, robust to outliers, provides feature importances.

**Weaknesses:** Less interpretable than Logistic Regression (hundreds of trees),
slower to train, memory-intensive for large forests.

---

### Model 3: XGBoost (Our Final Model)

XGBoost (**eXtreme Gradient Boosting**) is consistently one of the best-performing algorithms
on tabular data. It has won hundreds of Kaggle competitions, including the original
Home Credit competition.

**Boosting vs. Bagging — the key difference:**

| | Random Forest (Bagging) | XGBoost (Boosting) |
|--|-------------------------|---------------------|
| Trees built | Independently, in parallel | Sequentially, one after another |
| Each tree focuses on | Random subset of data | Errors from previous trees |
| Final prediction | Average of all trees | Weighted sum of all trees |
| Analogy | Ask 100 experts simultaneously | A student who studies their wrong answers |

**How XGBoost works step by step:**

```
Round 1: Build tree #1 → makes predictions → many applicants misclassified
Round 2: Build tree #2 → focuses extra attention on cases wrong in Round 1
Round 3: Build tree #3 → focuses on cases still wrong after Rounds 1+2
...
Round N: Final prediction = weighted combination of all N trees
```

This sequential error-correction is what makes boosting so powerful.

**Key hyperparameters:**

| Parameter | What It Does | Our Value |
|-----------|-------------|-----------|
| `n_estimators` | Number of trees | 200 |
| `max_depth` | How deep each tree grows | 6 |
| `learning_rate` | How much each tree contributes | 0.05 (small = careful) |
| `scale_pos_weight` | Weight for the minority class | ~11 (handles imbalance) |

**Strengths:** Usually best performance on tabular data, handles missing values natively,
built-in regularization, highly customizable.

**Weaknesses:** Many hyperparameters to tune, less interpretable without SHAP.

---

## 📁 Project Structure

```
loan-default-predictor/
│
├── README.md                              ← You are reading this file
├── requirements.txt                       ← All Python packages needed
│
├── data/
│   └── raw/                               ← Place Kaggle CSVs here
│       ├── application_train.csv          ← 307,511 rows, 122 columns
│       └── application_test.csv           ← 48,744 rows, 121 columns
│
├── stages/                                ← The five learning notebooks
│   ├── stage_01_setup_and_data.ipynb      ← Libraries, load data, first look
│   ├── stage_02_eda.ipynb                 ← Exploratory data analysis
│   ├── stage_03_preprocessing_and_features.ipynb  ← Clean, encode, engineer
│   ├── stage_04_models_and_evaluation.ipynb       ← Train and compare models
│   └── stage_05_shap_and_export.ipynb     ← Explain, save, and export
│
├── models/                                ← Created automatically by Stage 5
│   ├── final_model.pkl                    ← Trained XGBoost model
│   ├── scaler.pkl                         ← Fitted StandardScaler
│   └── feature_names.pkl                 ← List of feature column names
│
├── app/
│   └── app.py                             ← Streamlit web application
│
├── dashboard/
│   └── data_exports/                      ← CSVs for Tableau (from Stage 5)
│       ├── predictions.csv                ← Test set predictions + risk labels
│       ├── feature_importance.csv         ← XGBoost feature importances
│       └── sample_applicants.csv          ← Sample profiles for Dashboard 3
│
└── reports/
    └── figures/                           ← All plots saved by the notebooks
```

---

## 🗺️ The Five Stages — Learning Roadmap

Work through these notebooks in order. Each builds on the previous one.

---

### 📦 Stage 1 — Setup & Data Loading
`stage_01_setup_and_data.ipynb` · Estimated time: 20–30 minutes

**What you will do:**
- Import all Python libraries needed for the project
- Load the Home Credit dataset into a pandas DataFrame
- Inspect the data: shape, data types, missing value counts, summary statistics
- Visualize the target variable and understand the class imbalance

**Key question to answer:** Why is a model that achieves 91.9% accuracy by always predicting
"repaid" completely useless for a bank?

---

### 🔍 Stage 2 — Exploratory Data Analysis
`stage_02_eda.ipynb` · Estimated time: 30–45 minutes

EDA is detective work. You look for clues about what drives default.

**What you will do:**
- Analyze missing values — how much data is absent in each column and why
- Compare the age distributions of defaulters vs. repayers
- Calculate default rates by education level, income type, and housing type
- Examine the predictive power of external credit scores (EXT_SOURCE_1/2/3)
- Build a correlation matrix of key numerical features

**Key question to answer:** Which features show the strongest association with default?

---

### 🛠️ Stage 3 — Preprocessing & Feature Engineering
`stage_03_preprocessing_and_features.ipynb` · Estimated time: 40–60 minutes

Raw data is never ready for a machine learning model. This stage is about cleaning and
transforming it into a form the model can learn from.

**What you will do:**
- Select the most relevant features from the 122 available
- Handle missing values with median imputation (numerical) and "Unknown" (categorical)
- Fix the DAYS_EMPLOYED anomaly (the mysterious 365243 placeholder value)
- Apply One-Hot Encoding to convert categorical text into numbers
- Split data into train/test sets using stratified sampling
- Scale numerical features with StandardScaler
- Engineer new features: credit-to-income ratio, annuity-to-income ratio

**Key concepts introduced:** Data leakage, One-Hot Encoding, StandardScaler, feature engineering.

**Key question to answer:** Why must we fit the scaler only on the training set and then apply
(not re-fit) it on the test set?

---

### 🤖 Stage 4 — Imbalance, Models & Evaluation
`stage_04_models_and_evaluation.ipynb` · Estimated time: 60–90 minutes

This is the modeling core of the project.

**What you will do:**
- Apply SMOTE to create a balanced training set
- Train Logistic Regression as a baseline
- Train Random Forest and understand the concept of bagging
- Train XGBoost and understand the concept of boosting
- Compare all three models using AUC-ROC, Precision, Recall, F1-Score
- Build and interpret ROC curves and Precision-Recall curves
- Read a confusion matrix and understand the cost of each type of error
- Analyze feature importances from XGBoost

**Key question to answer:** Why does XGBoost outperform Logistic Regression on this dataset?

---

### 🧠 Stage 5 — SHAP & Export
`stage_05_shap_and_export.ipynb` · Estimated time: 30–45 minutes

This final stage makes the model understandable and deployable.

**What you will do:**
- Generate global SHAP explanations — which features matter most overall?
- Generate individual SHAP waterfall charts — why was this specific applicant scored this way?
- Save the trained model, scaler, and feature list to disk with joblib
- Export predictions and feature importances as CSVs for Tableau
- Understand the regulatory and ethical implications of deploying a credit model

**Key question to answer:** How would you explain to a rejected loan applicant — or to a
regulator — why the model scored them the way it did?

---

## ⚙️ Setup & Installation

### Prerequisites

Before starting, ensure you have:
- **Python 3.8 or higher** — [python.org](https://www.python.org/downloads/)
- **pip** — comes bundled with Python
- **Jupyter Notebook or JupyterLab** — to open `.ipynb` files
- **A Kaggle account** — free, needed to download the dataset

Check your version:
```bash
python --version   # Should show 3.8 or higher
```

### Step 1 — Get the Code

```bash
git clone https://github.com/yourusername/loan-default-predictor.git
cd loan-default-predictor
```

Or download the ZIP from GitHub and unzip it.

### Step 2 — Create a Virtual Environment (Recommended)

A virtual environment keeps this project's packages separate from your other Python projects.

```bash
# Create it
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Your terminal prompt should now show (venv)
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages. Allow 2–5 minutes.

**What you are installing:**

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥1.5.0 | Data manipulation — DataFrames |
| `numpy` | ≥1.23.0 | Numerical computing — arrays and math |
| `matplotlib` | ≥3.6.0 | Static plots and charts |
| `seaborn` | ≥0.12.0 | Statistical visualizations |
| `scikit-learn` | ≥1.2.0 | ML toolkit — preprocessing, models, metrics |
| `xgboost` | ≥1.7.0 | Gradient boosting classifier |
| `imbalanced-learn` | ≥0.10.0 | SMOTE and other resampling tools |
| `shap` | ≥0.41.0 | Model explanation library |
| `streamlit` | ≥1.20.0 | Web application framework |
| `joblib` | ≥1.2.0 | Save and load Python objects |
| `plotly` | ≥5.13.0 | Interactive visualizations |

### Step 4 — Create Required Folders

```bash
mkdir -p data/raw models reports/figures dashboard/data_exports
```

### Step 5 — Launch Jupyter

```bash
jupyter notebook
```

A browser window will open. Navigate to the `stages/` folder and open Stage 1.

---

## 📥 How to Download the Data

1. Go to [https://www.kaggle.com/c/home-credit-default-risk/data](https://www.kaggle.com/c/home-credit-default-risk/data)
2. Create a free Kaggle account if you do not have one
3. Click **"I Understand and Accept"** to accept the competition rules
4. Download `application_train.csv` (~166 MB) and `application_test.csv` (~74 MB)
5. Place both files inside `data/raw/`

**Verify the download:**
```python
import pandas as pd
df = pd.read_csv("data/raw/application_train.csv")
print(df.shape)   # Should print: (307511, 122)
print(df["TARGET"].value_counts())
# 0    282686
# 1     24825
```

---

## 🌐 Running the Web Application

After completing all five stages (the model files are saved automatically by Stage 5):

```bash
streamlit run app/app.py
```

Your browser opens at `http://localhost:8501`.

**What the app does:**

The sidebar lets you enter an applicant's details. When you click "Predict":

1. Inputs are collected and preprocessed using the same pipeline as training
2. The saved XGBoost model produces a default probability (0–100%)
3. The app classifies risk as Low / Medium / High
4. A SHAP waterfall chart shows which factors drove the prediction

**Try these test scenarios to see the model in action:**

**Scenario A — Low Risk:**
Age 45, Income $80,000, Loan $200,000, Education: Higher, Employed 10 years, EXT_SOURCE_2: 0.75
→ Expected: Low risk (~10–15% default probability)

**Scenario B — High Risk:**
Age 22, Income $22,000, Loan $180,000, Education: Secondary, Employed 6 months, EXT_SOURCE_2: 0.28
→ Expected: High risk (~55–70% default probability)

Try adjusting individual sliders and observe how the prediction and SHAP chart change.
This is a powerful way to build intuition for what drives the model.

---

## ⚖️ Regulatory and Ethical Considerations

This section is not optional reading. Understanding the ethical implications of the systems
you build is a core professional responsibility in data science.

### The Fair Lending Problem

Credit scoring models can discriminate, even unintentionally. If a model is trained on
historical data from a period when certain groups were systematically denied credit due to
human bias, the model may learn to associate group membership with default risk — not because
of any genuine risk difference, but because of historical discrimination embedded in the data.

This is called **disparate impact**: a model that appears neutral but has unequal effects
on protected groups.

### Key Regulations to Know

**ECOA — Equal Credit Opportunity Act (USA)**
Prohibits discrimination based on race, color, religion, national origin, sex, marital
status, age, or receipt of public assistance in any aspect of a credit transaction.

**Regulation B**
Implements ECOA. Requires lenders to provide applicants with a written notice of adverse
action and the specific reasons for it. This is why model explainability is a legal requirement,
not just a nice-to-have.

**GDPR — General Data Protection Regulation (European Union)**
Includes the right to explanation — individuals have the right to meaningful information
about the logic behind automated decisions that significantly affect them.

**Basel III**
International banking regulation requiring banks to hold capital reserves proportional to
their credit risk exposure. Better risk models can reduce required capital, creating a direct
financial incentive to improve prediction quality.

### What We Have Done in This Project

1. **Explainability via SHAP:** Every prediction includes a full feature-level explanation.
   This satisfies adverse action notice requirements.

2. **No explicitly prohibited attributes:** We do not use race, religion, or national origin.
   Note that gender and age are used — their legality in credit scoring varies by jurisdiction
   and is an active area of regulatory debate.

3. **Imbalance handling:** We actively work to ensure the model is not biased toward the
   majority class, which would disproportionately harm applicants from groups with limited
   credit history.

### Important Disclaimer

> ⚠️ **This project is for educational purposes only.**
> It is not intended for real lending decisions.
> Deploying a credit model in production requires legal review, regulatory approval,
> extensive fairness testing, ongoing performance monitoring, and governance processes
> far beyond the scope of this educational exercise.

---

## 📖 Glossary of Terms

Every technical term used in this project, defined in plain language.

**Algorithm** — A set of instructions a computer follows to solve a problem. In ML, an
algorithm learns patterns from data rather than following explicitly programmed rules.

**AUC-ROC** — Area Under the Receiver Operating Characteristic Curve. Measures how well a
classifier separates two classes. 0.5 = random, 1.0 = perfect.

**Bagging (Bootstrap Aggregating)** — An ensemble technique that trains many models on
different random subsets of training data and combines their predictions. Used in Random Forest.

**Baseline Model** — The simplest model you build first, to set a performance benchmark.
Logistic Regression is our baseline.

**Binary Classification** — A task where the output is one of exactly two categories
(default or repay, spam or not spam).

**Boosting** — An ensemble technique that trains models sequentially, each correcting the
errors of the previous ones. Used in XGBoost.

**Class Imbalance** — When one class is much more common than the other. Our dataset:
91.9% repaid vs. 8.1% defaulted.

**Classification** — A ML task predicting a category or class label, not a number.

**Confusion Matrix** — A table showing TP, TN, FP, and FN counts for a classifier.

**Cross-Validation** — Estimating model performance by splitting training data into multiple
folds and evaluating on each fold in turn.

**Data Leakage** — When test set information accidentally influences training, causing
overly optimistic performance estimates. One of the most common and costly mistakes in ML.

**Decision Tree** — A model making predictions through a series of yes/no questions on features.

**Default** — Failure to meet loan repayment obligations, typically defined as 90+ days past due.

**Ensemble** — A model combining predictions from many individual models.

**F1-Score** — Harmonic mean of Precision and Recall. Balanced metric when both matter.

**False Negative (FN)** — Predicting "repaid" when the truth is "defaulted."
In lending: approving a borrower who will default. The costliest error for a bank.

**False Positive (FP)** — Predicting "defaulted" when the truth is "repaid."
In lending: denying a borrower who would have repaid.

**Feature** — A column in the dataset used as input to the model (income, age, score, etc.).

**Feature Engineering** — Creating new, informative features from existing ones.

**Feature Importance** — How much each feature contributed to the model's predictions,
averaged across all training examples.

**Generalization** — A model's ability to perform well on new, unseen data.

**Hyperparameter** — A model setting chosen before training (e.g., number of trees).
Distinct from parameters, which are learned during training.

**Imbalanced Dataset** — A dataset where class distribution is significantly unequal.

**Imputation** — Filling in missing values using a strategy (median, mean, mode, etc.).

**Label** — Another word for the target variable.

**Logistic Regression** — A linear model predicting the probability of a binary outcome.

**Model** — A mathematical function trained on data to make predictions.

**One-Hot Encoding** — Converting a categorical variable with N categories into N binary
columns. For example, Gender {M, F} → Gender_M {0,1}, Gender_F {0,1}.

**Overfitting** — A model that memorizes training data (including its noise) and fails
on new data. The opposite of generalization.

**Precision** — Of all predicted positives, the fraction that were correct.
`Precision = TP / (TP + FP)`

**Recall (Sensitivity)** — Of all actual positives, the fraction correctly identified.
`Recall = TP / (TP + FN)`

**Regularization** — Techniques that prevent overfitting by penalizing model complexity.

**SHAP (SHapley Additive exPlanations)** — A method explaining individual predictions by
fairly attributing each feature's contribution to the output.

**SMOTE (Synthetic Minority Oversampling Technique)** — Creates synthetic minority class
examples to address class imbalance during training.

**StandardScaler** — Transforms features to mean=0, standard deviation=1.

**Stratified Split** — A train/test split that preserves the class distribution.

**Supervised Learning** — ML where training data includes labeled examples (known outputs).

**Target Variable** — The column the model predicts. Our target: `TARGET` (0 or 1).

**Test Set** — Data held back during training, used only for final evaluation.

**Training Set** — Data used to fit the model.

**True Negative (TN)** — Correctly predicting "repaid" for someone who actually repaid.

**True Positive (TP)** — Correctly predicting "defaulted" for someone who actually defaulted.

**Underfitting** — A model too simple to capture patterns in the data.

**XGBoost (eXtreme Gradient Boosting)** — A powerful ensemble boosting algorithm.
Often the best-performing algorithm on tabular data.

---

## 📚 Further Reading

If you want to go deeper on any topic covered in this project:

### Books

**Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** — Aurélien Géron.
The single best practical ML book. Chapter 3 covers classification in excellent depth.

**The Elements of Statistical Learning** — Hastie, Tibshirani, Friedman.
More mathematical, but the gold standard reference for ML theory.
Free PDF at [hastie.su.domains](https://hastie.su.domains/ElemStatLearn/)

**Interpretable Machine Learning** — Christoph Molnar.
Dedicated entirely to explaining ML models. Covers SHAP in depth.
Free at [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book)

### Online Courses

- [fast.ai Practical Machine Learning](https://course.fast.ai) — Practical, free, world-class.
- [Kaggle Learn](https://www.kaggle.com/learn) — Short, focused modules on pandas, ML, SHAP.
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — The official
  documentation is genuinely excellent and full of examples.

### Key Papers

- **XGBoost: A Scalable Tree Boosting System** — Chen & Guestrin, 2016.
  The original XGBoost paper. Worth reading for any serious practitioner.

- **A Unified Approach to Interpreting Model Predictions** — Lundberg & Lee, 2017.
  The SHAP paper. Explains the theory clearly.

- **SMOTE: Synthetic Minority Over-sampling Technique** — Chawla et al., 2002.
  The original SMOTE paper.

### Kaggle Resources

- [Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk) —
  Browse the top-ranked solutions. Many winning teams published detailed notebooks.

- [Will Koehrsen's Home Credit Notebooks](https://www.kaggle.com/willkoehrsen) —
  Exceptionally clear and educational notebooks on this exact dataset.

---

## ❓ Frequently Asked Questions

**Q: Do I need prior machine learning knowledge to start?**

No. You need basic Python (variables, loops, functions, lists). Everything about machine
learning is taught from scratch within the notebooks. If you need a Python refresher,
complete the Kaggle Python course first (about 5 hours, free).

**Q: Why XGBoost instead of a neural network?**

For tabular data (rows and columns in a table), gradient boosted trees like XGBoost typically
outperform neural networks. Neural networks excel at unstructured data: images, audio, and text.
On structured tabular data — the vast majority of business data science — XGBoost is usually
the better choice: faster to train, easier to tune, and more interpretable.

**Q: What is the reload cell at the start of each stage?**

Each Stage 2–5 notebook opens with a "Reload Cell" that re-runs essential setup from prior stages.
This lets you start any stage fresh without running all previous stages in the same Jupyter session.
If you already ran the previous stage and have all variables in memory, skip it.

**Q: SMOTE is very slow. Is that normal?**

Yes. SMOTE on 245,000 training samples can take 2–5 minutes on a typical laptop. This is normal.
In production you would precompute the balanced dataset and save it to disk rather than regenerating
it each session.

**Q: My AUC is slightly different from what the README says. Is something wrong?**

Small differences are expected. SMOTE and XGBoost both have internal randomness. We set
`random_state=42` wherever possible, but some variation remains. If your AUC is in the range
0.73–0.78, your results are correct.

**Q: Can I use this model for real loan decisions?**

Absolutely not. A production credit model requires legal review, regulatory approval, fairness
auditing, model risk management governance, and ongoing monitoring for model drift — far beyond
this educational exercise. Please read the Regulatory and Ethical Considerations section.

**Q: How do I add features from bureau.csv and the other files?**

The other files must be aggregated by client ID (`SK_ID_CURR`) and joined to the main table.
This is an excellent extension exercise. Browse top Kaggle notebooks for this competition
to see how professionals approach the join and aggregation.

**Q: Why median imputation instead of mean?**

The median is robust to outliers. Financial data often contains extreme values — very high
incomes or very large credit amounts. These outliers pull the mean away from the typical
value. The median — the middle value when sorted — is unaffected by outliers and better
represents the typical applicant we want to impute for.

**Q: What is `random_state=42` and why 42?**

`random_state` is a seed for Python's random number generator. Setting it ensures that any
step involving randomness (train/test split, SMOTE, tree building) produces the same result
every time you run the code. 42 is a popular default in the data science community —
a reference to *The Hitchhiker's Guide to the Galaxy*, where 42 is "the answer to everything."
Any integer works just as well.

---

## 📄 License

This project is released under the **MIT License**.
You are free to use, copy, modify, and distribute this code for any purpose, including
commercial use, provided you include the original license notice.

---

## 🙏 Acknowledgements

- **Home Credit Group** for releasing this dataset to the public via Kaggle
- The **Kaggle community**, especially Will Koehrsen, whose notebooks inspired this project's structure
- The **scikit-learn**, **XGBoost**, and **SHAP** development teams for outstanding open-source tools
- While the fundamental design and primary development efforts are the author's own, **AI technology** was leveraged to refine the project's presentation and code quality. This included utilizing AI for code linting, formatting consistency, and identifying areas for performance improvement. Every AI-suggested modification was subject to the author's review and approval to ensure it met the project's specific requirements and standards.

---

*Built for educational purposes. Designed to teach the complete data science lifecycle —
from messy raw data to a deployed, explainable machine learning application.*

*Happy learning! 🚀*
