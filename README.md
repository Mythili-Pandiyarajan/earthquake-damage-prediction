# 🛡️ EarthShield — Earthquake Building Damage Predictor

> Predicting building damage grade from structural and geographic features using Machine Learning — supporting post-earthquake triage, disaster preparedness, and seismic risk assessment.

---

# 🔴 Live Demo

https://earthquake-damage-prediction-mythili.streamlit.app/

---

# 📌 Problem Statement

The 2015 Gorkha earthquake in Nepal caused widespread destruction across thousands of buildings.

Physical inspection of every damaged building after a disaster is time-consuming and resource-intensive.

This project builds an ML system to predict the damage grade of a building from its structural and geographic characteristics — helping engineers, policymakers, and seismologists:

- Prioritise post-earthquake inspection and rescue efforts
- Identify high-risk building types before a disaster
- Guide future building codes and construction standards

---

# 🎯 Problem Type

### Multiclass Ordinal Classification

Target variable — `damage_grade`

| Grade | Meaning |
|---|---|
| 1 | Low damage |
| 2 | Medium damage |
| 3 | Almost complete destruction |

---

# 📊 Dataset

| Item | Detail |
|---|---|
| Source | DrivenData — Richter's Predictor Competition |
| Size | 260,601 buildings |
| After deduplication | 248,282 rows |
| Features | 40 (after engineering) |
| Target | `damage_grade` (1 / 2 / 3) |

### Class Distribution

- Grade 2: 56.9%
- Grade 3: 33.5%
- Grade 1: 9.6%

---

# ⚙️ ML Pipeline

| Step | Detail |
|---|---|
| Duplicates removed | 12,319 exact duplicates dropped |
| Class imbalance | `class_weight='balanced'` + F1 Macro metric + stratified split |
| Target encoding | `geo_level_1/2/3` replaced with mean damage grade per region — biggest single accuracy boost (~0.65 → ~0.75) |
| Feature engineering | `age_x_floors`, `area_x_height` — capture interaction effects |
| Numeric preprocessing | Median imputation → StandardScaler |
| Categorical preprocessing | Mode imputation → OrdinalEncoder |
| Models compared | Logistic Regression, Decision Tree, Random Forest, CatBoost, XGBoost |
| Tuning | GridSearchCV (3-fold CV, F1 Macro scoring) |
| Best model | XGBoost (GridSearchCV tuned) |

---

# 📈 Results

| Model | Accuracy | F1 Macro |
|---|---|---|
| Logistic Regression | 0.6891 | 0.6612 |
| Decision Tree | 0.6476 | 0.5975 |
| Random Forest | 0.7311 | 0.6761 |
| CatBoost | 0.7497 | 0.7040 |
| XGBoost (Base) | 0.7501 | 0.6997 |
| XGBoost (Tuned) | 0.7529 | 0.7046 |

### 5-fold Cross Validation

```python
Mean F1 Macro = 0.7038 ± 0.003
```

### Best Hyperparameters

```python
learning_rate = 0.05
max_depth = 10
n_estimators = 300
subsample = 0.8
colsample_bytree = 0.8
```

### Per-class Performance

| Grade | Precision | Recall | F1 |
|---|---|---|---|
| Grade 1 · Low | 0.58 | 0.55 | 0.56 |
| Grade 2 · Medium | 0.77 | 0.82 | 0.79 |
| Grade 3 · High | 0.72 | 0.64 | 0.68 |

---

# 🔍 Key Findings

- Geographic micro-region is the strongest predictor — `geo_level_3_id` has feature importance of `0.179`
- Where a building is matters more than how it was built
- Roof type is the 2nd most important feature — bamboo/timber roofs show the highest damage rate
- Mud mortar stone superstructures strongly correlate with Grade 2/3 damage
- RC engineered construction is the most protective wall material
- Older + taller buildings are most vulnerable — `age_x_floors` ranked in top 5
- SMOTE was deliberately excluded — `class_weight='balanced'` achieved equivalent results without data leakage risk

---

# 🧠 Suggestions to Seismologists

- Prioritise inspection of stone/brick foundation buildings in high geo-risk zones
- Mandate RC engineering in new construction in Grade 3 zones
- Buildings older than 50 years with bamboo/timber roofs should be retrofitted first
- Building height regulations should account for foundation type in seismic zones

---

# 🚀 Deployment

| Item | Detail |
|---|---|
| Framework | Streamlit |
| Platform | Streamlit Cloud |
| Python version | 3.11 |
| Model serialisation | joblib |

### App Features

- Single prediction with confidence score
- SHAP-style explainability
- Example scenario buttons
- Batch CSV prediction
- Analytics dashboard with confusion matrix

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Plotly
- Streamlit

---

# 📁 Project Structure

```text
earthquake-damage-prediction/
│
├── app.py
├── earthquake_model.pkl
├── requirements.txt
├── runtime.txt
├── .python-version
└── earthquake_prediction.ipynb
```

---

# 🔮 Future Improvements

- Integrate true SHAP library for production-grade explainability
- Add soil liquefaction and topographic data as features
- Train on multiple earthquake datasets for better generalisation
- Docker containerisation for portable deployment
- REST API endpoint for GIS/disaster management system integration

---

# 👩‍💻 Author

### Mythili P · Data Science Portfolio Project
