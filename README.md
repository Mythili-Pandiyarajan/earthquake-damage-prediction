🛡️ EarthShield — Earthquake Building Damage Predictor
Predicting building damage grade from structural and geographic features using Machine Learning — supporting post-earthquake triage, disaster preparedness, and seismic risk assessment.
🔴 Live Demo
earthquake-damage-prediction-mythili.streamlit.app

📌 Problem Statement
The 2015 Gorkha earthquake in Nepal caused widespread destruction across thousands of buildings. Physical inspection of every damaged building after a disaster is time-consuming and resource-intensive.
This project builds an ML system to predict the damage grade of a building from its structural and geographic characteristics — helping engineers, policymakers, and seismologists:

Prioritise post-earthquake inspection and rescue efforts
Identify high-risk building types before a disaster
Guide future building codes and construction standards

Problem Type: Multiclass Ordinal Classification
Target variable — damage_grade:

1 = Low damage
2 = Medium damage
3 = Almost complete destruction


📊 Dataset
ItemDetailSourceDrivenData — Richter's Predictor CompetitionSize260,601 buildingsAfter deduplication248,282 rowsFeatures40 (after engineering)Targetdamage_grade (1 / 2 / 3)Class distributionGrade 2: 56.9% · Grade 3: 33.5% · Grade 1: 9.6%

⚙️ ML Pipeline
StepDetailDuplicates removed12,319 exact duplicates droppedClass imbalanceclass_weight='balanced' + F1 Macro metric + stratified splitTarget encodinggeo_level_1/2/3 replaced with mean damage grade per region — biggest single accuracy boost (~0.65 → ~0.75)Feature engineeringage_x_floors, area_x_height — capture interaction effectsNumeric preprocessingMedian imputation → StandardScalerCategorical preprocessingMode imputation → OrdinalEncoderModels comparedLogistic Regression, Decision Tree, Random Forest, CatBoost, XGBoostTuningGridSearchCV (3-fold CV, F1 Macro scoring)Best modelXGBoost (GridSearchCV tuned)

📈 Results
ModelAccuracyF1 MacroLogistic Regression0.68910.6612Decision Tree0.64760.5975Random Forest0.73110.6761CatBoost0.74970.7040XGBoost (Base)0.75010.6997XGBoost (Tuned)0.75290.7046
5-fold Cross Validation: Mean F1 Macro = 0.7038 ± 0.003
Best Hyperparameters:
learning_rate     = 0.05
max_depth         = 10
n_estimators      = 300
subsample         = 0.8
colsample_bytree  = 0.8
Per-class performance:
GradePrecisionRecallF1Grade 1 · Low0.580.550.56Grade 2 · Medium0.770.820.79Grade 3 · High0.720.640.68

🔍 Key Findings

Geographic micro-region is the strongest predictor — geo_level_3_id has feature importance of 0.179. Where a building is matters more than how it was built.
Roof type is the 2nd most important feature — bamboo/timber roofs show the highest damage rate in the dataset.
Mud mortar stone superstructures strongly correlate with Grade 2/3 damage.
RC engineered construction is the most protective wall material.
Older + taller buildings are most vulnerable — age_x_floors ranked in top 5, validating domain reasoning.
SMOTE was deliberately excluded — class_weight='balanced' achieved equivalent results without data leakage risk.

Suggestions to seismologists:

Prioritise inspection of stone/brick foundation buildings in high geo-risk zones
Mandate RC engineering in new construction in Grade 3 zones
Buildings older than 50 years with bamboo/timber roofs should be retrofitted first
Building height regulations should account for foundation type in seismic zones


🚀 Deployment
ItemDetailFrameworkStreamlitPlatformStreamlit CloudPython version3.11Model serialisationjoblib
App features: Single prediction with confidence score · SHAP-style explainability · Example scenario buttons · Batch CSV prediction · Analytics dashboard with confusion matrix

🛠️ Tech Stack
Show Image
Show Image
Show Image
Show Image
Show Image
Show Image

📁 Project Structure
earthquake-damage-prediction/
│
├── app.py                          → Streamlit application (4 pages)
├── earthquake_model.pkl            → Trained XGBoost pipeline (joblib)
├── requirements.txt                → Dependencies with pinned versions
├── runtime.txt                     → Python 3.11 for Streamlit Cloud
├── .python-version                 → Python version pin
└── earthquake_prediction.ipynb    → Full ML notebook

🔮 Future Improvements

Integrate true SHAP library for production-grade explainability
Add soil liquefaction and topographic data as features
Train on multiple earthquake datasets for better generalisation
Docker containerisation for portable deployment
REST API endpoint for GIS/disaster management system integration


👩‍💻 Author
Mythili P · Data Science Portfolio Project
