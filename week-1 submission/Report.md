# Machine-Learning (Week-1)

## Approach

### Finding Dataset
Kaggle

### Model Selection and Training
Selected a scikit-learn RandomForestClassifier.
Serialized model using joblib (`rf_model.joblib`).

### FastAPI Integration
Implemented a POST endpoint `/predict` using FastAPI to receive feature inputs and return predictions. Used Pydantic for request validation.

## Challenges
Encountered scikit-learn version compatibility issues during model loading (`InconsistentVersionWarning`). Resolved by ensuring version consistency.

## Results
Successfully deployed the scikit-learn model using FastAPI.
Learned about ml-basics , feature engineering , shap , FastAPI.

---
