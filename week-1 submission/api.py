from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

# Load the model with dtype fix
model_path = 'rf_model.joblib'

with open(model_path, 'rb') as f:
    model = joblib.load(f)

print(model)

class Transaction(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type_CASH_IN: int
    type_CASH_OUT: int
    type_DEBIT: int
    type_PAYMENT: int
    type_TRANSFER: int
    
# Define POST endpoint for fraud detection
@app.post("/predict_fraud")
def predict_fraud(transaction: Transaction):
    # Extract features from request body and convert to numpy array
    features = np.array([[transaction.step, transaction.amount, transaction.oldbalanceOrg,
                          transaction.newbalanceOrig, transaction.oldbalanceDest,
                          transaction.newbalanceDest, transaction.type_CASH_IN,
                          transaction.type_CASH_OUT, transaction.type_DEBIT,
                          transaction.type_PAYMENT, transaction.type_TRANSFER]])
    
    # Predict fraud probability (assuming the model predicts probabilities)
    fraud_probability = model.predict_proba(features)[:, 1]  # Assuming 1 is the index for fraud class
    
    # Return prediction as JSON
    return {"fraud_probability": float(fraud_probability)}

@app.get("/")
def read_root():
    return {'message': 'Welcome to Home Page!'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)