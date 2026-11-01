import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend.churn_engine import preprocess_and_train_churn, predict_single_customer_churn, get_churn_insights, MODEL_METRICS

app = FastAPI(
    title="Telco Customer Churn Prediction & Retention Platform",
    description="Enterprise-grade Churn Analytics Platform using Supervised Random Forest Classification",
    version="2.0.0"
)

RAW_CHURN_DF = pd.DataFrame()
PROCESSED_CHURN_DF = pd.DataFrame()
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_default_dataset():
    global RAW_CHURN_DF, PROCESSED_CHURN_DF
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    RAW_CHURN_DF = pd.read_csv(DATA_PATH)
    PROCESSED_CHURN_DF = preprocess_and_train_churn(RAW_CHURN_DF)

@app.on_event("startup")
def startup_event():
    load_default_dataset()

class CustomerChurnInput(BaseModel):
    customerID: str = "7590-VHVEG"
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "No"
    MultipleLines: str = "No phone service"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

@app.get("/api/summary")
def get_summary():
    global RAW_CHURN_DF
    if RAW_CHURN_DF.empty:
        load_default_dataset()
        
    insights = get_churn_insights(RAW_CHURN_DF)
    contract_counts = RAW_CHURN_DF['Contract'].value_counts().to_dict()
    internet_counts = RAW_CHURN_DF['InternetService'].value_counts().to_dict()
    
    from backend.churn_engine import MODEL_METRICS
    return {
        "insights": insights,
        "model_metrics": MODEL_METRICS,
        "contract_counts": contract_counts,
        "internet_counts": internet_counts
    }

@app.get("/api/segments")
def get_segments():
    global RAW_CHURN_DF
    if RAW_CHURN_DF.empty:
        load_default_dataset()
        
    contract_stats = []
    for contract, group in RAW_CHURN_DF.groupby('Contract'):
        total = len(group)
        churned = int((group['Churn'] == 'Yes').sum())
        rate = round((churned / total) * 100, 1) if total > 0 else 0
        contract_stats.append({
            "dimension": contract,
            "total": total,
            "churned": churned,
            "churn_rate": rate,
            "avg_monthly_charges": round(float(group['MonthlyCharges'].mean()), 2)
        })

    payment_stats = []
    for pm, group in RAW_CHURN_DF.groupby('PaymentMethod'):
        total = len(group)
        churned = int((group['Churn'] == 'Yes').sum())
        rate = round((churned / total) * 100, 1) if total > 0 else 0
        payment_stats.append({
            "dimension": pm,
            "total": total,
            "churned": churned,
            "churn_rate": rate,
            "avg_monthly_charges": round(float(group['MonthlyCharges'].mean()), 2)
        })

    return {
        "contract_stats": contract_stats,
        "payment_stats": payment_stats
    }

@app.post("/api/predict-churn")
def predict_churn(data: CustomerChurnInput):
    global RAW_CHURN_DF
    if RAW_CHURN_DF.empty:
        load_default_dataset()
        
    input_dict = data.dict()
    result = predict_single_customer_churn(input_dict)
    result["customerID"] = data.customerID
    result["MonthlyCharges"] = data.MonthlyCharges
    result["tenure"] = data.tenure
    return result

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global RAW_CHURN_DF, PROCESSED_CHURN_DF
    try:
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
        
        if 'Churn' not in df.columns:
            raise HTTPException(status_code=400, detail="Uploaded dataset missing 'Churn' target column.")
            
        RAW_CHURN_DF = df
        PROCESSED_CHURN_DF = preprocess_and_train_churn(df)
        
        return {
            "message": "Dataset uploaded and Churn Classification Model retrained successfully!",
            "total_records": len(df),
            "churn_rate": f"{round((df['Churn'] == 'Yes').mean() * 100, 1)}%"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")