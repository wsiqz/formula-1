from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import io
import json
from typing import List, Dict, Any, Optional
import numpy as np

# Import your F1Pipeline class
# from your_module import F1Pipeline  # Uncomment and adjust import path

class F1Pipeline:
    '''
    A simple pipeline class to encapsulate the preprocessor and model for F1 predictions.
    This class allows fitting from raw data and making predictions on new data.
    '''
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.fitted = False
    
    def fit_from_raw(self, df_history):
        df_history = df_history.copy()
        df_history['dataset'] = 'train'
        self.preprocessor.fit(df_history)
        df_transformed = self.preprocessor.transform(df_history)
        df_transformed = df_transformed.copy()
        df_transformed['dataset'] = df_history.loc[self.preprocessor.surviving_indices_, 'dataset'].reset_index(drop=True)
        df_train = df_transformed[df_transformed['dataset'] == 'train'].drop(columns=['dataset'])
        y_train = df_train['podium']
        X_train = df_train.drop(columns=['podium', 'year'])
        self.model.fit(X_train, y_train)
        self.fitted = True
    
    def predict_from_raw(self, df_history, df_new):
        assert self.fitted, "You must call fit_from_raw() before prediction"
        df_history = df_history.copy()
        df_new = df_new.copy()
        df_history['dataset'] = 'train'
        df_new['dataset'] = 'test'
        df_all = pd.concat([df_history, df_new], ignore_index=True)
        df_transformed = self.preprocessor.transform(df_all)
        df_transformed = df_transformed.copy()
        df_transformed['dataset'] = df_all.loc[self.preprocessor.surviving_indices_, 'dataset'].reset_index(drop=True)
        df_test = df_transformed[df_transformed['dataset'] == 'test'].drop(columns=['dataset'])
        X_test = df_test.drop(columns=['podium', 'year'])
        return self.model.predict(X_test)
    
    def predict_proba_from_raw(self, df_history, df_new):
        assert self.fitted, "You must call fit_from_raw() before prediction"
        df_history = df_history.copy()
        df_new = df_new.copy()
        df_history['dataset'] = 'train'
        df_new['dataset'] = 'test'
        df_all = pd.concat([df_history, df_new], ignore_index=True)
        df_transformed = self.preprocessor.transform(df_all)
        df_transformed = df_transformed.copy()
        df_transformed['dataset'] = df_all.loc[self.preprocessor.surviving_indices_, 'dataset'].reset_index(drop=True)
        df_test = df_transformed[df_transformed['dataset'] == 'test'].drop(columns=['dataset'])
        X_test = df_test.drop(columns=['podium', 'year'])
        return self.model.predict_proba(X_test)

app = FastAPI(
    title="F1 Prediction API",
    description="API for F1 race predictions using machine learning pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[F1Pipeline] = None

# Pydantic models for request/response
class PipelineStatus(BaseModel):
    fitted: bool
    message: str

class PredictionRequest(BaseModel):
    history_data: List[Dict[str, Any]]
    new_data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[int]
    success: bool
    message: str

class ProbabilityResponse(BaseModel):
    probabilities: List[List[float]]
    success: bool
    message: str

class TrainingResponse(BaseModel):
    success: bool
    message: str

@app.get("/")
async def root():
    return {"message": "F1 Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "pipeline_loaded": pipeline is not None}

@app.get("/pipeline/status", response_model=PipelineStatus)
async def get_pipeline_status():
    if pipeline is None:
        return PipelineStatus(fitted=False, message="Pipeline not loaded")
    return PipelineStatus(fitted=pipeline.fitted, message="Pipeline loaded")

@app.post("/pipeline/load")
async def load_pipeline(file: UploadFile = File(...)):
    """Load a pre-trained pipeline from a pickle file"""
    global pipeline
    try:
        contents = await file.read()
        pipeline = pickle.loads(contents)
        return {"success": True, "message": "Pipeline loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading pipeline: {str(e)}")

@app.post("/pipeline/train", response_model=TrainingResponse)
async def train_pipeline(file: UploadFile = File(...)):
    """Train the pipeline with historical data from CSV file"""
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not loaded. Please load a pipeline first.")
    
    try:
        contents = await file.read()
        df_history = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        pipeline.fit_from_raw(df_history)
        return TrainingResponse(success=True, message="Pipeline trained successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/pipeline/train/json", response_model=TrainingResponse)
async def train_pipeline_json(data: List[Dict[str, Any]]):
    """Train the pipeline with historical data from JSON"""
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not loaded. Please load a pipeline first.")
    
    try:
        df_history = pd.DataFrame(data)
        pipeline.fit_from_raw(df_history)
        return TrainingResponse(success=True, message="Pipeline trained successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the trained pipeline"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not loaded")
    
    if not pipeline.fitted:
        raise HTTPException(status_code=400, detail="Pipeline not trained. Please train the pipeline first.")
    
    try:
        df_history = pd.DataFrame(request.history_data)
        df_new = pd.DataFrame(request.new_data)
        predictions = pipeline.predict_from_raw(df_history, df_new)
        
        # Convert numpy types to Python types for JSON serialization
        predictions_list = [int(pred) for pred in predictions]
        
        return PredictionResponse(
            predictions=predictions_list,
            success=True,
            message="Predictions generated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/proba", response_model=ProbabilityResponse)
async def predict_proba(request: PredictionRequest):
    """Get prediction probabilities using the trained pipeline"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not loaded")
    
    if not pipeline.fitted:
        raise HTTPException(status_code=400, detail="Pipeline not trained. Please train the pipeline first.")
    
    try:
        df_history = pd.DataFrame(request.history_data)
        df_new = pd.DataFrame(request.new_data)
        probabilities = pipeline.predict_proba_from_raw(df_history, df_new)
        
        # Convert numpy array to list for JSON serialization
        probabilities_list = probabilities.tolist()
        
        return ProbabilityResponse(
            probabilities=probabilities_list,
            success=True,
            message="Probabilities generated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Probability prediction failed: {str(e)}")

@app.post("/predict/csv")
async def predict_csv(
    history_file: UploadFile = File(...),
    new_data_file: UploadFile = File(...)
):
    """Make predictions using CSV files"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not loaded")
    
    if not pipeline.fitted:
        raise HTTPException(status_code=400, detail="Pipeline not trained. Please train the pipeline first.")
    
    try:
        # Read CSV files
        history_contents = await history_file.read()
        new_data_contents = await new_data_file.read()
        
        df_history = pd.read_csv(io.StringIO(history_contents.decode('utf-8')))
        df_new = pd.read_csv(io.StringIO(new_data_contents.decode('utf-8')))
        
        predictions = pipeline.predict_from_raw(df_history, df_new)
        predictions_list = [int(pred) for pred in predictions]
        
        return {
            "predictions": predictions_list,
            "success": True,
            "message": "Predictions generated successfully from CSV files"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")

@app.get("/pipeline/save")
async def save_pipeline():
    """Save the current pipeline to a pickle file (returns the file)"""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="No pipeline to save")
    
    try:
        from fastapi.responses import Response
        pipeline_bytes = pickle.dumps(pipeline)
        return Response(
            content=pipeline_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=f1_pipeline.pkl"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save pipeline: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)