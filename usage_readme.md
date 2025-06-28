# F1 Prediction API

A FastAPI-based REST API for F1 race predictions using machine learning pipelines.

## Features

- Load pre-trained ML pipelines
- Train pipelines with historical data
- Make predictions on new data
- Get prediction probabilities
- Support for both JSON and CSV data formats
- Health checks and status monitoring
- Dockerized deployment

## Quick Start

### Using Docker Compose (Recommended)

1. Build and run the container:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`
3. Access the interactive documentation at `http://localhost:8000/docs`

### Using Docker

1. Build the image:
```bash
docker build -t f1-prediction-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 f1-prediction-api
```

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

## API Endpoints

### Health & Status
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /pipeline/status` - Check pipeline status

### Pipeline Management
- `POST /pipeline/load` - Load a pre-trained pipeline from pickle file
- `POST /pipeline/train` - Train pipeline with CSV data
- `POST /pipeline/train/json` - Train pipeline with JSON data
- `GET /pipeline/save` - Download current pipeline as pickle file

### Predictions
- `POST /predict` - Make predictions with JSON data
- `POST /predict/proba` - Get prediction probabilities with JSON data
- `POST /predict/csv` - Make predictions with CSV files

## Usage Examples

### 1. Load a Pre-trained Pipeline
```bash
curl -X POST "http://localhost:8000/pipeline/load" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_pipeline.pkl"
```

### 2. Train the Pipeline
```bash
curl -X POST "http://localhost:8000/pipeline/train" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@historical_data.csv"
```

### 3. Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "history_data": [
      {"driver": "Hamilton", "team": "Mercedes", "podium": 1, "year": 2023},
      {"driver": "Verstappen", "team": "Red Bull", "podium": 1, "year": 2023}
    ],
    "new_data": [
      {"driver": "Hamilton", "team": "Mercedes", "year": 2024},
      {"driver": "Verstappen", "team": "Red Bull", "year": 2024}
    ]
  }'
```

### 4. Get Prediction Probabilities
```bash
curl -X POST "http://localhost:8000/predict/proba" \
  -H "Content-Type: application/json" \
  -d '{
    "history_data": [...],
    "new_data": [...]
  }'
```

## Data Format

### Required Columns
Your data should include columns that your preprocessor expects. Common columns might include:
- `podium` (target variable for training)
- `year`
- Driver information
- Team information
- Race conditions
- Performance metrics

### JSON Format
```json
{
  "history_data": [
    {
      "column1": "value1",
      "column2": "value2",
      "podium": 1,
      "year": 2023
    }
  ],
  "new_data": [
    {
      "column1": "value1",
      "column2": "value2",
      "year": 2024
    }
  ]
}
```

## Docker Configuration

### Environment Variables
- `PYTHONPATH`: Set to `/app` for proper module imports
- `PYTHONDONTWRITEBYTECODE`: Prevents Python from writing pyc files
- `PYTHONUNBUFFERED`: Ensures stdout/stderr are unbuffered

### Volumes
- `./models:/app/models` - Mount local models directory for persistent storage

### Health Checks
The container includes health checks that verify the API is responding correctly.

## Development

### Adding Custom Preprocessors
1. Import your preprocessor in `main.py`
2. Ensure your preprocessor has `fit()` and `transform()` methods
3. Make sure it sets `surviving_indices_` attribute

### Error Handling
The API includes comprehensive error handling for:
- Missing pipeline
- Untrained pipeline
- Invalid data formats
- Processing errors

### CORS
CORS is enabled for all origins. Modify the CORS settings in `main.py` for production use.

## Production Considerations

1. **Security**: 
   - Disable CORS for all origins
   - Add authentication/authorization
   - Use HTTPS

2. **Performance**:
   - Consider using gunicorn with multiple workers
   - Add caching for frequently used models
   - Implement request rate limiting

3. **Monitoring**:
   - Add logging
   - Integrate with monitoring tools
   - Set up alerts for health checks

4. **Storage**:
   - Use persistent volumes for model storage
   - Consider cloud storage for large models