# Psyk-Proxy: Mental Health Prediction API

A FastAPI backend service that provides mental health care option predictions using a pre-trained HistGradientBoostingClassifier model.

## Features

- **Model Inference**: Uses pre-trained ML models for mental health care option prediction
- **Data Processing**: Polars-based preprocessing with categorical encoding
- **Async MongoDB**: Stores prediction results with user tracking
- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Pydantic Validation**: Robust input validation and response formatting
- **Clean Architecture**: Modular design with separation of concerns

## Tech Stack

- **Backend**: FastAPI
- **Database**: MongoDB (Async PyMongo driver)
- **Data Processing**: Polars (primary), Pandas (for encoder compatibility)
- **Model Loading**: joblib
- **Config Management**: Pydantic BaseSettings
- **Schema Validation**: Pydantic models

## Project Structure

```
app/
├── main.py                  # FastAPI app entrypoint
├── model/                   # Pre-trained ML models
│   ├── model.pkl
│   ├── encoder.pkl
│   ├── features.pkl
│   ├── targets.pkl
│   └── model_package.pkl
├── api/
│   └── predict.py          # /predict endpoint
├── core/
│   ├── config.py           # Pydantic BaseSettings for env vars
│   └── model_loader.py     # Load model, encoder, features, targets
├── db/
│   └── mongodb.py          # Async PyMongo connection
├── models/
│   ├── request.py          # Pydantic request schema
│   └── response.py         # Pydantic response schema
├── services/
│   ├── preprocessing.py    # Data preprocessing using polars
│   └── prediction.py       # Model inference logic
└── utils/
    └── logger.py           # Environment aware logging helper
```

## API Endpoints

### POST `/api/v1/predict`

Accepts user mental health data and returns predicted care option.

**Request Body:**
```json
{
  "user_id": "user_123456",
  "gender": 1,
  "country": "USA",
  "occupation": "Student",
  "family_history": 0,
  "treatment": 1,
  "days_indoors": "1-14 days",
  "stress": "Low",
  "mental_health_history": "No",
  "mood_swings": "Rarely",
  "coping_struggles": 0,
  "work_interest": "Same",
  "social_anxiety": "Never",
  "consult_history": "No"
}
```

**Response:**
```json
{
  "prediction": "care_options_Yes",
  "confidence": 0.91
}
```

### GET `/api/v1/predict/history/{user_id}`

Retrieves prediction history for a specific user.

**Response:**
```json
{
  "user_id": "user_123456",
  "prediction_count": 5,
  "predictions": [...]
}
```

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd psyk-proxy
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB connection string and other settings
   ```

4. **Start MongoDB**
   Make sure MongoDB is running on your system or update the connection string in `.env`.

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `False` |
| `HOST` | Host to bind the server | `0.0.0.0` |
| `PORT` | Port to bind the server | `8000` |
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017` |
| `DATABASE_NAME` | Database name | `psyk_db` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `["*"]` |

## Model Information

The service uses a pre-trained HistGradientBoostingClassifier with the following components:

- **model.pkl**: Main MultiOutputClassifier with HistGradientBoostingClassifier
- **encoder.pkl**: LeaveOneOutEncoder for country encoding
- **features.pkl**: Feature names in correct order for model input
- **targets.pkl**: Target names for prediction output
- **model_package.pkl**: Complete model package with performance metrics

## Data Processing Pipeline

1. **Input Validation**: Pydantic schema validation
2. **Categorical Encoding**: One-hot encoding for categorical features
3. **Country Encoding**: LeaveOneOut encoding for country feature
4. **Feature Alignment**: Reorder features to match training data
5. **Prediction**: Multi-output classification with confidence scores
6. **Storage**: Async MongoDB storage with user tracking

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black app/
isort app/
```

### Linting
```bash
flake8 app/
```

## API Documentation

When the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Health Check

Check if the service is running:
```bash
curl http://localhost:8000/health
```

## Error Handling

The API provides detailed error responses with appropriate HTTP status codes:
- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Server-side errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]
