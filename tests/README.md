# Test Files

This directory contains the test suite for the psyk-proxy FastAPI application.

## Test Structure

- `conftest.py` - Test configuration and shared fixtures
- `test_api.py` - API endpoint tests
- `test_preprocessing.py` - Data preprocessing service tests
- `test_prediction.py` - Prediction service tests
- `test_model_loader.py` - Model loader tests
- `test_models.py` - Pydantic model validation tests
- `test_integration.py` - End-to-end integration tests

## Running Tests

### Option 1: Direct pytest
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run with verbose output
pytest -v
```

### Option 2: Using the test runner script
```bash
python run_tests.py
```

## Test Dependencies

Make sure you have the following test dependencies installed:

```bash
pip install pytest pytest-asyncio httpx pytest-cov
```

Or install all dependencies:
```bash
pip install -e ".[dev]"
```

## Test Categories

### Unit Tests
- `test_preprocessing.py` - Tests data preprocessing logic
- `test_prediction.py` - Tests prediction service logic
- `test_model_loader.py` - Tests model loading functionality
- `test_models.py` - Tests Pydantic model validation

### Integration Tests
- `test_api.py` - Tests API endpoints with mocked dependencies
- `test_integration.py` - Tests complete workflows

## Test Coverage

The tests aim to cover:
- ✅ All API endpoints
- ✅ Input validation
- ✅ Data preprocessing pipeline
- ✅ Model inference logic
- ✅ Database operations (mocked)
- ✅ Error handling
- ✅ Response formatting

## Mocking Strategy

Tests use mocking for:
- ML models and encoders
- Database operations (MongoDB)
- External dependencies
- File system operations

This ensures tests run quickly and don't require actual ML models or database connections.

## Writing New Tests

When adding new features, please add corresponding tests:

1. **Unit tests** for individual functions/classes
2. **Integration tests** for complete workflows
3. **API tests** for new endpoints

Follow the existing patterns in the test files for consistency.
