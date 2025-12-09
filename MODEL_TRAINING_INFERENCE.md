# Model Training & Inference Guide

## Overview

The KNN Dashboard now includes comprehensive model training and inference capabilities, along with data simulation and advanced analytics features.

## New Features

### 1. Model Training (`model_training.py`)

Train and save ML models for:
- **KNN Relationship Models**: For detecting behavioral relationships between accounts
- **Predictive Models**: For forecasting target variables (lead scores, turnover, transaction amounts)
- **Full Model Suite**: Train all models at once

#### Key Functions

- `ModelTrainer.train_knn_model()`: Train KNN model for relationship detection
- `ModelTrainer.train_predictive_model()`: Train predictive models (Random Forest or Gradient Boosting)
- `ModelTrainer.train_full_model_suite()`: Train complete set of models
- `ModelTrainer.load_model()`: Load saved models from disk
- `ModelTrainer.list_available_models()`: List all trained models

#### Usage Example

```python
from model_training import ModelTrainer

trainer = ModelTrainer()

# Train KNN model
pipeline, cat_features = trainer.train_knn_model(
    df, n_neighbors=6, model_name="knn_model", save_model=True
)

# Train predictive model
model, metrics = trainer.train_predictive_model(
    df, target_column="lead_score_bri",
    model_type="random_forest", model_name="lead_score_model",
    save_model=True
)
```

### 2. Model Inference (`model_inference.py`)

Use trained models to make predictions and detect relationships.

#### Key Functions

- `ModelInference.predict_relationship()`: Detect relationships using trained KNN model
- `ModelInference.predict_target()`: Predict target variables using trained models
- `ModelInference.batch_predict()`: Make multiple predictions at once
- `ModelInference.load_model()`: Load models for inference

#### Usage Example

```python
from model_inference import ModelInference

inference = ModelInference()

# Predict relationships
edges_df, distances, indices = inference.predict_relationship(
    df, model_name="knn_relationship_model",
    n_neighbors=5, max_distance=2.0
)

# Predict target variable
predictions, confidence = inference.predict_target(
    df, target_column="lead_score_bri"
)
```

### 3. Data Simulation (`data_simulation.py`)

Generate synthetic data for testing and development:
- **Internal Data**: Simulate BRI's own customer data with historical trends
- **External Data**: Simulate data from other banks, public registries, credit bureaus
- **Combined Data**: Mix of internal and external data sources

#### Key Functions

- `DataSimulator.simulate_internal_data()`: Generate internal BRI data
- `DataSimulator.simulate_external_data()`: Generate external data
- `DataSimulator.simulate_combined_dataset()`: Generate combined dataset
- `DataSimulator.add_noise_to_data()`: Add realistic noise
- `DataSimulator.simulate_missing_data()`: Simulate missing data patterns

#### Usage Example

```python
from data_simulation import DataSimulator

simulator = DataSimulator()

# Simulate internal data
internal_df = simulator.simulate_internal_data(
    n_accounts=100, include_historical=True, historical_months=6
)

# Simulate external data
external_df = simulator.simulate_external_data(
    n_accounts=50, data_source="external_bank", include_temporal=True
)

# Combined dataset
combined_df = simulator.simulate_combined_dataset(
    n_internal=80, n_external=50
)
```

### 4. Advanced Analytics (`advanced_analytics.py`)

Comprehensive analytics capabilities:
- **Anomaly Detection**: Identify unusual account patterns using Isolation Forest
- **Risk Scoring**: Calculate comprehensive risk scores for accounts
- **Growth Analysis**: Analyze growth trends and metrics
- **Forecasting**: Predict future turnover and opportunity scores
- **Segment Analysis**: Analyze segments by role, anchor group, etc.

#### Key Functions

- `AdvancedAnalytics.detect_anomalies()`: Detect anomalous accounts
- `AdvancedAnalytics.risk_scoring()`: Calculate risk scores
- `AdvancedAnalytics.calculate_growth_metrics()`: Analyze growth trends
- `AdvancedAnalytics.predict_future_turnover()`: Forecast turnover
- `AdvancedAnalytics.opportunity_forecasting()`: Forecast opportunity scores
- `AdvancedAnalytics.create_analytics_dashboard_data()`: Comprehensive analytics

#### Usage Example

```python
from advanced_analytics import AdvancedAnalytics

analytics = AdvancedAnalytics()

# Detect anomalies
df_anomalies = analytics.detect_anomalies(df, contamination=0.1)

# Calculate risk scores
df_risk = analytics.risk_scoring(df)

# Forecast turnover
forecast = analytics.predict_future_turnover(
    df, account_id="ACC_001", months_ahead=6, method="trend"
)

# Comprehensive analysis
results = analytics.create_analytics_dashboard_data(df)
```

## Dashboard Integration

All features are integrated into the main dashboard with new tabs:

### Tab 7: Model Training
- Train KNN relationship models
- Train predictive models for different targets
- Train full model suite
- View available trained models and metadata

### Tab 8: Model Inference
- Use trained models for relationship detection
- Make predictions on target variables
- Batch predictions for multiple targets
- View prediction results and confidence metrics

### Tab 9: Data Simulation
- Generate internal (BRI) data
- Generate external data from various sources
- Generate combined datasets
- Add noise and missing data patterns
- Download simulated data

### Tab 10: Advanced Analytics
- Comprehensive analytics dashboard
- Anomaly detection with severity levels
- Risk scoring with risk levels
- Growth analysis and trends
- Forecasting for individual accounts
- Segment-level analysis

## Model Storage

Trained models are saved in the `trained_models/` directory:
- Model files: `{model_name}.pkl` (using joblib)
- Metadata files: `{model_name}_metadata.json`

Metadata includes:
- Model type and configuration
- Training date and time
- Feature names
- Performance metrics (for predictive models)
- Number of training samples

## Best Practices

1. **Model Training**:
   - Use sufficient data (recommended: 100+ accounts)
   - Split data for predictive models (default: 80/20)
   - Use cross-validation for model evaluation
   - Save models with descriptive names

2. **Model Inference**:
   - Load models before inference
   - Ensure input data has required features
   - Handle missing values appropriately
   - Check prediction confidence/uncertainty

3. **Data Simulation**:
   - Use realistic parameters
   - Include temporal data for time-series analysis
   - Add appropriate noise levels
   - Validate simulated data distributions

4. **Advanced Analytics**:
   - Adjust contamination for anomaly detection
   - Customize risk scoring weights
   - Use appropriate forecasting methods
   - Validate analytics results

## Dependencies

All modules use existing dependencies:
- scikit-learn (for models and preprocessing)
- pandas, numpy (for data handling)
- joblib (for model persistence)
- plotly (for visualizations in dashboard)

No additional dependencies required beyond the existing project setup.

## Example Workflow

1. **Train Models**:
   ```python
   trainer = ModelTrainer()
   trainer.train_full_model_suite(df, n_neighbors=6)
   ```

2. **Use Trained Models**:
   ```python
   inference = ModelInference()
   edges_df, _, _ = inference.predict_relationship(df, model_name="knn_relationship_model")
   predictions, _ = inference.predict_target(df, target_column="lead_score_bri")
   ```

3. **Simulate Data**:
   ```python
   simulator = DataSimulator()
   new_data = simulator.simulate_combined_dataset(n_internal=100, n_external=50)
   ```

4. **Run Analytics**:
   ```python
   analytics = AdvancedAnalytics()
   results = analytics.create_analytics_dashboard_data(df)
   ```

## Notes

- Models are saved locally in `trained_models/` directory
- Model metadata is stored in JSON format for easy inspection
- All modules support both internal and external data sources
- Advanced analytics can work with or without historical data
- Simulation supports realistic banking and ecosystem patterns

