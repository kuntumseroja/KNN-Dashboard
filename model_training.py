"""
Model Training Module
Handles training, saving, and loading of ML models for the KNN Dashboard.
Supports both KNN-based relationship models and predictive models.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Feature definitions
NUMERIC_FEATURES = [
    "avg_txn_amount_30d",
    "txn_count_30d",
    "turnover_90d",
    "cash_withdrawal_ratio_90d",
    "merchant_diversity_90d",
    "unique_devices_90d",
    "unique_ips_90d",
    "country_risk_score",
    "channel_mix_online_ratio",
    "lead_score_bri",
]

CATEGORICAL_FEATURES = ["segment_code", "ecosystem_role"]

MODELS_DIR = "trained_models"
os.makedirs(MODELS_DIR, exist_ok=True)


class ModelTrainer:
    """Train and manage ML models for relationship detection and predictions."""
    
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.trained_models = {}
        self.model_metadata = {}
    
    def train_knn_model(
        self,
        df: pd.DataFrame,
        n_neighbors: int = 6,
        model_name: str = "knn_relationship_model",
        save_model: bool = True
    ) -> Tuple[Pipeline, List[str]]:
        """
        Train KNN model for relationship detection.
        
        Args:
            df: Training dataframe
            n_neighbors: Number of neighbors for KNN
            model_name: Name to save the model
            save_model: Whether to save the model to disk
        
        Returns:
            Trained pipeline and categorical features list
        """
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        
        cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, cat_features),
            ]
        )
        
        knn = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(df)),
            metric="euclidean",
            algorithm="auto"
        )
        
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("knn", knn)
        ])
        
        X = df[NUMERIC_FEATURES + cat_features]
        pipeline.fit(X)
        
        # Store model metadata
        metadata = {
            "model_type": "knn_relationship",
            "n_neighbors": n_neighbors,
            "n_samples": len(df),
            "n_features": len(NUMERIC_FEATURES) + len(cat_features),
            "trained_at": datetime.now().isoformat(),
            "feature_names": NUMERIC_FEATURES + cat_features,
        }
        
        if save_model:
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            
            joblib.dump(pipeline, model_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.model_metadata[model_name] = metadata
        
        self.trained_models[model_name] = pipeline
        return pipeline, cat_features
    
    def train_predictive_model(
        self,
        df: pd.DataFrame,
        target_column: str = "lead_score_bri",
        model_type: str = "random_forest",
        model_name: str = "predictive_model",
        test_size: float = 0.2,
        save_model: bool = True
    ) -> Tuple[object, Dict]:
        """
        Train predictive model for forecasting target variables.
        
        Args:
            df: Training dataframe
            target_column: Column to predict
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            model_name: Name to save the model
            test_size: Proportion of data for testing
            save_model: Whether to save the model to disk
        
        Returns:
            Trained model and performance metrics
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Prepare features
        cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
        feature_cols = NUMERIC_FEATURES + cat_features
        
        # Remove target from features if present
        feature_cols = [f for f in feature_cols if f != target_column and f in df.columns]
        
        if len(feature_cols) == 0:
            raise ValueError(f"No valid features found. Required columns: {NUMERIC_FEATURES + CATEGORICAL_FEATURES}")
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Ensure target is numeric for regression
        if not pd.api.types.is_numeric_dtype(y):
            try:
                y = pd.to_numeric(y, errors='coerce')
            except:
                raise ValueError(f"Target column '{target_column}' must be numeric for regression models.")
        
        # Handle missing values - separate numeric and categorical
        for col in X.columns:
            if X[col].dtype in [np.int64, np.float64, 'int64', 'float64']:
                # Numeric columns: fill with mean
                X[col] = X[col].fillna(X[col].mean())
            else:
                # Categorical columns: fill with mode
                mode_value = X[col].mode()
                if len(mode_value) > 0:
                    X[col] = X[col].fillna(mode_value.iloc[0])
                else:
                    # If no mode, use first non-null value or default
                    first_valid = X[col].dropna()
                    if len(first_valid) > 0:
                        X[col] = X[col].fillna(first_valid.iloc[0])
        
        # Handle target variable missing values
        if pd.api.types.is_numeric_dtype(y):
            y = y.fillna(y.mean())
        else:
            mode_value = y.mode()
            if len(mode_value) > 0:
                y = y.fillna(mode_value.iloc[0])
        
        # Preprocessing
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        
        numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns and c != target_column]
        cat_cols = [c for c in cat_features if c in X.columns]
        
        # Only include columns that exist and are not empty
        numeric_cols = [c for c in numeric_cols if c in X.columns and X[c].notna().any()]
        cat_cols = [c for c in cat_cols if c in X.columns and X[c].notna().any()]
        
        transformers = []
        if len(numeric_cols) > 0:
            transformers.append(("num", numeric_transformer, numeric_cols))
        if len(cat_cols) > 0:
            transformers.append(("cat", categorical_transformer, cat_cols))
        
        if len(transformers) == 0:
            raise ValueError("No valid features found for training. Check that required columns exist in the dataframe.")
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"  # Drop any remaining columns instead of passthrough
        )
        
        # Select model
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                learning_rate=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        metrics = {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        metrics["cv_r2_mean"] = cv_scores.mean()
        metrics["cv_r2_std"] = cv_scores.std()
        
        # Store metadata
        metadata = {
            "model_type": model_type,
            "target_column": target_column,
            "trained_at": datetime.now().isoformat(),
            "feature_names": feature_cols,
            "metrics": metrics,
        }
        
        if save_model:
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            
            joblib.dump(pipeline, model_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.model_metadata[model_name] = metadata
        
        self.trained_models[model_name] = pipeline
        return pipeline, metrics
    
    def load_model(self, model_name: str) -> Optional[Pipeline]:
        """Load a trained model from disk."""
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(model_path):
            return None
        
        model = joblib.load(model_path)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.model_metadata[model_name] = metadata
        
        self.trained_models[model_name] = model
        return model
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a trained model."""
        if model_name in self.model_metadata:
            return self.model_metadata[model_name]
        
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.model_metadata[model_name] = metadata
            return metadata
        
        return None
    
    def list_available_models(self) -> List[str]:
        """List all available trained models."""
        models = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith('.pkl') and not file.endswith('_metadata.json'):
                    model_name = file.replace('.pkl', '')
                    models.append(model_name)
        return models
    
    def train_full_model_suite(
        self,
        df: pd.DataFrame,
        n_neighbors: int = 6,
        save_models: bool = True
    ) -> Dict[str, object]:
        """
        Train a complete suite of models for the dashboard.
        
        Returns:
            Dictionary of trained models
        """
        models = {}
        
        # Train KNN relationship model
        knn_pipeline, cat_features = self.train_knn_model(
            df, n_neighbors=n_neighbors,
            model_name="knn_relationship_model",
            save_model=save_models
        )
        models["knn_relationship"] = knn_pipeline
        
        # Train predictive models for different targets
        targets = ["lead_score_bri", "turnover_90d", "avg_txn_amount_30d"]
        
        for target in targets:
            if target in df.columns:
                try:
                    model, metrics = self.train_predictive_model(
                        df,
                        target_column=target,
                        model_type="random_forest",
                        model_name=f"predictive_{target}",
                        save_model=save_models
                    )
                    models[f"predictive_{target}"] = model
                except Exception as e:
                    # Silently skip targets that can't be trained
                    pass
        
        return models


def train_from_data(
    df: pd.DataFrame,
    model_type: str = "knn",
    **kwargs
) -> Tuple[object, Dict]:
    """
    Convenience function to train models from data.
    
    Args:
        df: Training dataframe
        model_type: Type of model to train ('knn' or 'predictive')
        **kwargs: Additional arguments for training
    
    Returns:
        Trained model and metadata
    """
    trainer = ModelTrainer()
    
    if model_type == "knn":
        pipeline, cat_features = trainer.train_knn_model(df, **kwargs)
        metadata = trainer.get_model_metadata(kwargs.get("model_name", "knn_relationship_model"))
        return pipeline, metadata
    elif model_type == "predictive":
        model, metrics = trainer.train_predictive_model(df, **kwargs)
        metadata = trainer.get_model_metadata(kwargs.get("model_name", "predictive_model"))
        return model, metadata
    else:
        raise ValueError(f"Unknown model type: {model_type}")

