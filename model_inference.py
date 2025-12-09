"""
Model Inference Module
Handles loading and using trained models for predictions and relationship detection.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from sklearn.pipeline import Pipeline
import joblib
import json

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


class ModelInference:
    """Handle inference using trained models."""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_metadata = {}
    
    def load_model(self, model_name: str) -> Optional[Pipeline]:
        """Load a trained model."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            return None
        
        model = joblib.load(model_path)
        self.loaded_models[model_name] = model
        
        # Load metadata
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_metadata[model_name] = json.load(f)
        
        return model
    
    def predict_relationship(
        self,
        df: pd.DataFrame,
        model_name: str = "knn_relationship_model",
        n_neighbors: int = 5,
        max_distance: float = 2.0
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Predict relationships using trained KNN model.
        
        Args:
            df: Input dataframe
            model_name: Name of the trained model
            n_neighbors: Number of neighbors to find
            max_distance: Maximum distance threshold
        
        Returns:
            Edges dataframe, distances array, indices array
        """
        model = self.load_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found. Please train the model first.")
        
        cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
        X = df[NUMERIC_FEATURES + cat_features]
        
        # Preprocess
        X_processed = model.named_steps["preprocess"].transform(X)
        
        # Find neighbors
        distances, indices = model.named_steps["knn"].kneighbors(
            X_processed,
            n_neighbors=min(n_neighbors + 1, len(df))
        )
        
        # Create edges
        account_ids = df["account_id"].values
        edges = []
        seen_pairs = set()
        
        anchor_map = dict(zip(df["account_id"], df.get("anchor_group", "Independent")))
        anchor_level_map = dict(zip(df["account_id"], df.get("anchor_level", 2)))
        
        for i, (d_row, idx_row) in enumerate(zip(distances, indices)):
            src_acc = account_ids[i]
            src_anchor = anchor_map.get(src_acc, "Independent")
            src_level = anchor_level_map.get(src_acc, 2)
            
            for dist, j in zip(d_row[1:], idx_row[1:]):
                dst_acc = account_ids[j]
                dst_anchor = anchor_map.get(dst_acc, "Independent")
                dst_level = anchor_level_map.get(dst_acc, 2)
                
                edge_key = tuple(sorted([src_acc, dst_acc]))
                if edge_key in seen_pairs:
                    continue
                seen_pairs.add(edge_key)
                
                if dist <= max_distance:
                    similarity = float(np.exp(-dist))
                    is_cross_anchor = (
                        (src_anchor != dst_anchor) and
                        (src_anchor != "Independent") and
                        (dst_anchor != "Independent")
                    )
                    
                    edges.append({
                        "src": src_acc,
                        "dst": dst_acc,
                        "distance": float(dist),
                        "similarity": similarity,
                        "src_anchor": src_anchor,
                        "dst_anchor": dst_anchor,
                        "is_cross_anchor": is_cross_anchor,
                        "anchor_level_diff": abs(src_level - dst_level),
                    })
        
        return pd.DataFrame(edges), distances, indices
    
    def predict_target(
        self,
        df: pd.DataFrame,
        target_column: str = "lead_score_bri",
        model_name: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict target variable using trained predictive model.
        
        Args:
            df: Input dataframe
            target_column: Target variable to predict
            model_name: Name of the model (auto-detected if None)
        
        Returns:
            Predictions array and confidence metrics
        """
        if model_name is None:
            model_name = f"predictive_{target_column}"
        
        model = self.load_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found. Please train the model first.")
        
        # Get feature columns from metadata
        metadata = self.model_metadata.get(model_name, {})
        feature_names = metadata.get("feature_names", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        
        # Remove target from features if present
        feature_names = [f for f in feature_names if f != target_column]
        
        # Prepare features
        X = df[feature_names].copy()
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0] if len(X) > 0 else X)
        
        # Predict
        predictions = model.predict(X)
        
        # Calculate confidence (for tree-based models)
        confidence = {}
        if hasattr(model.named_steps["model"], "predict_proba"):
            # For classification models
            proba = model.named_steps["model"].predict_proba(X)
            confidence["mean_confidence"] = np.mean(np.max(proba, axis=1))
        elif hasattr(model.named_steps["model"], "estimators_"):
            # For ensemble models, use prediction variance as uncertainty
            if hasattr(model.named_steps["model"], "estimators_"):
                individual_predictions = np.array([
                    est.predict(model.named_steps["preprocess"].transform(X))
                    for est in model.named_steps["model"].estimators_[:10]  # Sample for speed
                ])
                prediction_std = np.std(individual_predictions, axis=0)
                confidence["mean_uncertainty"] = float(np.mean(prediction_std))
                confidence["std_uncertainty"] = float(np.std(prediction_std))
        
        return predictions, confidence
    
    def batch_predict(
        self,
        df: pd.DataFrame,
        predictions: List[str] = None
    ) -> pd.DataFrame:
        """
        Make multiple predictions on a dataframe.
        
        Args:
            df: Input dataframe
            predictions: List of target columns to predict
        
        Returns:
            Dataframe with predictions added
        """
        if predictions is None:
            predictions = ["lead_score_bri", "turnover_90d", "avg_txn_amount_30d"]
        
        result_df = df.copy()
        
        for target in predictions:
            try:
                pred_values, confidence = self.predict_target(df, target_column=target)
                result_df[f"predicted_{target}"] = pred_values
                result_df[f"prediction_confidence_{target}"] = confidence.get("mean_uncertainty", 0.0)
            except Exception as e:
                print(f"Error predicting {target}: {e}")
        
        return result_df
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a trained model."""
        if model_name not in self.model_metadata:
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
        
        return self.model_metadata.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        models = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith('.pkl') and not file.endswith('_metadata.json'):
                    model_name = file.replace('.pkl', '')
                    models.append(model_name)
        return models


def infer_from_model(
    df: pd.DataFrame,
    model_name: str,
    inference_type: str = "relationship"
) -> pd.DataFrame:
    """
    Convenience function for inference.
    
    Args:
        df: Input dataframe
        model_name: Name of the model
        inference_type: Type of inference ('relationship' or 'prediction')
    
    Returns:
        Results dataframe
    """
    inference = ModelInference()
    
    if inference_type == "relationship":
        edges_df, _, _ = inference.predict_relationship(df, model_name=model_name)
        return edges_df
    elif inference_type == "prediction":
        result_df = inference.batch_predict(df)
        return result_df
    else:
        raise ValueError(f"Unknown inference type: {inference_type}")

