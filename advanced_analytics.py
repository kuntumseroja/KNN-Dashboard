"""
Advanced Analytics Module
Provides advanced analytics capabilities including predictions, forecasting,
anomaly detection, and trend analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px


class AdvancedAnalytics:
    """Advanced analytics and predictions."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def predict_future_turnover(
        self,
        df: pd.DataFrame,
        account_id: str,
        months_ahead: int = 3,
        method: str = "trend"
    ) -> Dict:
        """
        Predict future turnover for an account.
        
        Args:
            df: Historical data
            account_id: Account to predict
            months_ahead: Number of months to forecast
            method: Prediction method ('trend', 'moving_average', 'exponential')
        
        Returns:
            Dictionary with predictions and confidence intervals
        """
        account_data = df[df["account_id"] == account_id]
        if len(account_data) == 0:
            return {"error": "Account not found"}
        
        # Extract historical turnover if available
        historical_cols = [col for col in df.columns if "turnover" in col.lower() and "month" in col.lower()]
        
        if len(historical_cols) > 0:
            historical_values = []
            for col in sorted(historical_cols):
                val = account_data[col].iloc[0] if col in account_data.columns else None
                if val and not pd.isna(val):
                    historical_values.append(float(val))
            
            if len(historical_values) >= 2:
                if method == "trend":
                    # Linear trend
                    x = np.arange(len(historical_values))
                    coeffs = np.polyfit(x, historical_values, 1)
                    trend = np.poly1d(coeffs)
                    
                    future_x = np.arange(len(historical_values), len(historical_values) + months_ahead)
                    predictions = trend(future_x)
                    
                    # Calculate confidence intervals
                    residuals = historical_values - trend(x)
                    std_error = np.std(residuals)
                    confidence_lower = predictions - 1.96 * std_error
                    confidence_upper = predictions + 1.96 * std_error
                
                elif method == "moving_average":
                    # Moving average
                    window = min(3, len(historical_values))
                    ma = np.mean(historical_values[-window:])
                    growth_rate = (historical_values[-1] / historical_values[-2] - 1) if len(historical_values) >= 2 else 0
                    
                    predictions = []
                    for i in range(months_ahead):
                        predictions.append(ma * (1 + growth_rate) ** (i + 1))
                    predictions = np.array(predictions)
                    
                    std_error = np.std(historical_values) * 0.1
                    confidence_lower = predictions - 1.96 * std_error
                    confidence_upper = predictions + 1.96 * std_error
                
                else:  # exponential
                    # Exponential smoothing
                    alpha = 0.3
                    smoothed = [historical_values[0]]
                    for val in historical_values[1:]:
                        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
                    
                    last_smoothed = smoothed[-1]
                    growth_rate = (historical_values[-1] / historical_values[-2] - 1) if len(historical_values) >= 2 else 0
                    
                    predictions = []
                    for i in range(months_ahead):
                        predictions.append(last_smoothed * (1 + growth_rate) ** (i + 1))
                    predictions = np.array(predictions)
                    
                    std_error = np.std(historical_values) * 0.15
                    confidence_lower = predictions - 1.96 * std_error
                    confidence_upper = predictions + 1.96 * std_error
            else:
                # Fallback: use current turnover
                current_turnover = account_data["turnover_90d"].iloc[0]
                predictions = np.array([current_turnover] * months_ahead)
                confidence_lower = predictions * 0.8
                confidence_upper = predictions * 1.2
        else:
            # No historical data, use current value
            current_turnover = account_data["turnover_90d"].iloc[0]
            predictions = np.array([current_turnover] * months_ahead)
            confidence_lower = predictions * 0.8
            confidence_upper = predictions * 1.2
        
        return {
            "account_id": account_id,
            "predictions": predictions.tolist(),
            "confidence_lower": confidence_lower.tolist(),
            "confidence_upper": confidence_upper.tolist(),
            "method": method,
            "months_ahead": months_ahead
        }
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        contamination: float = 0.1,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Detect anomalous accounts using Isolation Forest.
        
        Args:
            df: Input dataframe
            contamination: Expected proportion of anomalies
            features: Features to use (default: numeric features)
        
        Returns:
            Dataframe with anomaly scores
        """
        if features is None:
            numeric_features = [
                "avg_txn_amount_30d",
                "txn_count_30d",
                "turnover_90d",
                "cash_withdrawal_ratio_90d",
                "merchant_diversity_90d",
            ]
            features = [f for f in numeric_features if f in df.columns]
        
        X = df[features].fillna(df[features].mean())
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        anomaly_scores = iso_forest.fit_predict(X_scaled)
        anomaly_scores_normalized = iso_forest.score_samples(X_scaled)
        
        result_df = df.copy()
        result_df["anomaly_flag"] = (anomaly_scores == -1).astype(int)
        result_df["anomaly_score"] = -anomaly_scores_normalized  # Negative scores indicate anomalies
        result_df["anomaly_severity"] = pd.cut(
            result_df["anomaly_score"],
            bins=[-np.inf, -0.5, -0.3, 0, np.inf],
            labels=["Critical", "High", "Medium", "Normal"]
        )
        
        return result_df
    
    def calculate_growth_metrics(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate growth metrics for accounts.
        
        Args:
            df: Input dataframe with temporal columns
        
        Returns:
            Dataframe with growth metrics
        """
        result_df = df.copy()
        
        # Find turnover columns
        turnover_cols = [col for col in df.columns if "turnover" in col.lower()]
        historical_turnover = [col for col in turnover_cols if "month" in col.lower() or "period" in col.lower()]
        
        if len(historical_turnover) >= 2:
            # Calculate month-over-month growth
            for i in range(len(historical_turnover) - 1):
                current_col = historical_turnover[i]
                previous_col = historical_turnover[i + 1]
                
                if current_col in df.columns and previous_col in df.columns:
                    growth_col = f"growth_mom_{i}"
                    result_df[growth_col] = (
                        (df[current_col] - df[previous_col]) / df[previous_col] * 100
                    ).fillna(0)
        
        # Calculate average growth rate
        growth_cols = [col for col in result_df.columns if "growth_mom" in col]
        if len(growth_cols) > 0:
            result_df["avg_growth_rate"] = result_df[growth_cols].mean(axis=1)
            result_df["growth_trend"] = result_df[growth_cols].apply(
                lambda x: "Increasing" if x.iloc[-1] > x.iloc[0] else "Decreasing" if len(x) > 1 else "Stable",
                axis=1
            )
        
        return result_df
    
    def segment_analysis(
        self,
        df: pd.DataFrame,
        segment_by: str = "ecosystem_role"
    ) -> Dict:
        """
        Perform segment-level analysis.
        
        Args:
            df: Input dataframe
            segment_by: Column to segment by
        
        Returns:
            Dictionary with segment statistics
        """
        if segment_by not in df.columns:
            return {"error": f"Column {segment_by} not found"}
        
        segments = df[segment_by].unique()
        segment_stats = {}
        
        for segment in segments:
            segment_data = df[df[segment_by] == segment]
            
            segment_stats[segment] = {
                "count": len(segment_data),
                "avg_turnover": segment_data["turnover_90d"].mean() if "turnover_90d" in segment_data.columns else 0,
                "avg_lead_score": segment_data["lead_score_bri"].mean() if "lead_score_bri" in segment_data.columns else 0,
                "ntb_rate": (segment_data["bri_status"] == "NTB").sum() / len(segment_data) if "bri_status" in segment_data.columns else 0,
                "avg_txn_amount": segment_data["avg_txn_amount_30d"].mean() if "avg_txn_amount_30d" in segment_data.columns else 0,
            }
        
        return segment_stats
    
    def risk_scoring(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive risk scores for accounts.
        
        Args:
            df: Input dataframe
            weights: Custom weights for risk factors
        
        Returns:
            Dataframe with risk scores
        """
        if weights is None:
            weights = {
                "country_risk": 0.25,
                "cash_ratio": 0.20,
                "device_diversity": 0.15,
                "ip_diversity": 0.15,
                "merchant_diversity": 0.15,
                "online_ratio": 0.10,
            }
        
        result_df = df.copy()
        
        # Normalize risk factors (higher = more risky)
        if "country_risk_score" in df.columns:
            country_risk = df["country_risk_score"] * 100
        else:
            country_risk = np.zeros(len(df))
        
        cash_risk = df["cash_withdrawal_ratio_90d"] * 100 if "cash_withdrawal_ratio_90d" in df.columns else np.zeros(len(df))
        
        # Lower diversity = higher risk
        device_risk = (1 - df["unique_devices_90d"] / df["unique_devices_90d"].max()) * 100 if "unique_devices_90d" in df.columns else np.zeros(len(df))
        ip_risk = (1 - df["unique_ips_90d"] / df["unique_ips_90d"].max()) * 100 if "unique_ips_90d" in df.columns else np.zeros(len(df))
        merchant_risk = (1 - df["merchant_diversity_90d"] / df["merchant_diversity_90d"].max()) * 100 if "merchant_diversity_90d" in df.columns else np.zeros(len(df))
        
        # Lower online ratio = higher risk (more cash transactions)
        online_risk = (1 - df["channel_mix_online_ratio"]) * 100 if "channel_mix_online_ratio" in df.columns else np.zeros(len(df))
        
        # Calculate weighted risk score
        risk_score = (
            country_risk * weights.get("country_risk", 0.25) +
            cash_risk * weights.get("cash_ratio", 0.20) +
            device_risk * weights.get("device_diversity", 0.15) +
            ip_risk * weights.get("ip_diversity", 0.15) +
            merchant_risk * weights.get("merchant_diversity", 0.15) +
            online_risk * weights.get("online_ratio", 0.10)
        )
        
        result_df["risk_score"] = risk_score
        result_df["risk_level"] = pd.cut(
            risk_score,
            bins=[0, 30, 50, 70, 100],
            labels=["Low", "Medium", "High", "Critical"]
        )
        
        return result_df
    
    def opportunity_forecasting(
        self,
        df: pd.DataFrame,
        forecast_months: int = 6
    ) -> pd.DataFrame:
        """
        Forecast opportunity scores for accounts.
        
        Args:
            df: Input dataframe
            forecast_months: Number of months to forecast
        
        Returns:
            Dataframe with forecasted opportunity scores
        """
        result_df = df.copy()
        
        # Simple forecasting based on current trends
        if "opportunity_score" in df.columns:
            base_scores = df["opportunity_score"]
        else:
            # Calculate base opportunity score
            base_scores = (
                df["lead_score_bri"] * 0.5 +
                (df["turnover_90d"] / df["turnover_90d"].max() * 100) * 0.3 +
                (df["bri_status"] == "NTB").astype(int) * 20
            )
        
        # Forecast future scores (assuming slight improvement for NTB accounts)
        for month in range(1, forecast_months + 1):
            forecast_col = f"forecasted_opportunity_score_month_{month}"
            
            # NTB accounts have potential for improvement
            improvement_factor = np.where(
                df["bri_status"] == "NTB",
                1 + (month * 0.02),  # 2% improvement per month
                1.0  # Existing accounts stable
            )
            
            result_df[forecast_col] = base_scores * improvement_factor
            result_df[forecast_col] = np.clip(result_df[forecast_col], 0, 100)
        
        return result_df
    
    def create_analytics_dashboard_data(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Create comprehensive analytics data for dashboard.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dictionary with all analytics results
        """
        analytics = {}
        
        # Anomaly detection
        df_anomalies = self.detect_anomalies(df)
        analytics["anomalies"] = {
            "total": len(df_anomalies[df_anomalies["anomaly_flag"] == 1]),
            "critical": len(df_anomalies[df_anomalies["anomaly_severity"] == "Critical"]),
            "high": len(df_anomalies[df_anomalies["anomaly_severity"] == "High"]),
            "data": df_anomalies[["account_id", "anomaly_flag", "anomaly_score", "anomaly_severity"]]
        }
        
        # Risk scoring
        df_risk = self.risk_scoring(df)
        analytics["risk"] = {
            "avg_risk_score": df_risk["risk_score"].mean(),
            "high_risk_count": len(df_risk[df_risk["risk_level"].isin(["High", "Critical"])]),
            "data": df_risk[["account_id", "risk_score", "risk_level"]]
        }
        
        # Growth metrics
        df_growth = self.calculate_growth_metrics(df)
        analytics["growth"] = {
            "avg_growth_rate": df_growth["avg_growth_rate"].mean() if "avg_growth_rate" in df_growth.columns else 0,
            "data": df_growth
        }
        
        # Segment analysis
        analytics["segments"] = self.segment_analysis(df)
        
        # Opportunity forecasting
        df_forecast = self.opportunity_forecasting(df)
        analytics["forecast"] = df_forecast
        
        return analytics
    
    # ==================== WHOLESALE BANKING RM ANALYTICS ====================
    
    def calculate_lead_quality_score(
        self,
        df: pd.DataFrame,
        G=None,
        opportunity_scores: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive lead quality score for wholesale banking RM.
        Combines financial metrics, network value, and relationship potential.
        
        Args:
            df: Input dataframe
            G: NetworkX graph (optional, for network metrics)
            opportunity_scores: Pre-calculated opportunity scores (optional)
        
        Returns:
            Dataframe with lead quality scores and components
        """
        import networkx as nx
        
        result_df = df.copy()
        
        # 1. Financial Quality Score (0-30 points)
        # Based on turnover, transaction volume, and growth potential
        max_turnover = df["turnover_90d"].max() if "turnover_90d" in df.columns else 1
        financial_score = (
            (df["turnover_90d"] / max_turnover * 15) +
            (df["avg_txn_amount_30d"] / df["avg_txn_amount_30d"].max() * 10) +
            (df["txn_count_30d"] / df["txn_count_30d"].max() * 5)
        ).fillna(0)
        
        # 2. Network Value Score (0-25 points)
        # Based on network position and ecosystem influence
        if G is not None and len(G.nodes()) > 0:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            network_value = (
                pd.Series([degree_centrality.get(acc, 0) for acc in df["account_id"]]) * 15 +
                pd.Series([betweenness_centrality.get(acc, 0) for acc in df["account_id"]]) * 10
            )
        else:
            network_value = pd.Series([0] * len(df))
        
        # 3. Relationship Potential Score (0-25 points)
        # NTB status, anchor proximity, ecosystem role importance
        relationship_potential = pd.Series([0] * len(df))
        if "bri_status" in df.columns:
            relationship_potential += (df["bri_status"] == "NTB").astype(int) * 15
        if "anchor_level" in df.columns:
            relationship_potential += (df["anchor_level"] <= 1).astype(int) * 10
        
        # Ecosystem role importance (Anchor > Feed Mill > Others)
        role_importance = {
            "Anchor_Corporate": 10,
            "Feed_Mill": 8,
            "Breeder_Farm": 6,
            "Slaughterhouse": 5,
            "Collector/Offtaker": 4,
            "Contract_Farmer": 3,
            "Logistics/Transport": 2,
            "Retailer/Warung": 1,
            "Input_Supplier": 2,
        }
        if "ecosystem_role" in df.columns:
            relationship_potential += df["ecosystem_role"].map(role_importance).fillna(0)
        
        # 4. Behavioral Quality Score (0-20 points)
        # Based on lead score, transaction patterns, digital adoption
        behavioral_score = (
            (df["lead_score_bri"] / 100 * 12) +
            (df["channel_mix_online_ratio"] * 5) +
            ((1 - df["cash_withdrawal_ratio_90d"]) * 3)
        ).fillna(0)
        
        # Total Lead Quality Score (0-100)
        lead_quality_score = (
            financial_score * 0.30 +
            network_value * 0.25 +
            relationship_potential * 0.25 +
            behavioral_score * 0.20
        )
        
        result_df["lead_quality_score"] = np.clip(lead_quality_score, 0, 100)
        result_df["lead_quality_grade"] = pd.cut(
            result_df["lead_quality_score"],
            bins=[0, 50, 70, 85, 100],
            labels=["C", "B", "A", "A+"]
        )
        
        # Component scores for transparency
        result_df["financial_quality"] = financial_score
        result_df["network_value"] = network_value
        result_df["relationship_potential"] = relationship_potential
        result_df["behavioral_quality"] = behavioral_score
        
        return result_df
    
    def calculate_customer_lifetime_value(
        self,
        df: pd.DataFrame,
        months: int = 36,
        discount_rate: float = 0.05
    ) -> pd.DataFrame:
        """
        Estimate Customer Lifetime Value (CLV) for wholesale banking accounts.
        
        Args:
            df: Input dataframe
            months: Projection period in months
            discount_rate: Monthly discount rate
        
        Returns:
            Dataframe with CLV estimates
        """
        result_df = df.copy()
        
        # Estimate monthly revenue (from turnover)
        monthly_revenue = df["turnover_90d"] / 3 if "turnover_90d" in df.columns else pd.Series([0] * len(df))
        
        # Estimate retention probability based on BRI status and lead score
        base_retention = 0.85  # Base monthly retention
        retention_adjustment = (
            (df["bri_status"] == "Existing").astype(int) * 0.10 +
            (df["lead_score_bri"] / 100 * 0.05)
        ).fillna(0)
        monthly_retention = np.clip(base_retention + retention_adjustment, 0.5, 0.98)
        
        # Calculate CLV
        clv = []
        for i, (revenue, retention) in enumerate(zip(monthly_revenue, monthly_retention)):
            cumulative_clv = 0
            cumulative_retention = 1.0
            
            for month in range(1, months + 1):
                # Revenue in this month (with growth assumption)
                month_revenue = revenue * (1.01 ** (month / 12))  # 1% annual growth
                
                # Discounted value
                discounted_value = month_revenue / ((1 + discount_rate) ** month)
                
                # Apply retention probability
                cumulative_retention *= retention
                clv_contribution = discounted_value * cumulative_retention
                
                cumulative_clv += clv_contribution
            
            clv.append(cumulative_clv)
        
        result_df["clv_estimate"] = clv
        result_df["clv_tier"] = pd.cut(
            result_df["clv_estimate"],
            bins=[0, 1e9, 5e9, 20e9, np.inf],
            labels=["Standard", "Premium", "Enterprise", "Strategic"]
        )
        
        return result_df
    
    def identify_cross_sell_opportunities(
        self,
        df: pd.DataFrame,
        G=None
    ) -> pd.DataFrame:
        """
        Identify cross-sell and upsell opportunities for wholesale banking.
        
        Args:
            df: Input dataframe
            G: NetworkX graph (optional)
        
        Returns:
            Dataframe with cross-sell opportunity scores
        """
        import networkx as nx
        
        result_df = df.copy()
        
        # Cross-sell opportunity indicators
        opportunities = []
        
        for idx, row in df.iterrows():
            score = 0
            reasons = []
            
            # 1. High turnover but low transaction count = potential for more products
            if "turnover_90d" in row and "txn_count_30d" in row:
                turnover_per_txn = row["turnover_90d"] / max(row["txn_count_30d"], 1)
                if turnover_per_txn > df["turnover_90d"].quantile(0.75) / max(df["txn_count_30d"].quantile(0.25), 1):
                    score += 20
                    reasons.append("High value per transaction")
            
            # 2. NTB accounts = new relationship opportunity
            if row.get("bri_status") == "NTB":
                score += 25
                reasons.append("New-to-Bank opportunity")
            
            # 3. High network value = ecosystem expansion opportunity
            if G is not None and row["account_id"] in G.nodes():
                degree = G.degree(row["account_id"])
                if degree > df["account_id"].map(lambda x: G.degree(x) if x in G.nodes() else 0).quantile(0.75):
                    score += 20
                    reasons.append("High network influence")
            
            # 4. Low digital adoption = digital product opportunity
            if row.get("channel_mix_online_ratio", 1) < 0.5:
                score += 15
                reasons.append("Digital product opportunity")
            
            # 5. High cash ratio = cash management product opportunity
            if row.get("cash_withdrawal_ratio_90d", 0) > 0.3:
                score += 10
                reasons.append("Cash management opportunity")
            
            # 6. Anchor or Feed Mill = corporate banking products
            if row.get("ecosystem_role") in ["Anchor_Corporate", "Feed_Mill"]:
                score += 10
                reasons.append("Corporate banking products")
            
            opportunities.append({
                "cross_sell_score": min(score, 100),
                "opportunity_reasons": "; ".join(reasons) if reasons else "Standard monitoring"
            })
        
        opp_df = pd.DataFrame(opportunities)
        result_df = pd.concat([result_df, opp_df], axis=1)
        
        result_df["cross_sell_priority"] = pd.cut(
            result_df["cross_sell_score"],
            bins=[0, 30, 60, 80, 100],
            labels=["Low", "Medium", "High", "Critical"]
        )
        
        return result_df
    
    def calculate_relationship_strength(
        self,
        df: pd.DataFrame,
        G=None,
        edges_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate relationship strength score for wholesale banking RM.
        
        Args:
            df: Input dataframe
            G: NetworkX graph
            edges_df: Edges dataframe with relationship data
        
        Returns:
            Dataframe with relationship strength scores
        """
        import networkx as nx
        
        result_df = df.copy()
        
        relationship_strength = pd.Series([0] * len(df))
        
        # 1. Account tenure (Existing vs NTB)
        if "bri_status" in df.columns:
            relationship_strength += (df["bri_status"] == "Existing").astype(int) * 30
        
        # 2. Transaction frequency and volume
        if "txn_count_30d" in df.columns and "avg_txn_amount_30d" in df.columns:
            normalized_frequency = (df["txn_count_30d"] / df["txn_count_30d"].max() * 20)
            normalized_volume = (df["avg_txn_amount_30d"] / df["avg_txn_amount_30d"].max() * 20)
            relationship_strength += normalized_frequency + normalized_volume
        
        # 3. Network connectivity (stronger network = stronger relationship)
        if G is not None and len(G.nodes()) > 0:
            degree_centrality = nx.degree_centrality(G)
            network_strength = pd.Series([
                degree_centrality.get(acc, 0) * 20 for acc in df["account_id"]
            ])
            relationship_strength += network_strength
        
        # 4. Relationship quality indicators
        if "lead_score_bri" in df.columns:
            relationship_strength += (df["lead_score_bri"] / 100 * 10)
        
        result_df["relationship_strength"] = np.clip(relationship_strength, 0, 100)
        result_df["relationship_tier"] = pd.cut(
            result_df["relationship_strength"],
            bins=[0, 40, 60, 80, 100],
            labels=["Weak", "Moderate", "Strong", "Strategic"]
        )
        
        return result_df
    
    def predict_churn_risk(
        self,
        df: pd.DataFrame,
        G=None
    ) -> pd.DataFrame:
        """
        Predict churn risk for wholesale banking accounts.
        
        Args:
            df: Input dataframe
            G: NetworkX graph (optional)
        
        Returns:
            Dataframe with churn risk scores
        """
        import networkx as nx
        
        result_df = df.copy()
        
        churn_risk = pd.Series([0] * len(df))
        
        # 1. Declining transaction patterns
        if "txn_count_30d" in df.columns:
            # Lower transaction count = higher risk
            normalized_txn = 1 - (df["txn_count_30d"] / df["txn_count_30d"].max())
            churn_risk += normalized_txn * 25
        
        # 2. Low lead score = potential dissatisfaction
        if "lead_score_bri" in df.columns:
            normalized_lead = 1 - (df["lead_score_bri"] / 100)
            churn_risk += normalized_lead * 20
        
        # 3. High cash ratio = potential migration
        if "cash_withdrawal_ratio_90d" in df.columns:
            churn_risk += df["cash_withdrawal_ratio_90d"] * 15
        
        # 4. Low network connectivity = weak relationship
        if G is not None and len(G.nodes()) > 0:
            degree_centrality = nx.degree_centrality(G)
            network_weakness = pd.Series([
                1 - degree_centrality.get(acc, 0) for acc in df["account_id"]
            ])
            churn_risk += network_weakness * 20
        
        # 5. NTB status = already churned from other bank (but opportunity for us)
        if "bri_status" in df.columns:
            # NTB is opportunity, not churn risk
            churn_risk -= (df["bri_status"] == "NTB").astype(int) * 20
        
        # 6. Low digital adoption = potential for digital migration
        if "channel_mix_online_ratio" in df.columns:
            digital_weakness = 1 - df["channel_mix_online_ratio"]
            churn_risk += digital_weakness * 10
        
        result_df["churn_risk_score"] = np.clip(churn_risk, 0, 100)
        result_df["churn_risk_level"] = pd.cut(
            result_df["churn_risk_score"],
            bins=[0, 30, 50, 70, 100],
            labels=["Low", "Medium", "High", "Critical"]
        )
        
        return result_df
    
    def calculate_portfolio_health(
        self,
        df: pd.DataFrame,
        G=None
    ) -> Dict:
        """
        Calculate overall portfolio health metrics for RM.
        
        Args:
            df: Input dataframe
            G: NetworkX graph (optional)
        
        Returns:
            Dictionary with portfolio health metrics
        """
        import networkx as nx
        
        health_metrics = {}
        
        # 1. Portfolio composition
        if "bri_status" in df.columns:
            health_metrics["existing_rate"] = (df["bri_status"] == "Existing").sum() / len(df)
            health_metrics["ntb_rate"] = (df["bri_status"] == "NTB").sum() / len(df)
        
        # 2. Average lead quality
        if "lead_score_bri" in df.columns:
            health_metrics["avg_lead_score"] = df["lead_score_bri"].mean()
            health_metrics["high_quality_leads"] = (df["lead_score_bri"] >= 75).sum()
        
        # 3. Portfolio value
        if "turnover_90d" in df.columns:
            health_metrics["total_portfolio_value"] = df["turnover_90d"].sum()
            health_metrics["avg_account_value"] = df["turnover_90d"].mean()
            health_metrics["top_20_percent_value"] = df["turnover_90d"].nlargest(int(len(df) * 0.2)).sum()
        
        # 4. Network health
        if G is not None and len(G.nodes()) > 0:
            degree_centrality = nx.degree_centrality(G)
            avg_degree = np.mean(list(degree_centrality.values()))
            health_metrics["avg_network_connectivity"] = avg_degree
            health_metrics["hub_accounts"] = sum(1 for v in degree_centrality.values() if v > 0.1)
        
        # 5. Risk profile
        df_risk = self.risk_scoring(df)
        health_metrics["avg_risk_score"] = df_risk["risk_score"].mean()
        health_metrics["high_risk_accounts"] = (df_risk["risk_level"].isin(["High", "Critical"])).sum()
        
        # 6. Growth indicators
        if "turnover_90d" in df.columns:
            health_metrics["growth_potential"] = (df["bri_status"] == "NTB").sum() / len(df)
        
        # Overall health score (0-100)
        health_score = (
            health_metrics.get("existing_rate", 0) * 20 +
            (health_metrics.get("avg_lead_score", 0) / 100) * 30 +
            min(health_metrics.get("avg_network_connectivity", 0) * 100, 20) +
            (1 - health_metrics.get("avg_risk_score", 50) / 100) * 30
        )
        health_metrics["overall_health_score"] = min(health_score, 100)
        health_metrics["health_grade"] = (
            "A" if health_score >= 80 else
            "B" if health_score >= 60 else
            "C" if health_score >= 40 else "D"
        )
        
        return health_metrics
    
    def create_rm_action_prioritization(
        self,
        df: pd.DataFrame,
        G=None,
        opportunity_scores: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive RM action prioritization for wholesale banking.
        Combines all analytics to recommend RM actions.
        
        Args:
            df: Input dataframe
            G: NetworkX graph
            opportunity_scores: Pre-calculated opportunity scores
        
        Returns:
            Dataframe with RM action recommendations
        """
        result_df = df.copy()
        
        # Calculate all relevant scores
        result_df = self.calculate_lead_quality_score(result_df, G, opportunity_scores)
        result_df = self.identify_cross_sell_opportunities(result_df, G)
        result_df = self.calculate_relationship_strength(result_df, G)
        result_df = self.predict_churn_risk(result_df, G)
        
        # RM Action Priority Score (weighted combination)
        action_priority = (
            result_df["lead_quality_score"] * 0.30 +
            result_df["cross_sell_score"] * 0.25 +
            result_df["relationship_strength"] * 0.20 +
            (100 - result_df["churn_risk_score"]) * 0.15 +  # Invert churn risk
            (result_df["opportunity_score"] if "opportunity_score" in result_df.columns else result_df["lead_quality_score"]) * 0.10
        )
        
        result_df["rm_action_priority"] = np.clip(action_priority, 0, 100)
        result_df["rm_action_priority_level"] = pd.cut(
            result_df["rm_action_priority"],
            bins=[0, 50, 70, 85, 100],
            labels=["Monitor", "Follow-up", "Priority", "Immediate Action"]
        )
        
        # Recommended actions
        def recommend_action(row):
            actions = []
            
            if row.get("bri_status") == "NTB" and row.get("lead_quality_score", 0) >= 70:
                actions.append("Acquire - High Quality NTB")
            
            if row.get("churn_risk_score", 0) >= 70:
                actions.append("Retention - High Churn Risk")
            
            if row.get("cross_sell_score", 0) >= 60:
                actions.append("Cross-sell - Product Opportunity")
            
            if row.get("relationship_strength", 0) < 40:
                actions.append("Strengthen Relationship")
            
            if row.get("lead_quality_score", 0) >= 85:
                actions.append("Strategic Account - VIP Treatment")
            
            return "; ".join(actions) if actions else "Standard Monitoring"
        
        result_df["recommended_rm_action"] = result_df.apply(recommend_action, axis=1)
        
        return result_df


def analyze_data(
    df: pd.DataFrame,
    analysis_type: str = "comprehensive"
) -> Dict:
    """
    Convenience function for analytics.
    
    Args:
        df: Input dataframe
        analysis_type: Type of analysis ('comprehensive', 'anomalies', 'risk', 'growth')
    
    Returns:
        Analysis results
    """
    analytics = AdvancedAnalytics()
    
    if analysis_type == "comprehensive":
        return analytics.create_analytics_dashboard_data(df)
    elif analysis_type == "anomalies":
        return {"anomalies": analytics.detect_anomalies(df)}
    elif analysis_type == "risk":
        return {"risk": analytics.risk_scoring(df)}
    elif analysis_type == "growth":
        return {"growth": analytics.calculate_growth_metrics(df)}
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

