"""
Data Simulation Module
Generates synthetic data for both external and internal data sources.
Supports realistic simulation of banking and ecosystem data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import random

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

ECOSYSTEM_ROLES = [
    "Anchor_Corporate",
    "Feed_Mill",
    "Breeder_Farm",
    "Contract_Farmer",
    "Collector/Offtaker",
    "Slaughterhouse",
    "Retailer/Warung",
    "Logistics/Transport",
    "Input_Supplier",
]

ANCHOR_GROUPS = ["Japfa_Group", "CP_Group", "Charoen_Group", "Malindo_Group", "Independent"]


class DataSimulator:
    """Simulate external and internal data sources."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
    
    def _normalize_probs(self, probs):
        """Normalize probability array to sum to 1.0."""
        probs = np.array(probs)
        return probs / probs.sum()
    
    def simulate_external_data(
        self,
        n_accounts: int = 100,
        data_source: str = "external_bank",
        include_temporal: bool = True,
        time_periods: int = 3
    ) -> pd.DataFrame:
        """
        Simulate external data (from other banks, public sources, etc.).
        
        Args:
            n_accounts: Number of accounts to simulate
            data_source: Source identifier ('external_bank', 'public_registry', 'credit_bureau')
            include_temporal: Whether to include time-series data
            time_periods: Number of time periods to simulate
        
        Returns:
            Simulated external data dataframe
        """
        accounts = []
        
        for i in range(n_accounts):
            # Role distribution for external data (more diverse)
            role = np.random.choice(
                ECOSYSTEM_ROLES,
                p=self._normalize_probs([0.05, 0.10, 0.08, 0.30, 0.12, 0.08, 0.15, 0.07, 0.05])
            )
            
            # External accounts are more likely to be independent
            anchor_group = np.random.choice(
                ANCHOR_GROUPS,
                p=self._normalize_probs([0.15, 0.15, 0.10, 0.10, 0.50])  # Higher probability for Independent
            )
            
            segment = np.random.choice(
                ["Corporate", "SME", "Micro"],
                p=self._normalize_probs([0.20, 0.40, 0.40])
            )
            
            # External data characteristics (different patterns)
            if role == "Anchor_Corporate":
                avg_txn = np.random.uniform(500000000, 1000000000)
                txn_count = np.random.randint(200, 400)
                turnover = np.random.uniform(5000000000, 10000000000)
                cash_ratio = np.random.uniform(0.10, 0.20)
                online_ratio = np.random.uniform(0.40, 0.60)
                lead_score = np.random.randint(80, 95)
            elif role == "Feed_Mill":
                avg_txn = np.random.uniform(300000000, 500000000)
                txn_count = np.random.randint(150, 300)
                turnover = np.random.uniform(2500000000, 4500000000)
                cash_ratio = np.random.uniform(0.15, 0.25)
                online_ratio = np.random.uniform(0.35, 0.55)
                lead_score = np.random.randint(75, 90)
            elif role == "Contract_Farmer":
                avg_txn = np.random.uniform(10000000, 60000000)
                txn_count = np.random.randint(40, 120)
                turnover = np.random.uniform(80000000, 500000000)
                cash_ratio = np.random.uniform(0.30, 0.45)
                online_ratio = np.random.uniform(0.55, 0.80)
                lead_score = np.random.randint(65, 85)
            else:
                avg_txn = np.random.uniform(20000000, 150000000)
                txn_count = np.random.randint(50, 150)
                turnover = np.random.uniform(150000000, 800000000)
                cash_ratio = np.random.uniform(0.20, 0.35)
                online_ratio = np.random.uniform(0.45, 0.70)
                lead_score = np.random.randint(60, 85)
            
            account_id = f"EXT_{data_source.upper()[:3]}_{str(i+1).zfill(4)}"
            
            account_data = {
                "account_id": account_id,
                "legal_name": f"External Entity {i+1}",
                "ecosystem_role": role,
                "anchor_group": anchor_group,
                "anchor_level": np.random.choice([0, 1, 2], p=self._normalize_probs([0.1, 0.3, 0.6])),
                "segment_code": segment,
                "primary_bank": np.random.choice(["Other", "NTB"], p=self._normalize_probs([0.7, 0.3])),
                "bri_status": "NTB",  # External data is always NTB
                "ntb_status": "NTB",
                "data_source": data_source,
                "avg_txn_amount_30d": avg_txn,
                "txn_count_30d": txn_count,
                "turnover_90d": turnover,
                "cash_withdrawal_ratio_90d": cash_ratio,
                "merchant_diversity_90d": np.random.randint(8, 35),
                "unique_devices_90d": np.random.randint(1, 8),
                "unique_ips_90d": np.random.randint(2, 12),
                "country_risk_score": np.random.uniform(0.10, 0.35),
                "channel_mix_online_ratio": online_ratio,
                "lead_score_bri": lead_score,
            }
            
            # Add temporal data if requested
            if include_temporal:
                for period in range(time_periods):
                    period_date = datetime.now() - timedelta(days=30 * (period + 1))
                    account_data[f"turnover_90d_period_{period}"] = turnover * np.random.uniform(0.85, 1.15)
                    account_data[f"txn_count_30d_period_{period}"] = txn_count * np.random.uniform(0.90, 1.10)
            
            accounts.append(account_data)
        
        return pd.DataFrame(accounts)
    
    def simulate_internal_data(
        self,
        n_accounts: int = 100,
        data_source: str = "internal_bri",
        include_historical: bool = True,
        historical_months: int = 6
    ) -> pd.DataFrame:
        """
        Simulate internal data (BRI's own customer data).
        
        Args:
            n_accounts: Number of accounts to simulate
            data_source: Source identifier ('internal_bri', 'crm', 'transaction_db')
            include_historical: Whether to include historical trends
            historical_months: Number of historical months to simulate
        
        Returns:
            Simulated internal data dataframe
        """
        accounts = []
        
        for i in range(n_accounts):
            role = np.random.choice(
                ECOSYSTEM_ROLES,
                p=self._normalize_probs([0.08, 0.12, 0.10, 0.35, 0.12, 0.08, 0.10, 0.08, 0.07])
            )
            
            # Internal accounts are more likely to be in anchor groups
            anchor_group = np.random.choice(
                ANCHOR_GROUPS,
                p=self._normalize_probs([0.30, 0.25, 0.15, 0.10, 0.20])
            )
            
            segment = np.random.choice(
                ["Corporate", "SME", "Micro"],
                p=self._normalize_probs([0.25, 0.45, 0.30])
            )
            
            # BRI status (internal data has existing customers)
            bri_status = np.random.choice(
                ["Existing", "NTB"],
                p=self._normalize_probs([0.70, 0.30])
            )
            
            # Internal data characteristics (more detailed)
            if role == "Anchor_Corporate":
                avg_txn = np.random.uniform(800000000, 1200000000)
                txn_count = np.random.randint(350, 500)
                turnover = np.random.uniform(7000000000, 12000000000)
                cash_ratio = np.random.uniform(0.08, 0.15)
                online_ratio = np.random.uniform(0.30, 0.45)
                lead_score = np.random.randint(90, 100)
            elif role == "Feed_Mill":
                avg_txn = np.random.uniform(400000000, 600000000)
                txn_count = np.random.randint(250, 350)
                turnover = np.random.uniform(3500000000, 5000000000)
                cash_ratio = np.random.uniform(0.15, 0.22)
                online_ratio = np.random.uniform(0.35, 0.50)
                lead_score = np.random.randint(85, 98)
            elif role == "Contract_Farmer":
                avg_txn = np.random.uniform(15000000, 80000000)
                txn_count = np.random.randint(60, 150)
                turnover = np.random.uniform(100000000, 600000000)
                cash_ratio = np.random.uniform(0.25, 0.40)
                online_ratio = np.random.uniform(0.50, 0.75)
                lead_score = np.random.randint(70, 88)
            else:
                avg_txn = np.random.uniform(30000000, 150000000)
                txn_count = np.random.randint(60, 140)
                turnover = np.random.uniform(200000000, 700000000)
                cash_ratio = np.random.uniform(0.20, 0.35)
                online_ratio = np.random.uniform(0.45, 0.65)
                lead_score = np.random.randint(65, 85)
            
            account_id = f"INT_{data_source.upper()[:3]}_{str(i+1).zfill(4)}"
            
            account_data = {
                "account_id": account_id,
                "legal_name": f"Internal Customer {i+1}",
                "ecosystem_role": role,
                "anchor_group": anchor_group,
                "anchor_level": np.random.choice([0, 1, 2], p=self._normalize_probs([0.15, 0.40, 0.45])),
                "segment_code": segment,
                "primary_bank": "BRI" if bri_status == "Existing" else "Other",
                "bri_status": bri_status,
                "ntb_status": "Existing" if bri_status == "Existing" else "NTB",
                "data_source": data_source,
                "avg_txn_amount_30d": avg_txn,
                "txn_count_30d": txn_count,
                "turnover_90d": turnover,
                "cash_withdrawal_ratio_90d": cash_ratio,
                "merchant_diversity_90d": np.random.randint(10, 45),
                "unique_devices_90d": np.random.randint(2, 10),
                "unique_ips_90d": np.random.randint(3, 20),
                "country_risk_score": np.random.uniform(0.08, 0.25),
                "channel_mix_online_ratio": online_ratio,
                "lead_score_bri": lead_score,
            }
            
            # Add historical trends if requested
            if include_historical:
                base_turnover = turnover
                base_txn_count = txn_count
                
                for month in range(historical_months):
                    month_date = datetime.now() - timedelta(days=30 * (month + 1))
                    # Simulate growth/decline trends
                    trend_factor = 1.0 + (month * 0.02)  # Slight growth trend
                    noise = np.random.uniform(0.90, 1.10)
                    
                    account_data[f"turnover_month_{month}"] = base_turnover * trend_factor * noise
                    account_data[f"txn_count_month_{month}"] = int(base_txn_count * trend_factor * noise)
                    account_data[f"month_{month}_date"] = month_date.strftime("%Y-%m-%d")
            
            accounts.append(account_data)
        
        return pd.DataFrame(accounts)
    
    def simulate_combined_dataset(
        self,
        n_internal: int = 80,
        n_external: int = 50,
        include_temporal: bool = True
    ) -> pd.DataFrame:
        """
        Simulate combined dataset with both internal and external data.
        
        Args:
            n_internal: Number of internal accounts
            n_external: Number of external accounts
            include_temporal: Whether to include temporal features
        
        Returns:
            Combined dataframe
        """
        internal_df = self.simulate_internal_data(
            n_accounts=n_internal,
            include_historical=include_temporal
        )
        
        external_df = self.simulate_external_data(
            n_accounts=n_external,
            include_temporal=include_temporal
        )
        
        combined_df = pd.concat([internal_df, external_df], ignore_index=True)
        return combined_df
    
    def add_noise_to_data(
        self,
        df: pd.DataFrame,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Add realistic noise to data for simulation purposes.
        
        Args:
            df: Input dataframe
            noise_level: Level of noise (0.0 to 1.0)
        
        Returns:
            Dataframe with added noise
        """
        df_noisy = df.copy()
        
        for col in NUMERIC_FEATURES:
            if col in df_noisy.columns:
                noise = np.random.normal(0, noise_level, len(df_noisy))
                df_noisy[col] = df_noisy[col] * (1 + noise)
                # Ensure non-negative values
                df_noisy[col] = np.maximum(df_noisy[col], 0)
        
        return df_noisy
    
    def simulate_missing_data(
        self,
        df: pd.DataFrame,
        missing_rate: float = 0.05
    ) -> pd.DataFrame:
        """
        Simulate missing data patterns.
        
        Args:
            df: Input dataframe
            missing_rate: Proportion of values to make missing
        
        Returns:
            Dataframe with missing values
        """
        df_missing = df.copy()
        
        for col in NUMERIC_FEATURES:
            if col in df_missing.columns:
                n_missing = int(len(df_missing) * missing_rate)
                missing_indices = np.random.choice(
                    df_missing.index,
                    size=n_missing,
                    replace=False
                )
                df_missing.loc[missing_indices, col] = np.nan
        
        return df_missing


def simulate_data(
    data_type: str = "combined",
    n_accounts: int = 100,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to simulate data.
    
    Args:
        data_type: Type of data ('internal', 'external', 'combined')
        n_accounts: Number of accounts
        **kwargs: Additional arguments
    
    Returns:
        Simulated dataframe
    """
    simulator = DataSimulator()
    
    if data_type == "internal":
        return simulator.simulate_internal_data(n_accounts=n_accounts, **kwargs)
    elif data_type == "external":
        return simulator.simulate_external_data(n_accounts=n_accounts, **kwargs)
    elif data_type == "combined":
        n_internal = kwargs.get("n_internal", n_accounts // 2)
        n_external = kwargs.get("n_external", n_accounts - n_internal)
        return simulator.simulate_combined_dataset(
            n_internal=n_internal,
            n_external=n_external,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

