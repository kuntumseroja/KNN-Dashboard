import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import io
import re
import hashlib
from typing import Dict, Optional

# Import new modules
from model_training import ModelTrainer, train_from_data
from model_inference import ModelInference, infer_from_model
from data_simulation import DataSimulator, simulate_data
from advanced_analytics import AdvancedAnalytics, analyze_data

# Try to import spacy for NER, fallback to regex-based if not available
SPACY_AVAILABLE = False
spacy = None
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    pass


st.set_page_config(
    page_title="OBS Account Relationship Dashboard",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    .metric-card-red {
        background: linear-gradient(135deg, #CB356B 0%, #BD3F32 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: white;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0EA5E9;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #F8FAFC;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


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

ROLE_COLORS = {
    "Anchor_Corporate": "#1E3A5F",
    "Feed_Mill": "#2563EB",
    "Breeder_Farm": "#059669",
    "Contract_Farmer": "#10B981",
    "Collector/Offtaker": "#F59E0B",
    "Slaughterhouse": "#EF4444",
    "Retailer/Warung": "#8B5CF6",
    "Logistics/Transport": "#6366F1",
    "Input_Supplier": "#EC4899",
}

ANCHOR_GROUPS = ["Japfa_Group", "CP_Group", "Charoen_Group", "Malindo_Group", "Independent"]

# NER and Anonymization Configuration
@st.cache_resource
def load_ner_model():
    """
    Load NER model for entity recognition. Falls back to regex-based if spacy not available.
    
    To install spaCy models (optional, for better accuracy):
    - Indonesian: python -m spacy download id_core_web_sm
    - Multilingual: python -m spacy download xx_ent_wiki_sm
    - English: python -m spacy download en_core_web_sm
    """
    if SPACY_AVAILABLE and spacy is not None:
        try:
            # Try to load Indonesian model first, then multilingual, then English
            try:
                nlp = spacy.load("id_core_web_sm")
                return nlp, True
            except OSError:
                try:
                    nlp = spacy.load("xx_ent_wiki_sm")  # Multilingual
                    return nlp, True
                except OSError:
                    try:
                        nlp = spacy.load("en_core_web_sm")  # English as fallback
                        return nlp, True
                    except OSError:
                        return None, False
        except Exception:
            return None, False
    return None, False


def detect_entities_regex(text: str) -> Dict[str, list]:
    """Regex-based entity detection for Indonesian names and companies."""
    entities = {
        "PERSON": [],
        "ORG": [],
        "LOC": []
    }
    
    if not text or pd.isna(text):
        return entities
    
    text_str = str(text)
    
    # Indonesian company patterns: PT, CV, UD, Koperasi, etc.
    org_patterns = [
        r'\b(PT|CV|UD|PD|Perum|Persero|Koperasi)\s+([A-Z][A-Za-z\s&]+?)(?:\s+(?:Tbk|TBK|Tbk\.|TBK\.))?',
        r'\b([A-Z][A-Za-z\s&]+?)\s+(?:PT|CV|UD|PD|Perum|Persero|Koperasi)',
    ]
    
    for pattern in org_patterns:
        matches = re.finditer(pattern, text_str, re.IGNORECASE)
        for match in matches:
            entities["ORG"].append(match.group(0))
    
    # Indonesian location patterns (common provinces, cities)
    loc_patterns = [
        r'\b(Jawa\s+(?:Barat|Tengah|Timur)|Sumatera\s+(?:Utara|Selatan)|Sulawesi\s+(?:Selatan|Utara)|Kalimantan\s+(?:Selatan|Barat|Timur)|Bali|NTB|NTT|Lampung|Yogyakarta|Jakarta|Bandung|Surabaya|Medan|Makassar)',
    ]
    
    for pattern in loc_patterns:
        matches = re.finditer(pattern, text_str, re.IGNORECASE)
        for match in matches:
            entities["LOC"].append(match.group(0))
    
    # Person names (Indonesian common patterns: 2-4 words, capitalized)
    # This is a simplified pattern - real NER would be better
    person_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    # Exclude common company words
    exclude_words = {'PT', 'CV', 'UD', 'Toko', 'Apotek', 'Farm', 'Breeding', 'Sentra', 
                    'RPH', 'Armada', 'Ayam', 'Telur', 'Hewan', 'Livebird', 'Regional'}
    
    matches = re.finditer(person_pattern, text_str)
    for match in matches:
        potential_name = match.group(0)
        words = potential_name.split()
        # If it doesn't contain excluded words and has 2-4 words, likely a person name
        if not any(word in exclude_words for word in words) and 2 <= len(words) <= 4:
            entities["PERSON"].append(potential_name)
    
    return entities


def anonymize_text(text: str, anonymization_map: Dict[str, str], use_ner: bool = True) -> str:
    """
    Anonymize text by replacing detected entities with anonymized versions.
    Complies with UU PDP (Indonesian Personal Data Protection Law) and banking confidentiality.
    
    Args:
        text: Text to anonymize
        anonymization_map: Dictionary to store entity mappings for consistency
        use_ner: Whether to use NER model (falls back to regex if unavailable)
    
    Returns:
        Anonymized text string
    """
    if not text or pd.isna(text):
        return text
    
    text_str = str(text)
    anonymized = text_str
    
    entities_to_anonymize = {}
    
    if use_ner:
        # Load NER model if available
        nlp, ner_available = load_ner_model()
        
        if ner_available and nlp is not None:
            try:
                # Use spaCy NER
                doc = nlp(text_str)
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG"]:  # Anonymize persons and organizations
                        entities_to_anonymize[ent.text] = ent.label_
            except Exception:
                # Fall back to regex if NER fails
                detected = detect_entities_regex(text_str)
                for entity_type in ["PERSON", "ORG"]:
                    for entity in detected.get(entity_type, []):
                        entities_to_anonymize[entity] = entity_type
        else:
            # Use regex-based detection
            detected = detect_entities_regex(text_str)
            for entity_type in ["PERSON", "ORG"]:
                for entity in detected.get(entity_type, []):
                    entities_to_anonymize[entity] = entity_type
    else:
        # Use regex-based detection
        detected = detect_entities_regex(text_str)
        for entity_type in ["PERSON", "ORG"]:
            for entity in detected.get(entity_type, []):
                entities_to_anonymize[entity] = entity_type
    
    # Anonymize detected entities
    for entity, entity_type in entities_to_anonymize.items():
        if entity not in anonymization_map:
            # Create consistent anonymized identifier
            # Use hash for consistency but make it readable
            hash_id = hashlib.md5(entity.encode()).hexdigest()[:8].upper()
            if entity_type == "PERSON":
                anonymization_map[entity] = f"PERSON_{hash_id}"
            elif entity_type == "ORG":
                # Keep company type prefix if present
                if entity.startswith(("PT ", "CV ", "UD ", "PD ")):
                    prefix = entity.split()[0]
                    anonymization_map[entity] = f"{prefix} ENTITY_{hash_id}"
                else:
                    anonymization_map[entity] = f"ENTITY_{hash_id}"
            else:
                anonymization_map[entity] = f"ENTITY_{hash_id}"
        
        # Replace in text (case-insensitive, whole word)
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        anonymized = pattern.sub(anonymization_map[entity], anonymized)
    
    return anonymized


def anonymize_dataframe(df: pd.DataFrame, anonymize_fields: list = None) -> pd.DataFrame:
    """
    Anonymize sensitive fields in dataframe for UU PDP compliance.
    Maintains consistent anonymization across the dataset.
    """
    if anonymize_fields is None:
        anonymize_fields = ["legal_name"]
    
    df_anon = df.copy()
    anonymization_map = {}
    
    # Initialize session state for consistent anonymization
    if "anonymization_map" not in st.session_state:
        st.session_state.anonymization_map = {}
    
    anonymization_map = st.session_state.anonymization_map
    
    # Anonymize specified fields
    for field in anonymize_fields:
        if field in df_anon.columns:
            df_anon[field] = df_anon[field].apply(
                lambda x: anonymize_text(x, anonymization_map, use_ner=True)
            )
    
    # Update session state
    st.session_state.anonymization_map = anonymization_map
    
    return df_anon


@st.cache_data
def generate_poultry_ecosystem_data(n_accounts=80, seed=42):
    np.random.seed(seed)
    
    accounts = []
    
    anchor_configs = [
        {"prefix": "JAPFA", "group": "Japfa_Group", "name": "Japfa Comfeed Indonesia Tbk", "role": "Anchor_Corporate"},
        {"prefix": "CP", "group": "CP_Group", "name": "Charoen Pokphand Indonesia Tbk", "role": "Anchor_Corporate"},
    ]
    
    for i, anchor in enumerate(anchor_configs):
        accounts.append({
            "account_id": f"ACC_{anchor['prefix']}_01",
            "legal_name": anchor["name"],
            "ecosystem_role": anchor["role"],
            "anchor_group": anchor["group"],
            "anchor_level": 0,
            "segment_code": "Corporate",
            "primary_bank": "BRI",
            "bri_status": "Existing",
            "ntb_status": "NTB",
            "avg_txn_amount_30d": np.random.uniform(800000000, 1200000000),
            "txn_count_30d": np.random.randint(350, 500),
            "turnover_90d": np.random.uniform(7000000000, 12000000000),
            "cash_withdrawal_ratio_90d": np.random.uniform(0.08, 0.15),
            "merchant_diversity_90d": np.random.randint(40, 50),
            "unique_devices_90d": np.random.randint(6, 10),
            "unique_ips_90d": np.random.randint(12, 20),
            "country_risk_score": np.random.uniform(0.08, 0.15),
            "channel_mix_online_ratio": np.random.uniform(0.30, 0.45),
            "lead_score_bri": np.random.randint(95, 100),
        })
        
        accounts.append({
            "account_id": f"ACC_{anchor['prefix']}_FM01",
            "legal_name": f"{anchor['prefix'].title()} Regional Feed Mill",
            "ecosystem_role": "Feed_Mill",
            "anchor_group": anchor["group"],
            "anchor_level": 0,
            "segment_code": "Corporate",
            "primary_bank": "BRI",
            "bri_status": "Existing",
            "ntb_status": "NTB",
            "avg_txn_amount_30d": np.random.uniform(400000000, 600000000),
            "txn_count_30d": np.random.randint(250, 350),
            "turnover_90d": np.random.uniform(3500000000, 5000000000),
            "cash_withdrawal_ratio_90d": np.random.uniform(0.15, 0.22),
            "merchant_diversity_90d": np.random.randint(32, 42),
            "unique_devices_90d": np.random.randint(5, 8),
            "unique_ips_90d": np.random.randint(9, 14),
            "country_risk_score": np.random.uniform(0.08, 0.15),
            "channel_mix_online_ratio": np.random.uniform(0.35, 0.50),
            "lead_score_bri": np.random.randint(90, 98),
        })
    
    farm_names_prefix = ["CV Sumber Ayam", "PT Mitra Unggas", "UD Ayam Makmur", "Koperasi Peternak", 
                         "CV Ternak Sejahtera", "PT Broiler Prima", "UD Peternakan Jaya", "CV Unggas Mandiri",
                         "PT Farm Nusantara", "UD Ayam Berkah", "CV Peternak Maju", "PT Agro Poultry"]
    farm_names_suffix = ["Sejahtera", "Mandiri", "Jaya", "Makmur", "Prima", "Utama", "Sentosa", "Abadi"]
    
    regions = ["Jawa Barat", "Jawa Tengah", "Jawa Timur", "Sumatera Utara", "Sulawesi Selatan", 
               "Lampung", "Bali", "NTB", "Kalimantan Selatan", "Yogyakarta"]
    
    remaining = n_accounts - len(accounts)
    
    role_distribution = {
        "Breeder_Farm": 0.08,
        "Contract_Farmer": 0.35,
        "Collector/Offtaker": 0.12,
        "Slaughterhouse": 0.08,
        "Retailer/Warung": 0.20,
        "Logistics/Transport": 0.08,
        "Input_Supplier": 0.09,
    }
    
    account_counter = {}
    
    for role, pct in role_distribution.items():
        count = int(remaining * pct)
        
        for j in range(count):
            role_prefix = role.split("/")[0].split("_")[0].upper()[:4]
            account_counter[role] = account_counter.get(role, 0) + 1
            
            if role in ["Retailer/Warung", "Contract_Farmer"]:
                anchor_group = np.random.choice(["Japfa_Group", "CP_Group", "Independent"], p=[0.4, 0.3, 0.3])
                segment = np.random.choice(["SME", "Micro"], p=[0.3, 0.7])
                anchor_level = np.random.choice([1, 2], p=[0.4, 0.6])
            elif role in ["Input_Supplier"]:
                anchor_group = "Independent"
                segment = np.random.choice(["SME", "Micro"], p=[0.6, 0.4])
                anchor_level = 2
            else:
                anchor_group = np.random.choice(["Japfa_Group", "CP_Group"], p=[0.6, 0.4])
                segment = "SME"
                anchor_level = 1
            
            if role == "Contract_Farmer":
                legal_name = f"{np.random.choice(farm_names_prefix)} {np.random.choice(farm_names_suffix)} {np.random.choice(regions)}"
            elif role == "Retailer/Warung":
                legal_name = f"Toko Ayam & Telur {np.random.choice(['Berkah', 'Sejahtera', 'Makmur', 'Jaya'])} {j+1}"
            elif role == "Collector/Offtaker":
                legal_name = f"PT Sentra Livebird {np.random.choice(regions)}"
            elif role == "Slaughterhouse":
                legal_name = f"PT RPH Ayam {np.random.choice(['Sehat', 'Higienis', 'Prima', 'Bersih'])} {np.random.choice(regions)}"
            elif role == "Logistics/Transport":
                legal_name = f"CV Armada {np.random.choice(['Dingin', 'Cepat', 'Ekspres'])} {np.random.choice(regions)}"
            elif role == "Input_Supplier":
                legal_name = f"Apotek Hewan {np.random.choice(['Sehat', 'Prima', 'Jaya'])} {np.random.choice(regions)}"
            elif role == "Breeder_Farm":
                legal_name = f"PT Breeding Farm {np.random.choice(farm_names_suffix)} {np.random.choice(regions)}"
            else:
                legal_name = f"PT {role.replace('_', ' ')} {j+1}"
            
            primary_bank = np.random.choice(["BRI", "Other", "NTB"], p=[0.5, 0.3, 0.2])
            
            if primary_bank == "BRI":
                bri_status = "Existing"
                ntb_status = "NTB"
            elif primary_bank == "Other":
                bri_status = "NTB"
                ntb_status = "Existing"
            else:
                bri_status = "NTB"
                ntb_status = "NTB"
            
            if role == "Breeder_Farm":
                avg_txn = np.random.uniform(70000000, 120000000)
                txn_count = np.random.randint(140, 200)
                turnover = np.random.uniform(500000000, 900000000)
                cash_ratio = np.random.uniform(0.18, 0.28)
                merchant_div = np.random.randint(20, 30)
                devices = np.random.randint(3, 6)
                ips = np.random.randint(5, 10)
                online_ratio = np.random.uniform(0.45, 0.65)
                lead_score = np.random.randint(82, 92)
            elif role == "Contract_Farmer":
                avg_txn = np.random.uniform(15000000, 80000000)
                txn_count = np.random.randint(60, 150)
                turnover = np.random.uniform(100000000, 600000000)
                cash_ratio = np.random.uniform(0.25, 0.40)
                merchant_div = np.random.randint(10, 22)
                devices = np.random.randint(2, 5)
                ips = np.random.randint(3, 8)
                online_ratio = np.random.uniform(0.50, 0.75)
                lead_score = np.random.randint(70, 88)
            elif role == "Collector/Offtaker":
                avg_txn = np.random.uniform(80000000, 180000000)
                txn_count = np.random.randint(120, 180)
                turnover = np.random.uniform(700000000, 1200000000)
                cash_ratio = np.random.uniform(0.20, 0.30)
                merchant_div = np.random.randint(18, 28)
                devices = np.random.randint(3, 6)
                ips = np.random.randint(5, 9)
                online_ratio = np.random.uniform(0.38, 0.55)
                lead_score = np.random.randint(85, 95)
            elif role == "Slaughterhouse":
                avg_txn = np.random.uniform(150000000, 300000000)
                txn_count = np.random.randint(150, 220)
                turnover = np.random.uniform(1200000000, 2000000000)
                cash_ratio = np.random.uniform(0.15, 0.25)
                merchant_div = np.random.randint(24, 35)
                devices = np.random.randint(4, 7)
                ips = np.random.randint(6, 12)
                online_ratio = np.random.uniform(0.42, 0.58)
                lead_score = np.random.randint(75, 88)
            elif role == "Retailer/Warung":
                avg_txn = np.random.uniform(5000000, 20000000)
                txn_count = np.random.randint(50, 120)
                turnover = np.random.uniform(30000000, 150000000)
                cash_ratio = np.random.uniform(0.35, 0.50)
                merchant_div = np.random.randint(8, 18)
                devices = np.random.randint(1, 4)
                ips = np.random.randint(2, 6)
                online_ratio = np.random.uniform(0.65, 0.85)
                lead_score = np.random.randint(60, 80)
            elif role == "Logistics/Transport":
                avg_txn = np.random.uniform(40000000, 80000000)
                txn_count = np.random.randint(70, 120)
                turnover = np.random.uniform(300000000, 600000000)
                cash_ratio = np.random.uniform(0.15, 0.25)
                merchant_div = np.random.randint(15, 25)
                devices = np.random.randint(3, 6)
                ips = np.random.randint(4, 8)
                online_ratio = np.random.uniform(0.50, 0.68)
                lead_score = np.random.randint(78, 90)
            else:
                avg_txn = np.random.uniform(30000000, 60000000)
                txn_count = np.random.randint(60, 100)
                turnover = np.random.uniform(200000000, 450000000)
                cash_ratio = np.random.uniform(0.25, 0.38)
                merchant_div = np.random.randint(15, 25)
                devices = np.random.randint(2, 5)
                ips = np.random.randint(3, 7)
                online_ratio = np.random.uniform(0.45, 0.62)
                lead_score = np.random.randint(65, 82)
            
            accounts.append({
                "account_id": f"ACC_{role_prefix}_{str(account_counter[role]).zfill(3)}",
                "legal_name": legal_name,
                "ecosystem_role": role,
                "anchor_group": anchor_group,
                "anchor_level": anchor_level,
                "segment_code": segment,
                "primary_bank": primary_bank,
                "bri_status": bri_status,
                "ntb_status": ntb_status,
                "avg_txn_amount_30d": avg_txn,
                "txn_count_30d": txn_count,
                "turnover_90d": turnover,
                "cash_withdrawal_ratio_90d": cash_ratio,
                "merchant_diversity_90d": merchant_div,
                "unique_devices_90d": devices,
                "unique_ips_90d": ips,
                "country_risk_score": np.random.uniform(0.10, 0.30),
                "channel_mix_online_ratio": online_ratio,
                "lead_score_bri": lead_score,
            })
    
    return pd.DataFrame(accounts)


def validate_uploaded_data(df):
    required_cols = ["account_id"] + NUMERIC_FEATURES + ["segment_code"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        null_count = df[col].isna().sum()
        if null_count > 0:
            return False, f"Column '{col}' has {null_count} invalid/missing values"
    
    if df["account_id"].isna().any():
        return False, "Account ID column contains empty values"
    
    if df["account_id"].duplicated().any():
        return False, "Duplicate account IDs found"
    
    if len(df) < 3:
        return False, "Dataset must contain at least 3 accounts"
    
    if "ecosystem_role" not in df.columns:
        df["ecosystem_role"] = "Unknown"
    if "legal_name" not in df.columns:
        df["legal_name"] = df["account_id"]
    if "anchor_group" not in df.columns:
        df["anchor_group"] = "Independent"
    if "anchor_level" not in df.columns:
        df["anchor_level"] = 2
    if "primary_bank" not in df.columns:
        df["primary_bank"] = "Unknown"
    if "bri_status" not in df.columns:
        df["bri_status"] = "Unknown"
    if "ntb_status" not in df.columns:
        df["ntb_status"] = "Unknown"
    
    return True, df


def build_knn_pipeline(df, n_neighbors=6):
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
        metric="euclidean"
    )
    
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("knn", knn)
    ])
    
    X = df[NUMERIC_FEATURES + cat_features]
    pipeline.fit(X)
    
    return pipeline, cat_features


def find_neighbors_and_edges(df, pipeline, cat_features, max_distance=2.0):
    X = df[NUMERIC_FEATURES + cat_features]
    X_processed = pipeline.named_steps["preprocess"].transform(X)
    
    distances, indices = pipeline.named_steps["knn"].kneighbors(X_processed)
    
    account_ids = df["account_id"].values
    edges = []
    seen_pairs = set()
    
    # Create anchor group mapping for quick lookup
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
                
                # Determine if this is a cross-anchor relationship
                is_cross_anchor = (src_anchor != dst_anchor) and (src_anchor != "Independent") and (dst_anchor != "Independent")
                is_anchor_bridge = (src_level <= 1) or (dst_level <= 1)
                anchor_distance = abs(src_level - dst_level)
                
                edges.append({
                    "src": src_acc,
                    "dst": dst_acc,
                    "distance": float(dist),
                    "similarity": similarity,
                    "src_anchor": src_anchor,
                    "dst_anchor": dst_anchor,
                    "is_cross_anchor": is_cross_anchor,
                    "is_anchor_bridge": is_anchor_bridge,
                    "anchor_distance": anchor_distance,
                    "anchor_level_diff": anchor_distance
                })
    
    return pd.DataFrame(edges)


def build_graph(df, edges_df):
    G = nx.Graph()
    G.add_nodes_from(df["account_id"].values)
    
    if len(edges_df) > 0:
        for _, row in edges_df.iterrows():
            edge_attrs = {
                "distance": row["distance"], 
                "similarity": row["similarity"]
            }
            # Add cross-anchor attributes if available
            if "is_cross_anchor" in row:
                edge_attrs["is_cross_anchor"] = row["is_cross_anchor"]
                edge_attrs["src_anchor"] = row.get("src_anchor", "Unknown")
                edge_attrs["dst_anchor"] = row.get("dst_anchor", "Unknown")
                edge_attrs["is_anchor_bridge"] = row.get("is_anchor_bridge", False)
                edge_attrs["anchor_distance"] = row.get("anchor_distance", 0)
            
            G.add_edge(row["src"], row["dst"], **edge_attrs)
    
    return G


def detect_communities_louvain(G):
    if len(G.edges()) == 0:
        return {node: 0 for node in G.nodes()}
    
    communities = nx.community.louvain_communities(G, seed=42)
    community_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_map[node] = idx
    return community_map


def detect_communities_dbscan(df, cat_features, eps=0.5, min_samples=3):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, cat_features),
        ]
    )
    
    X = df[NUMERIC_FEATURES + cat_features]
    X_processed = preprocessor.fit_transform(X)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_processed)
    
    community_map = {}
    for i, account_id in enumerate(df["account_id"].values):
        community_map[account_id] = int(labels[i])
    
    return community_map


def calculate_opportunity_score(G, df, community_map):
    if len(G.nodes()) == 0:
        return {}
    
    degree_centrality = nx.degree_centrality(G) if len(G.edges()) > 0 else {n: 0 for n in G.nodes()}
    betweenness_centrality = nx.betweenness_centrality(G) if len(G.edges()) > 0 else {n: 0 for n in G.nodes()}
    
    community_sizes = {}
    for node, comm in community_map.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    scores = {}
    
    for account_id in df["account_id"].values:
        account_data = df[df["account_id"] == account_id].iloc[0]
        
        network_score = degree_centrality.get(account_id, 0) * 20
        influence_score = betweenness_centrality.get(account_id, 0) * 15
        
        community = community_map.get(account_id, 0)
        community_size = community_sizes.get(community, 1)
        ecosystem_score = min(community_size / 8, 1) * 15
        
        lead_score = account_data.get("lead_score_bri", 70) / 100 * 25
        
        potential_score = 0
        if account_data.get("bri_status") == "NTB":
            potential_score += 15
        if account_data.get("anchor_level", 2) <= 1:
            potential_score += 10
        
        total_score = network_score + influence_score + ecosystem_score + lead_score + potential_score
        
        if total_score >= 70:
            priority = "High"
        elif total_score >= 45:
            priority = "Medium"
        else:
            priority = "Low"
        
        scores[account_id] = {
            "total_score": min(total_score, 100),
            "network_score": network_score,
            "influence_score": influence_score,
            "ecosystem_score": ecosystem_score,
            "lead_score": lead_score,
            "potential_score": potential_score,
            "priority": priority
        }
    
    return scores


def create_network_visualization(G, df, community_map, selected_account=None, color_by="ecosystem_role", highlight_cross_anchor=True):
    if len(G.nodes()) == 0:
        return go.Figure()
    
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Separate edges for regular and cross-anchor relationships
    regular_edge_x = []
    regular_edge_y = []
    cross_anchor_edge_x = []
    cross_anchor_edge_y = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        is_cross_anchor = edge[2].get("is_cross_anchor", False) if len(edge[2]) > 0 else False
        
        if highlight_cross_anchor and is_cross_anchor:
            cross_anchor_edge_x.extend([x0, x1, None])
            cross_anchor_edge_y.extend([y0, y1, None])
        else:
            regular_edge_x.extend([x0, x1, None])
            regular_edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=regular_edge_x, y=regular_edge_y,
        line=dict(width=1, color="rgba(150,150,150,0.4)"),
        hoverinfo="none",
        mode="lines",
        name="Regular Relationships"
    )
    
    traces = [edge_trace]
    
    # Add cross-anchor edge trace if enabled
    if highlight_cross_anchor and len(cross_anchor_edge_x) > 0:
        cross_anchor_trace = go.Scatter(
            x=cross_anchor_edge_x, y=cross_anchor_edge_y,
            line=dict(width=2.5, color="rgba(239,68,68,0.7)"),
            hoverinfo="none",
            mode="lines",
            name="Cross-Anchor Relationships"
        )
        traces.append(cross_anchor_trace)
    
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    centrality = nx.degree_centrality(G) if len(G.edges()) > 0 else {n: 0 for n in G.nodes()}
    
    role_to_num = {role: i for i, role in enumerate(ECOSYSTEM_ROLES)}
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        account_data = df[df["account_id"] == node].iloc[0] if len(df[df["account_id"] == node]) > 0 else None
        
        if color_by == "ecosystem_role" and account_data is not None:
            role = account_data.get("ecosystem_role", "Unknown")
            node_colors.append(role_to_num.get(role, 0))
        elif color_by == "anchor_group" and account_data is not None:
            groups = ["Japfa_Group", "CP_Group", "Charoen_Group", "Malindo_Group", "Independent"]
            group = account_data.get("anchor_group", "Independent")
            node_colors.append(groups.index(group) if group in groups else 4)
        else:
            node_colors.append(community_map.get(node, 0))
        
        degree = G.degree(node)
        base_size = 12 if account_data is not None and account_data.get("ecosystem_role") == "Anchor_Corporate" else 8
        node_sizes.append(base_size + degree * 2)
        
        if account_data is not None:
            text = f"<b>{account_data.get('legal_name', node)}</b><br>"
            text += f"Account: {node}<br>"
            text += f"Role: {account_data.get('ecosystem_role', 'N/A')}<br>"
            text += f"Anchor: {account_data.get('anchor_group', 'N/A')}<br>"
            text += f"Segment: {account_data['segment_code']}<br>"
            text += f"Bank Status: {account_data.get('bri_status', 'N/A')}<br>"
            text += f"Connections: {degree}<br>"
            text += f"Lead Score: {account_data.get('lead_score_bri', 0)}"
        else:
            text = node
        node_text.append(text)
    
    if color_by == "ecosystem_role":
        colorscale = "Viridis"
        colorbar_title = "Role"
    elif color_by == "anchor_group":
        colorscale = "Portland"
        colorbar_title = "Anchor"
    else:
        colorscale = "Viridis"
        colorbar_title = "Community"
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title=dict(text=colorbar_title, side="right"),
                xanchor="left"
            ),
            line=dict(width=2, color="white")
        )
    )
    
    traces.append(node_trace)
    
    if selected_account and selected_account in G.nodes():
        selected_idx = list(G.nodes()).index(selected_account)
        highlight_trace = go.Scatter(
            x=[node_x[selected_idx]],
            y=[node_y[selected_idx]],
            mode="markers",
            hoverinfo="skip",
            marker=dict(
                size=node_sizes[selected_idx] + 12,
                color="rgba(255,0,0,0.3)",
                line=dict(width=3, color="#EF4444")
            ),
            showlegend=False
        )
        traces.append(highlight_trace)
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=dict(
            text="Poultry Ecosystem Network",
            font=dict(size=18, color="#1E3A5F")
        ),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=550,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig


def get_account_neighbors(G, account_id, df):
    if account_id not in G.nodes():
        return pd.DataFrame()
    
    neighbors = list(G.neighbors(account_id))
    if not neighbors:
        return pd.DataFrame()
    
    account_anchor = df[df["account_id"] == account_id]["anchor_group"].iloc[0] if len(df[df["account_id"] == account_id]) > 0 else "Unknown"
    
    neighbor_data = []
    for neighbor in neighbors:
        edge_data = G.get_edge_data(account_id, neighbor)
        account_info = df[df["account_id"] == neighbor].iloc[0].to_dict()
        account_info["similarity"] = edge_data.get("similarity", 0)
        account_info["distance"] = edge_data.get("distance", 0)
        
        # Add cross-anchor relationship info
        neighbor_anchor = account_info.get("anchor_group", "Unknown")
        account_info["is_cross_anchor"] = (account_anchor != neighbor_anchor) and (account_anchor != "Independent") and (neighbor_anchor != "Independent")
        account_info["anchor_relationship"] = f"{account_anchor} ↔ {neighbor_anchor}" if account_info["is_cross_anchor"] else "Same Anchor"
        
        neighbor_data.append(account_info)
    
    return pd.DataFrame(neighbor_data).sort_values("similarity", ascending=False)


def calculate_cross_anchor_metrics(G, df, edges_df):
    """Calculate metrics for cross-anchor relationships"""
    if len(edges_df) == 0:
        return {}
    
    # Ensure cross-anchor columns exist
    if "is_cross_anchor" not in edges_df.columns:
        anchor_map = dict(zip(df["account_id"], df.get("anchor_group", "Independent")))
        edges_df["src_anchor"] = edges_df["src"].map(anchor_map)
        edges_df["dst_anchor"] = edges_df["dst"].map(anchor_map)
        edges_df["is_cross_anchor"] = (
            (edges_df["src_anchor"] != edges_df["dst_anchor"]) & 
            (edges_df["src_anchor"] != "Independent") & 
            (edges_df["dst_anchor"] != "Independent")
        )
    
    total_edges = len(edges_df)
    cross_anchor_edges = edges_df["is_cross_anchor"].sum() if "is_cross_anchor" in edges_df.columns else 0
    cross_anchor_ratio = cross_anchor_edges / total_edges if total_edges > 0 else 0
    
    # Find anchor bridge accounts (accounts that connect different anchor groups)
    bridge_accounts = {}
    if "is_cross_anchor" in edges_df.columns:
        cross_anchor_edges_subset = edges_df[edges_df["is_cross_anchor"] == True]
        for _, edge in cross_anchor_edges_subset.iterrows():
            for acc in [edge["src"], edge["dst"]]:
                if acc not in bridge_accounts:
                    bridge_accounts[acc] = {
                        "cross_anchor_connections": 0,
                        "connected_anchors": set(),
                        "avg_similarity": []
                    }
                bridge_accounts[acc]["cross_anchor_connections"] += 1
                bridge_accounts[acc]["connected_anchors"].add(edge.get("src_anchor", "Unknown"))
                bridge_accounts[acc]["connected_anchors"].add(edge.get("dst_anchor", "Unknown"))
                bridge_accounts[acc]["avg_similarity"].append(edge.get("similarity", 0))
    
    # Calculate bridge scores
    for acc in bridge_accounts:
        bridge_accounts[acc]["connected_anchors"] = len(bridge_accounts[acc]["connected_anchors"])
        bridge_accounts[acc]["avg_similarity"] = np.mean(bridge_accounts[acc]["avg_similarity"]) if bridge_accounts[acc]["avg_similarity"] else 0
        bridge_accounts[acc]["bridge_score"] = (
            bridge_accounts[acc]["cross_anchor_connections"] * 0.4 +
            bridge_accounts[acc]["connected_anchors"] * 0.3 +
            bridge_accounts[acc]["avg_similarity"] * 30
        )
    
    # Anchor pair connectivity matrix
    anchor_pairs = {}
    if "is_cross_anchor" in edges_df.columns:
        for _, edge in edges_df[edges_df["is_cross_anchor"] == True].iterrows():
            src_anchor = edge.get("src_anchor", "Unknown")
            dst_anchor = edge.get("dst_anchor", "Unknown")
            pair_key = tuple(sorted([src_anchor, dst_anchor]))
            if pair_key not in anchor_pairs:
                anchor_pairs[pair_key] = {"count": 0, "avg_similarity": [], "total_similarity": 0}
            anchor_pairs[pair_key]["count"] += 1
            anchor_pairs[pair_key]["avg_similarity"].append(edge.get("similarity", 0))
            anchor_pairs[pair_key]["total_similarity"] += edge.get("similarity", 0)
    
    for pair_key in anchor_pairs:
        anchor_pairs[pair_key]["avg_similarity"] = np.mean(anchor_pairs[pair_key]["avg_similarity"]) if anchor_pairs[pair_key]["avg_similarity"] else 0
    
    return {
        "total_relationships": total_edges,
        "cross_anchor_relationships": int(cross_anchor_edges),
        "cross_anchor_ratio": cross_anchor_ratio,
        "bridge_accounts": bridge_accounts,
        "anchor_pairs": anchor_pairs,
        "top_bridges": sorted(bridge_accounts.items(), key=lambda x: x[1]["bridge_score"], reverse=True)[:10] if bridge_accounts else []
    }


def identify_cross_anchor_opportunities(df, edges_df, cross_anchor_metrics):
    """Identify opportunities based on cross-anchor relationships"""
    opportunities = []
    
    if "is_cross_anchor" not in edges_df.columns or edges_df["is_cross_anchor"].sum() == 0:
        return pd.DataFrame()
    
    cross_anchor_edges = edges_df[edges_df["is_cross_anchor"] == True].copy()
    
    for _, edge in cross_anchor_edges.iterrows():
        src_acc = edge["src"]
        dst_acc = edge["dst"]
        
        src_data = df[df["account_id"] == src_acc].iloc[0] if len(df[df["account_id"] == src_acc]) > 0 else None
        dst_data = df[df["account_id"] == dst_acc].iloc[0] if len(df[df["account_id"] == dst_acc]) > 0 else None
        
        if src_data is None or dst_data is None:
            continue
        
        # Calculate opportunity score for cross-anchor relationship
        opportunity_score = edge.get("similarity", 0) * 50
        
        # Boost score if one is NTB
        if src_data.get("bri_status") == "NTB" or dst_data.get("bri_status") == "NTB":
            opportunity_score += 20
        
        # Boost score if high lead scores
        avg_lead_score = (src_data.get("lead_score_bri", 0) + dst_data.get("lead_score_bri", 0)) / 2
        opportunity_score += avg_lead_score * 0.3
        
        # Boost if anchor level is close (direct relationships)
        anchor_level_diff = abs(src_data.get("anchor_level", 2) - dst_data.get("anchor_level", 2))
        if anchor_level_diff <= 1:
            opportunity_score += 15
        
        opportunities.append({
            "account_1": src_acc,
            "account_1_name": src_data.get("legal_name", src_acc),
            "account_1_anchor": src_data.get("anchor_group", "Unknown"),
            "account_1_status": src_data.get("bri_status", "Unknown"),
            "account_2": dst_acc,
            "account_2_name": dst_data.get("legal_name", dst_acc),
            "account_2_anchor": dst_data.get("anchor_group", "Unknown"),
            "account_2_status": dst_data.get("bri_status", "Unknown"),
            "similarity": edge.get("similarity", 0),
            "opportunity_score": min(opportunity_score, 100),
            "anchor_pair": f"{src_data.get('anchor_group', 'Unknown')} ↔ {dst_data.get('anchor_group', 'Unknown')}"
        })
    
    return pd.DataFrame(opportunities).sort_values("opportunity_score", ascending=False)


def export_to_csv(df, filename="export.csv"):
    return df.to_csv(index=False).encode('utf-8')


def format_currency(value):
    if value >= 1000000000:
        return f"Rp {value/1000000000:.1f}B"
    elif value >= 1000000:
        return f"Rp {value/1000000:.1f}M"
    elif value >= 1000:
        return f"Rp {value/1000:.1f}K"
    else:
        return f"Rp {value:.0f}"


def main():
    st.markdown('<p class="main-header">OBS Account Relationship Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover ecosystem relationships and opportunities in the Poultry Value Chain</p>', unsafe_allow_html=True)
    
    # UU PDP Compliance Notice
    st.info("""
    🔒 **Data Privacy & Confidentiality**: This dashboard uses NER-based anonymization to protect customer data 
    in compliance with UU PDP (Undang-Undang Perlindungan Data Pribadi) and banking confidentiality requirements. 
    All customer names and sensitive information are automatically anonymized.
    """)
    
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    
    with st.sidebar:
        st.markdown("### Configuration")
        
        st.markdown("#### Data Source")
        data_source = st.radio(
            "Select data source:",
            ["Poultry Ecosystem Demo", "Upload CSV/Excel"],
            label_visibility="collapsed"
        )
        
        if data_source == "Upload CSV/Excel":
            uploaded_file = st.file_uploader("Upload account data", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_uploaded = pd.read_csv(uploaded_file)
                    else:
                        df_uploaded = pd.read_excel(uploaded_file)
                    
                    valid, result = validate_uploaded_data(df_uploaded)
                    if valid:
                        # Apply anonymization for UU PDP compliance
                        result = anonymize_dataframe(result, anonymize_fields=["legal_name"])
                        st.session_state.uploaded_data = result
                        st.success(f"Loaded {len(result)} accounts (data anonymized for privacy compliance)")
                    else:
                        st.error(result)
                        st.session_state.uploaded_data = None
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.session_state.uploaded_data = None
            
            with st.expander("Template & Requirements"):
                st.markdown("**Required columns:**")
                st.code("account_id, segment_code, " + ", ".join(NUMERIC_FEATURES[:5]) + "...")
                st.markdown("**Download sample:**")
                sample_df = generate_poultry_ecosystem_data(n_accounts=20)
                st.download_button(
                    "Download Template",
                    export_to_csv(sample_df),
                    "poultry_ecosystem_template.csv",
                    "text/csv"
                )
        
        st.markdown("---")
        
        if data_source == "Poultry Ecosystem Demo":
            n_accounts = st.slider("Ecosystem Size", min_value=40, max_value=200, value=80, step=20)
        
        st.markdown("#### KNN Parameters")
        n_neighbors = st.slider("K Neighbors", min_value=3, max_value=12, value=5)
        max_distance = st.slider("Similarity Threshold", min_value=0.5, max_value=4.0, value=2.0, step=0.25)
        
        st.markdown("---")
        st.markdown("#### Cross-Anchor Analysis")
        enable_cross_anchor = st.checkbox("Enable Cross-Anchor Relationship Detection", value=True)
        highlight_cross_anchor = st.checkbox("Highlight Cross-Anchor Edges in Network", value=True)
        
        st.markdown("---")
        st.markdown("#### Clustering")
        clustering_method = st.selectbox("Algorithm", ["Louvain", "DBSCAN"])
        
        if clustering_method == "DBSCAN":
            dbscan_eps = st.slider("DBSCAN eps", min_value=0.3, max_value=2.0, value=0.6, step=0.1)
            dbscan_min_samples = st.slider("Min samples", min_value=2, max_value=8, value=3)
        
        st.markdown("---")
        st.markdown("#### Privacy & Compliance")
        enable_anonymization = st.checkbox("Enable Data Anonymization (UU PDP Compliance)", value=True, disabled=True)
        st.caption("Anonymization is always enabled to comply with UU PDP and banking confidentiality requirements.")
        
        # Show anonymization method status
        nlp, ner_available = load_ner_model()
        if ner_available:
            st.success("✓ Using NER-based anonymization (spaCy)")
        else:
            st.info("ℹ Using regex-based anonymization (spaCy model not installed)")
            with st.expander("Install spaCy model for better accuracy"):
                st.code("""
# Install spaCy (if not already installed)
pip install spacy

# Download a model (choose one):
python -m spacy download id_core_web_sm    # Indonesian (recommended)
python -m spacy download xx_ent_wiki_sm    # Multilingual
python -m spacy download en_core_web_sm    # English
                """)
        
        st.markdown("---")
        st.markdown("#### About")
        st.caption("This dashboard identifies behavioral relationships between accounts in the poultry supply chain ecosystem using K-Nearest Neighbors algorithm.")
    
    if data_source == "Upload CSV/Excel" and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data.copy()
    else:
        df = generate_poultry_ecosystem_data(n_accounts=n_accounts if data_source == "Poultry Ecosystem Demo" else 80)
    
    # Apply anonymization for UU PDP compliance
    df = anonymize_dataframe(df, anonymize_fields=["legal_name"])
    
    cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    pipeline, cat_features = build_knn_pipeline(df, n_neighbors=n_neighbors)
    edges_df = find_neighbors_and_edges(df, pipeline, cat_features, max_distance=max_distance)
    G = build_graph(df, edges_df)
    
    if clustering_method == "Louvain":
        community_map = detect_communities_louvain(G)
    else:
        community_map = detect_communities_dbscan(df, cat_features, eps=dbscan_eps, min_samples=dbscan_min_samples)
    
    df["community"] = df["account_id"].map(community_map)
    
    opportunity_scores = calculate_opportunity_score(G, df, community_map)
    df["opportunity_score"] = df["account_id"].map(lambda x: opportunity_scores.get(x, {}).get("total_score", 0))
    df["priority"] = df["account_id"].map(lambda x: opportunity_scores.get(x, {}).get("priority", "Low"))
    
    # Calculate cross-anchor relationship metrics
    if enable_cross_anchor and len(edges_df) > 0:
        cross_anchor_metrics = calculate_cross_anchor_metrics(G, df, edges_df)
        cross_anchor_opportunities = identify_cross_anchor_opportunities(df, edges_df, cross_anchor_metrics)
    else:
        cross_anchor_metrics = {}
        cross_anchor_opportunities = pd.DataFrame()
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Ecosystem Overview", "Account Details", "Anchor Analysis", 
        "RM Opportunities", "Cluster Analysis", "Data Export",
        "Model Training", "Model Inference", "Data Simulation", "Advanced Analytics"
    ])
    
    with tab1:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Accounts", len(df))
        with col2:
            st.metric("Relationships", len(edges_df))
        with col3:
            st.metric("Communities", len(set(community_map.values())))
        with col4:
            bri_existing = len(df[df["bri_status"] == "Existing"])
            st.metric("BRI Existing", bri_existing)
        with col5:
            ntb_accounts = len(df[df["bri_status"] == "NTB"])
            st.metric("NTB Prospects", ntb_accounts)
        with col6:
            if cross_anchor_metrics and cross_anchor_metrics.get("cross_anchor_relationships", 0) > 0:
                cross_anchor_count = cross_anchor_metrics.get("cross_anchor_relationships", 0)
                st.metric("Cross-Anchor Links", cross_anchor_count, 
                         delta=f"{cross_anchor_metrics.get('cross_anchor_ratio', 0):.1%}")
            else:
                st.metric("Cross-Anchor Links", 0)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            color_option = st.radio(
                "Color nodes by:",
                ["Ecosystem Role", "Anchor Group", "Community"],
                horizontal=True
            )
            color_by = {
                "Ecosystem Role": "ecosystem_role",
                "Anchor Group": "anchor_group",
                "Community": "community"
            }[color_option]
            
            fig = create_network_visualization(G, df, community_map, color_by=color_by, highlight_cross_anchor=highlight_cross_anchor)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Ecosystem Composition")
            
            role_counts = df["ecosystem_role"].value_counts()
            fig_roles = px.pie(
                values=role_counts.values,
                names=role_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_roles.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                margin=dict(l=20, r=20, t=20, b=20),
                height=280
            )
            st.plotly_chart(fig_roles, use_container_width=True)
            
            st.markdown("#### Bank Status")
            bri_status_counts = df["bri_status"].value_counts()
            fig_bri = px.bar(
                x=bri_status_counts.index,
                y=bri_status_counts.values,
                color=bri_status_counts.index,
                color_discrete_map={"Existing": "#10B981", "NTB": "#F59E0B", "Unknown": "#94A3B8"}
            )
            fig_bri.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Count",
                margin=dict(l=20, r=20, t=20, b=20),
                height=200
            )
            st.plotly_chart(fig_bri, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Top Connected Accounts (Ecosystem Hubs)")
        
        if len(G.edges()) > 0:
            centrality = nx.degree_centrality(G)
            top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:8]
            
            hub_data = []
            for acc_id, cent in top_hubs:
                acc_info = df[df["account_id"] == acc_id].iloc[0]
                hub_data.append({
                    "Account": acc_id,
                    "Legal Name": acc_info.get("legal_name", acc_id)[:40],
                    "Role": acc_info.get("ecosystem_role", "N/A"),
                    "Anchor": acc_info.get("anchor_group", "N/A"),
                    "BRI Status": acc_info.get("bri_status", "N/A"),
                    "Connections": G.degree(acc_id),
                    "Lead Score": acc_info.get("lead_score_bri", 0)
                })
            
            hub_df = pd.DataFrame(hub_data)
            st.dataframe(hub_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("#### Account Lookup")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            account_options = df.apply(
                lambda x: f"{x['account_id']} - {x.get('legal_name', x['account_id'])[:30]}",
                axis=1
            ).tolist()
            selected_display = st.selectbox("Select Account", options=account_options)
            selected_account = selected_display.split(" - ")[0] if selected_display else None
        
        if selected_account:
            account_data = df[df["account_id"] == selected_account].iloc[0]
            
            with col2:
                st.markdown(f"### {account_data.get('legal_name', selected_account)}")
                
                tag_col1, tag_col2, tag_col3, tag_col4 = st.columns(4)
                with tag_col1:
                    st.info(f"**Role:** {account_data.get('ecosystem_role', 'N/A')}")
                with tag_col2:
                    st.info(f"**Segment:** {account_data['segment_code']}")
                with tag_col3:
                    bri_color = "success" if account_data.get('bri_status') == "Existing" else "warning"
                    if bri_color == "success":
                        st.success(f"**BRI:** {account_data.get('bri_status', 'N/A')}")
                    else:
                        st.warning(f"**BRI:** {account_data.get('bri_status', 'N/A')}")
                with tag_col4:
                    opp_info = opportunity_scores.get(selected_account, {})
                    priority = opp_info.get("priority", "Low")
                    if priority == "High":
                        st.error(f"**Priority:** {priority}")
                    elif priority == "Medium":
                        st.warning(f"**Priority:** {priority}")
                    else:
                        st.info(f"**Priority:** {priority}")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Account Details")
                st.markdown(f"**Account ID:** {selected_account}")
                st.markdown(f"**Anchor Group:** {account_data.get('anchor_group', 'N/A')}")
                st.markdown(f"**Anchor Level:** {account_data.get('anchor_level', 'N/A')}")
                st.markdown(f"**Primary Bank:** {account_data.get('primary_bank', 'N/A')}")
                st.markdown(f"**Community:** {community_map.get(selected_account, 'N/A')}")
            
            with col2:
                st.markdown("##### Financial Metrics")
                st.markdown(f"**Avg Transaction:** {format_currency(account_data['avg_txn_amount_30d'])}")
                st.markdown(f"**Txn Count (30d):** {account_data['txn_count_30d']:.0f}")
                st.markdown(f"**Turnover (90d):** {format_currency(account_data['turnover_90d'])}")
                st.markdown(f"**Cash Withdrawal:** {account_data['cash_withdrawal_ratio_90d']:.1%}")
                st.markdown(f"**Online Ratio:** {account_data['channel_mix_online_ratio']:.1%}")
            
            with col3:
                st.markdown("##### Opportunity Score")
                opp_info = opportunity_scores.get(selected_account, {})
                st.metric("Total Score", f"{opp_info.get('total_score', 0):.0f}/100")
                st.markdown(f"**Network Score:** {opp_info.get('network_score', 0):.1f}")
                st.markdown(f"**Lead Score:** {opp_info.get('lead_score', 0):.1f}")
                st.markdown(f"**Potential Score:** {opp_info.get('potential_score', 0):.1f}")
            
            st.markdown("---")
            st.markdown("#### Connected Accounts")
            
            neighbors_df = get_account_neighbors(G, selected_account, df)
            
            if len(neighbors_df) > 0:
                # Include cross-anchor info if available
                display_cols = [
                    "account_id", "legal_name", "ecosystem_role", "anchor_group",
                    "bri_status", "similarity", "lead_score_bri"
                ]
                if "is_cross_anchor" in neighbors_df.columns:
                    display_cols.append("anchor_relationship")
                
                display_df = neighbors_df[display_cols].copy()
                col_names = ["Account", "Legal Name", "Role", "Anchor", "BRI Status", "Similarity", "Lead Score"]
                if "anchor_relationship" in display_df.columns:
                    col_names.append("Anchor Relationship")
                display_df.columns = col_names
                display_df["Similarity"] = display_df["Similarity"].round(3)
                display_df["Legal Name"] = display_df["Legal Name"].str[:35]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                st.markdown("##### Relationship Insight")
                top_neighbor = neighbors_df.iloc[0]
                st.info(f"""
                **{account_data.get('legal_name', selected_account)}** is most similar to 
                **{top_neighbor.get('legal_name', top_neighbor['account_id'])}** 
                (similarity: {top_neighbor['similarity']:.2f}) due to similar transaction patterns 
                and operational characteristics in the {account_data.get('ecosystem_role', 'ecosystem')}.
                """)
            else:
                st.warning("No connected accounts found within the similarity threshold.")
            
            st.markdown("---")
            st.markdown("#### Account in Network")
            fig_selected = create_network_visualization(G, df, community_map, selected_account, color_by="ecosystem_role", highlight_cross_anchor=highlight_cross_anchor)
            st.plotly_chart(fig_selected, use_container_width=True)
            
            # Show cross-anchor relationship summary
            if len(neighbors_df) > 0 and "is_cross_anchor" in neighbors_df.columns:
                cross_anchor_count = neighbors_df["is_cross_anchor"].sum()
                if cross_anchor_count > 0:
                    st.markdown("---")
                    st.markdown("#### Cross-Anchor Relationships")
                    st.info(f"This account has **{cross_anchor_count}** cross-anchor relationship(s), connecting to accounts from different anchor groups.")
                    
                    cross_anchor_neighbors = neighbors_df[neighbors_df["is_cross_anchor"] == True].copy()
                    if len(cross_anchor_neighbors) > 0:
                        cross_anchor_display = cross_anchor_neighbors[[
                            "account_id", "legal_name", "anchor_group", "anchor_relationship", "similarity"
                        ]].copy()
                        cross_anchor_display.columns = ["Account", "Legal Name", "Anchor Group", "Relationship", "Similarity"]
                        cross_anchor_display["Legal Name"] = cross_anchor_display["Legal Name"].str[:40]
                        cross_anchor_display["Similarity"] = cross_anchor_display["Similarity"].round(3)
                        st.dataframe(cross_anchor_display, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("#### Anchor Group Analysis")
        
        anchor_stats = df.groupby("anchor_group").agg({
            "account_id": "count",
            "turnover_90d": "sum",
            "lead_score_bri": "mean",
            "bri_status": lambda x: (x == "Existing").sum()
        }).reset_index()
        anchor_stats.columns = ["Anchor Group", "Accounts", "Total Turnover", "Avg Lead Score", "BRI Existing"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Accounts by Anchor Group")
            fig_anchor = px.bar(
                anchor_stats,
                x="Anchor Group",
                y="Accounts",
                color="Anchor Group",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_anchor.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_anchor, use_container_width=True)
        
        with col2:
            st.markdown("##### Turnover by Anchor Group")
            fig_turnover = px.pie(
                anchor_stats,
                values="Total Turnover",
                names="Anchor Group",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_turnover.update_layout(height=300)
            st.plotly_chart(fig_turnover, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Anchor Group Details")
        
        display_anchor = anchor_stats.copy()
        display_anchor["Total Turnover"] = display_anchor["Total Turnover"].apply(format_currency)
        display_anchor["Avg Lead Score"] = display_anchor["Avg Lead Score"].round(1)
        st.dataframe(display_anchor, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("##### Supply Chain by Role")
        
        role_anchor = df.groupby(["anchor_group", "ecosystem_role"]).size().reset_index(name="count")
        fig_sunburst = px.sunburst(
            role_anchor,
            path=["anchor_group", "ecosystem_role"],
            values="count",
            color="anchor_group",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_sunburst.update_layout(height=450)
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Cross-Anchor Relationship Analysis")
        
        if cross_anchor_metrics and cross_anchor_metrics.get("total_relationships", 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Relationships", cross_anchor_metrics.get("total_relationships", 0))
            with col2:
                st.metric("Cross-Anchor Links", cross_anchor_metrics.get("cross_anchor_relationships", 0))
            with col3:
                cross_ratio = cross_anchor_metrics.get("cross_anchor_ratio", 0)
                st.metric("Cross-Anchor Ratio", f"{cross_ratio:.1%}")
            with col4:
                bridge_count = len(cross_anchor_metrics.get("bridge_accounts", {}))
                st.metric("Bridge Accounts", bridge_count)
            
            st.markdown("---")
            
            # Anchor Pair Connectivity
            if cross_anchor_metrics.get("anchor_pairs"):
                st.markdown("##### Anchor Pair Connectivity Matrix")
                anchor_pairs_data = []
                for pair_key, pair_data in cross_anchor_metrics["anchor_pairs"].items():
                    anchor_pairs_data.append({
                        "Anchor Pair": f"{pair_key[0]} ↔ {pair_key[1]}",
                        "Relationships": pair_data["count"],
                        "Avg Similarity": f"{pair_data['avg_similarity']:.3f}",
                        "Strength": "Strong" if pair_data["avg_similarity"] > 0.5 else "Moderate" if pair_data["avg_similarity"] > 0.3 else "Weak"
                    })
                
                if anchor_pairs_data:
                    pairs_df = pd.DataFrame(anchor_pairs_data).sort_values("Relationships", ascending=False)
                    st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pairs = px.bar(
                            pairs_df,
                            x="Anchor Pair",
                            y="Relationships",
                            color="Avg Similarity",
                            color_continuous_scale="Viridis",
                            title="Cross-Anchor Relationship Count"
                        )
                        fig_pairs.update_layout(height=300, xaxis_tickangle=-45)
                        st.plotly_chart(fig_pairs, use_container_width=True)
                    
                    with col2:
                        fig_sim = px.scatter(
                            pairs_df,
                            x="Relationships",
                            y="Avg Similarity",
                            size="Relationships",
                            color="Anchor Pair",
                            hover_name="Anchor Pair",
                            title="Relationship Strength vs Count"
                        )
                        fig_sim.update_layout(height=300)
                        st.plotly_chart(fig_sim, use_container_width=True)
            
            st.markdown("---")
            st.markdown("##### Top Bridge Accounts (Cross-Anchor Connectors)")
            
            if cross_anchor_metrics.get("top_bridges"):
                bridge_data = []
                for acc_id, bridge_info in cross_anchor_metrics["top_bridges"]:
                    acc_info = df[df["account_id"] == acc_id].iloc[0] if len(df[df["account_id"] == acc_id]) > 0 else None
                    if acc_info is not None:
                        bridge_data.append({
                            "Account": acc_id,
                            "Legal Name": acc_info.get("legal_name", acc_id)[:40],
                            "Anchor": acc_info.get("anchor_group", "Unknown"),
                            "Role": acc_info.get("ecosystem_role", "N/A"),
                            "Bridge Score": f"{bridge_info['bridge_score']:.1f}",
                            "Cross-Anchor Links": bridge_info["cross_anchor_connections"],
                            "Connected Anchors": bridge_info["connected_anchors"],
                            "Avg Similarity": f"{bridge_info['avg_similarity']:.3f}",
                            "BRI Status": acc_info.get("bri_status", "Unknown")
                        })
                
                if bridge_data:
                    bridge_df = pd.DataFrame(bridge_data)
                    st.dataframe(bridge_df, use_container_width=True, hide_index=True)
            else:
                st.info("No bridge accounts found. Try adjusting similarity threshold to find more cross-anchor relationships.")
            
            st.markdown("---")
            st.markdown("##### Cross-Anchor Opportunities")
            
            if len(cross_anchor_opportunities) > 0:
                st.markdown("**Top Cross-Anchor Relationship Opportunities**")
                display_opps = cross_anchor_opportunities.head(20).copy()
                display_opps = display_opps[[
                    "account_1_name", "account_1_anchor", "account_1_status",
                    "account_2_name", "account_2_anchor", "account_2_status",
                    "similarity", "opportunity_score", "anchor_pair"
                ]]
                display_opps.columns = [
                    "Account 1", "Anchor 1", "Status 1",
                    "Account 2", "Anchor 2", "Status 2",
                    "Similarity", "Opp Score", "Anchor Pair"
                ]
                display_opps["Account 1"] = display_opps["Account 1"].str[:35]
                display_opps["Account 2"] = display_opps["Account 2"].str[:35]
                display_opps["Similarity"] = display_opps["Similarity"].round(3)
                st.dataframe(display_opps, use_container_width=True, hide_index=True)
                
                st.info("""
                **Cross-Anchor Opportunities** represent relationships between accounts from different anchor groups. 
                These are valuable for:
                - **Market Expansion**: Understanding ecosystem connections across anchor boundaries
                - **NTB Acquisition**: Identifying accounts that bridge different anchor ecosystems
                - **Risk Management**: Detecting unusual cross-anchor transaction patterns
                - **Strategic Partnerships**: Finding potential collaboration opportunities
                """)
            else:
                st.warning("No cross-anchor opportunities found. This may indicate strong anchor group isolation or need to adjust similarity thresholds.")
        else:
            st.info("No cross-anchor relationships detected. This could mean accounts are primarily connected within their own anchor groups, or similarity thresholds need adjustment.")
    
    with tab4:
        st.markdown("#### RM Opportunity Pipeline")
        
        col1, col2, col3 = st.columns(3)
        high_priority = df[df["priority"] == "High"]
        medium_priority = df[df["priority"] == "Medium"]
        ntb_high_value = df[(df["bri_status"] == "NTB") & (df["lead_score_bri"] >= 75)]
        
        with col1:
            st.metric("High Priority", len(high_priority))
        with col2:
            st.metric("Medium Priority", len(medium_priority))
        with col3:
            st.metric("NTB High-Value", len(ntb_high_value))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Priority Distribution")
            priority_counts = df["priority"].value_counts()
            fig_priority = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                color=priority_counts.index,
                color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
                hole=0.4
            )
            fig_priority.update_layout(height=300)
            st.plotly_chart(fig_priority, use_container_width=True)
        
        with col2:
            st.markdown("##### Opportunity Score Distribution")
            fig_score = px.histogram(
                df,
                x="opportunity_score",
                nbins=20,
                color="priority",
                color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
            )
            fig_score.update_layout(height=300, xaxis_title="Opportunity Score", yaxis_title="Count")
            st.plotly_chart(fig_score, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### High Priority Accounts (Recommended for RM Action)")
        
        if len(high_priority) > 0:
            display_high = high_priority[[
                "account_id", "legal_name", "ecosystem_role", "anchor_group",
                "bri_status", "segment_code", "opportunity_score", "lead_score_bri"
            ]].sort_values("opportunity_score", ascending=False).copy()
            display_high.columns = ["Account", "Legal Name", "Role", "Anchor", "BRI Status", "Segment", "Opp Score", "Lead Score"]
            display_high["Legal Name"] = display_high["Legal Name"].str[:40]
            st.dataframe(display_high, use_container_width=True, hide_index=True)
        else:
            st.info("No high-priority accounts with current thresholds.")
        
        st.markdown("---")
        st.markdown("##### NTB Acquisition Targets")
        
        ntb_targets = df[df["bri_status"] == "NTB"].sort_values("lead_score_bri", ascending=False).head(15)
        if len(ntb_targets) > 0:
            display_ntb = ntb_targets[[
                "account_id", "legal_name", "ecosystem_role", "anchor_group",
                "primary_bank", "lead_score_bri", "turnover_90d"
            ]].copy()
            display_ntb.columns = ["Account", "Legal Name", "Role", "Anchor", "Current Bank", "Lead Score", "Turnover 90d"]
            display_ntb["Legal Name"] = display_ntb["Legal Name"].str[:40]
            display_ntb["Turnover 90d"] = display_ntb["Turnover 90d"].apply(format_currency)
            st.dataframe(display_ntb, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("##### Opportunity Scoring Methodology")
        st.markdown("""
        The opportunity score (0-100) is calculated based on:
        - **Network Score (20%)**: Account connectivity in the ecosystem network
        - **Influence Score (15%)**: How often the account bridges different communities
        - **Ecosystem Score (15%)**: Size and strength of the account's ecosystem community
        - **Lead Score (25%)**: BRI lead scoring based on financial behavior
        - **Potential Score (25%)**: NTB status and anchor proximity (closer = higher potential)
        """)
    
    with tab5:
        st.markdown("#### Cluster Analysis")
        
        louvain_map = detect_communities_louvain(G)
        dbscan_map = detect_communities_dbscan(df, cat_features, eps=0.6, min_samples=3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Louvain Communities")
            louvain_clusters = len(set(louvain_map.values()))
            st.metric("Communities Found", louvain_clusters)
            
            louvain_sizes = pd.Series(louvain_map).value_counts().reset_index()
            louvain_sizes.columns = ["Community", "Size"]
            fig_louv = px.bar(louvain_sizes, x="Community", y="Size", color="Size",
                             color_continuous_scale="Blues")
            fig_louv.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig_louv, use_container_width=True)
        
        with col2:
            st.markdown("##### DBSCAN Clusters")
            dbscan_clusters = len(set(dbscan_map.values()))
            noise_points = sum(1 for v in dbscan_map.values() if v == -1)
            st.metric("Clusters", dbscan_clusters - (1 if -1 in dbscan_map.values() else 0))
            st.metric("Outliers", noise_points)
            
            dbscan_sizes = pd.Series(dbscan_map).value_counts().reset_index()
            dbscan_sizes.columns = ["Cluster", "Size"]
            fig_db = px.bar(dbscan_sizes, x="Cluster", y="Size", color="Size",
                           color_continuous_scale="Greens")
            fig_db.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig_db, use_container_width=True)
        
        st.markdown("---")
        st.markdown("##### Community Details")
        
        selected_community = st.selectbox("Select Community", options=sorted(df["community"].unique()))
        
        if selected_community is not None:
            community_accounts = df[df["community"] == selected_community]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Community {selected_community}** - {len(community_accounts)} accounts")
                
                role_dist = community_accounts["ecosystem_role"].value_counts()
                fig_role = px.pie(values=role_dist.values, names=role_dist.index, hole=0.4)
                fig_role.update_layout(height=250, title="Roles in Community")
                st.plotly_chart(fig_role, use_container_width=True)
            
            with col2:
                st.markdown("**Community Statistics**")
                stats = {
                    "Total Accounts": len(community_accounts),
                    "BRI Existing": len(community_accounts[community_accounts["bri_status"] == "Existing"]),
                    "NTB Prospects": len(community_accounts[community_accounts["bri_status"] == "NTB"]),
                    "Avg Lead Score": community_accounts["lead_score_bri"].mean(),
                    "Total Turnover": format_currency(community_accounts["turnover_90d"].sum())
                }
                for key, val in stats.items():
                    if isinstance(val, float):
                        st.markdown(f"**{key}:** {val:.1f}")
                    else:
                        st.markdown(f"**{key}:** {val}")
            
            st.markdown("**Accounts in Community**")
            display_comm = community_accounts[[
                "account_id", "legal_name", "ecosystem_role", "anchor_group", "bri_status", "lead_score_bri"
            ]].copy()
            display_comm.columns = ["Account", "Legal Name", "Role", "Anchor", "BRI Status", "Lead Score"]
            display_comm["Legal Name"] = display_comm["Legal Name"].str[:40]
            st.dataframe(display_comm, use_container_width=True, hide_index=True)
    
    with tab6:
        st.markdown("#### Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "Export Full Dataset",
                export_to_csv(df),
                "poultry_ecosystem_accounts.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if len(edges_df) > 0:
                st.download_button(
                    "Export Relationships",
                    export_to_csv(edges_df),
                    "ecosystem_relationships.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        with col3:
            opp_df = pd.DataFrame([
                {"account_id": k, **v} for k, v in opportunity_scores.items()
            ])
            st.download_button(
                "Export Opportunities",
                export_to_csv(opp_df),
                "rm_opportunities.csv",
                "text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        st.markdown("#### Full Dataset Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### Relationship Edges")
        if len(edges_df) > 0:
            st.dataframe(edges_df.head(50), use_container_width=True, hide_index=True)
            st.caption(f"Showing first 50 of {len(edges_df)} relationships")
        else:
            st.warning("No relationships found. Try adjusting the similarity threshold.")
        
        st.markdown("---")
        st.markdown("#### Feature Correlations")
        numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab7:
        st.markdown("#### Model Training")
        st.info("Train ML models from your data for relationship detection and predictions.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("##### Training Configuration")
            model_type = st.selectbox(
                "Model Type",
                ["KNN Relationship Model", "Predictive Model (Lead Score)", 
                 "Predictive Model (Turnover)", "Full Model Suite"]
            )
            
            if model_type == "KNN Relationship Model":
                n_neighbors_train = st.slider("K Neighbors", 3, 12, 6, key="train_k")
                model_name = st.text_input("Model Name", "knn_relationship_model")
            
            elif "Predictive Model" in model_type:
                target_col = "lead_score_bri" if "Lead Score" in model_type else "turnover_90d"
                model_algorithm = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting"])
                test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
                model_name = st.text_input("Model Name", f"predictive_{target_col}")
            
            else:  # Full Model Suite
                n_neighbors_train = st.slider("K Neighbors", 3, 12, 6, key="train_k_full")
                model_name = "full_model_suite"
            
            train_button = st.button("Train Model", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("##### Training Data")
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Total accounts: {len(df)}")
        
        if train_button:
            try:
                with st.spinner("Training model... This may take a few moments."):
                    trainer = ModelTrainer()
                    
                    if model_type == "KNN Relationship Model":
                        # Validate required columns
                        required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        else:
                            pipeline, cat_features = trainer.train_knn_model(
                                df, n_neighbors=n_neighbors_train,
                                model_name=model_name, save_model=True
                            )
                            st.success(f"✅ KNN model '{model_name}' trained successfully!")
                            metadata = trainer.get_model_metadata(model_name)
                            if metadata:
                                with st.expander("Model Metadata"):
                                    st.json(metadata)
                    
                    elif "Predictive Model" in model_type:
                        # Validate target column exists
                        if target_col not in df.columns:
                            st.error(f"Target column '{target_col}' not found in data. Available columns: {', '.join(df.columns[:10])}...")
                        else:
                            model, metrics = trainer.train_predictive_model(
                                df,
                                target_column=target_col,
                                model_type=model_algorithm.lower().replace(" ", "_"),
                                model_name=model_name,
                                test_size=test_size,
                                save_model=True
                            )
                            st.success(f"✅ Predictive model '{model_name}' trained successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Test R²", f"{metrics['test_r2']:.4f}")
                            with col2:
                                st.metric("Test MAE", f"{metrics['test_mae']:.2f}")
                            with col3:
                                st.metric("CV R² Mean", f"{metrics['cv_r2_mean']:.4f}")
                            
                            with st.expander("Detailed Metrics"):
                                st.json(metrics)
                    
                    else:  # Full Model Suite
                        st.info("Training full model suite (this may take a while)...")
                        models = trainer.train_full_model_suite(
                            df, n_neighbors=n_neighbors_train, save_models=True
                        )
                        st.success(f"✅ Full model suite trained successfully!")
                        st.info(f"Trained {len(models)} models: {', '.join(models.keys())}")
                        
                        # Show summary
                        for model_key, model_obj in models.items():
                            with st.expander(f"📦 {model_key}"):
                                metadata = trainer.get_model_metadata(model_key)
                                if metadata:
                                    st.json(metadata)
            
            except Exception as e:
                import traceback
                st.error(f"❌ Error training model: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        
        st.markdown("---")
        st.markdown("##### Available Trained Models")
        trainer = ModelTrainer()
        available_models = trainer.list_available_models()
        
        if available_models:
            for model_name in available_models:
                with st.expander(f"📦 {model_name}"):
                    metadata = trainer.get_model_metadata(model_name)
                    if metadata:
                        st.json(metadata)
        else:
            st.info("No trained models found. Train a model above.")
    
    with tab8:
        st.markdown("#### Model Inference")
        st.info("Use trained models to make predictions and detect relationships.")
        
        inference = ModelInference()
        available_models = inference.list_available_models()
        
        if not available_models:
            st.warning("No trained models available. Please train models in the 'Model Training' tab first.")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("##### Inference Configuration")
                selected_model = st.selectbox("Select Model", available_models)
                
                inference_type = st.selectbox(
                    "Inference Type",
                    ["Relationship Detection", "Target Prediction", "Batch Prediction"]
                )
                
                if inference_type == "Relationship Detection":
                    n_neighbors_inf = st.slider("K Neighbors", 3, 12, 5, key="inf_k")
                    max_dist_inf = st.slider("Max Distance", 0.5, 4.0, 2.0, 0.25, key="inf_dist")
                
                elif inference_type == "Target Prediction":
                    target_col = st.selectbox(
                        "Target Variable",
                        ["lead_score_bri", "turnover_90d", "avg_txn_amount_30d"]
                    )
                
                run_inference = st.button("Run Inference", type="primary", use_container_width=True)
            
            with col2:
                st.markdown("##### Input Data")
                st.dataframe(df.head(20), use_container_width=True)
            
            if run_inference:
                with st.spinner("Running inference..."):
                    try:
                        if inference_type == "Relationship Detection":
                            edges_df_inf, distances, indices = inference.predict_relationship(
                                df, model_name=selected_model,
                                n_neighbors=n_neighbors_inf, max_distance=max_dist_inf
                            )
                            st.success(f"✅ Found {len(edges_df_inf)} relationships!")
                            st.dataframe(edges_df_inf.head(50), use_container_width=True)
                        
                        elif inference_type == "Target Prediction":
                            predictions, confidence = inference.predict_target(
                                df, target_column=target_col, model_name=f"predictive_{target_col}"
                            )
                            result_df = df.copy()
                            result_df[f"predicted_{target_col}"] = predictions
                            
                            st.success(f"✅ Predictions generated for {target_col}!")
                            st.metric("Mean Prediction", f"{np.mean(predictions):.2f}")
                            if confidence:
                                st.json(confidence)
                            
                            display_cols = ["account_id", "legal_name", target_col, f"predicted_{target_col}"]
                            if target_col in result_df.columns:
                                result_df["prediction_error"] = abs(result_df[target_col] - result_df[f"predicted_{target_col}"])
                                display_cols.append("prediction_error")
                            
                            st.dataframe(result_df[display_cols].head(50), use_container_width=True)
                        
                        else:  # Batch Prediction
                            result_df = inference.batch_predict(df)
                            st.success("✅ Batch predictions completed!")
                            pred_cols = [col for col in result_df.columns if col.startswith("predicted_")]
                            st.dataframe(result_df[["account_id", "legal_name"] + pred_cols].head(50), use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
    
    with tab9:
        st.markdown("#### Data Simulation")
        st.info("Generate synthetic data for testing and development. Simulate both internal (BRI) and external data sources.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("##### Simulation Configuration")
            data_type = st.selectbox(
                "Data Type",
                ["Internal (BRI)", "External (Other Banks)", "Combined"]
            )
            
            n_accounts_sim = st.slider("Number of Accounts", 20, 500, 100, 20)
            
            if data_type == "Internal (BRI)":
                include_historical = st.checkbox("Include Historical Data", value=True)
                historical_months = st.slider("Historical Months", 3, 12, 6) if include_historical else 0
            
            elif data_type == "External (Other Banks)":
                data_source = st.selectbox(
                    "External Source",
                    ["external_bank", "public_registry", "credit_bureau"]
                )
                include_temporal = st.checkbox("Include Temporal Data", value=True)
                time_periods = st.slider("Time Periods", 1, 6, 3) if include_temporal else 0
            
            else:  # Combined
                n_internal = st.slider("Internal Accounts", 10, 200, 60, 10)
                n_external = st.slider("External Accounts", 10, 200, 40, 10)
                include_temporal = st.checkbox("Include Temporal Data", value=True)
            
            add_noise = st.checkbox("Add Noise", value=False)
            noise_level = st.slider("Noise Level", 0.01, 0.20, 0.05, 0.01) if add_noise else 0.0
            
            simulate_button = st.button("Generate Data", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("##### Simulated Data Preview")
            
            if simulate_button:
                with st.spinner("Generating simulated data..."):
                    try:
                        simulator = DataSimulator()
                        
                        if data_type == "Internal (BRI)":
                            sim_df = simulator.simulate_internal_data(
                                n_accounts=n_accounts_sim,
                                include_historical=include_historical,
                                historical_months=historical_months if include_historical else 0
                            )
                        
                        elif data_type == "External (Other Banks)":
                            sim_df = simulator.simulate_external_data(
                                n_accounts=n_accounts_sim,
                                data_source=data_source,
                                include_temporal=include_temporal,
                                time_periods=time_periods if include_temporal else 0
                            )
                        
                        else:  # Combined
                            sim_df = simulator.simulate_combined_dataset(
                                n_internal=n_internal,
                                n_external=n_external,
                                include_temporal=include_temporal
                            )
                        
                        if add_noise:
                            sim_df = simulator.add_noise_to_data(sim_df, noise_level=noise_level)
                        
                        st.success(f"✅ Generated {len(sim_df)} accounts!")
                        st.dataframe(sim_df.head(50), use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Accounts", len(sim_df))
                        with col2:
                            st.metric("Internal", len(sim_df[sim_df["data_source"].str.contains("internal", case=False)]) if "data_source" in sim_df.columns else 0)
                        with col3:
                            st.metric("External", len(sim_df[sim_df["data_source"].str.contains("external", case=False)]) if "data_source" in sim_df.columns else 0)
                        with col4:
                            st.metric("NTB", len(sim_df[sim_df["bri_status"] == "NTB"]) if "bri_status" in sim_df.columns else 0)
                        
                        # Download button
                        st.download_button(
                            "Download Simulated Data",
                            sim_df.to_csv(index=False).encode('utf-8'),
                            f"simulated_{data_type.lower().replace(' ', '_')}_{len(sim_df)}.csv",
                            "text/csv"
                        )
                    
                    except Exception as e:
                        st.error(f"Error generating data: {str(e)}")
            else:
                st.info("Configure settings and click 'Generate Data' to create simulated dataset.")
    
    with tab10:
        st.markdown("#### Advanced Analytics")
        st.info("Advanced analytics including anomaly detection, risk scoring, forecasting, and trend analysis.")
        
        analytics_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive", "Anomaly Detection", "Risk Scoring", "Growth Analysis", "Forecasting",
             "Lead Quality Scoring", "Customer Lifetime Value", "Cross-Sell Opportunities", 
             "Relationship Strength", "Churn Risk Prediction", "Portfolio Health", "RM Action Prioritization"]
        )
        
        run_analytics = st.button("Run Analysis", type="primary")
        
        if run_analytics:
            with st.spinner("Running advanced analytics..."):
                try:
                    analytics = AdvancedAnalytics()
                    
                    if analytics_type == "Comprehensive":
                        results = analytics.create_analytics_dashboard_data(df)
                        
                        # Anomalies
                        st.markdown("##### Anomaly Detection")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Anomalies", results["anomalies"]["total"])
                        with col2:
                            st.metric("Critical", results["anomalies"]["critical"])
                        with col3:
                            st.metric("High", results["anomalies"]["high"])
                        with col4:
                            st.metric("Medium", results["anomalies"]["data"][results["anomalies"]["data"]["anomaly_severity"] == "Medium"].shape[0])
                        
                        st.dataframe(results["anomalies"]["data"].head(20), use_container_width=True)
                        
                        # Risk Scoring
                        st.markdown("---")
                        st.markdown("##### Risk Scoring")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Risk Score", f"{results['risk']['avg_risk_score']:.2f}")
                            st.metric("High Risk Accounts", results["risk"]["high_risk_count"])
                        with col2:
                            risk_dist = results["risk"]["data"]["risk_level"].value_counts()
                            fig_risk = px.pie(values=risk_dist.values, names=risk_dist.index, hole=0.4)
                            st.plotly_chart(fig_risk, use_container_width=True)
                        
                        st.dataframe(results["risk"]["data"].head(20), use_container_width=True)
                        
                        # Growth
                        if "avg_growth_rate" in results["growth"] and results["growth"]["avg_growth_rate"] != 0:
                            st.markdown("---")
                            st.markdown("##### Growth Analysis")
                            st.metric("Average Growth Rate", f"{results['growth']['avg_growth_rate']:.2f}%")
                        
                        # Forecasting
                        st.markdown("---")
                        st.markdown("##### Opportunity Forecasting")
                        forecast_cols = [col for col in results["forecast"].columns if "forecasted_opportunity_score" in col]
                        if forecast_cols:
                            sample_forecast = results["forecast"][["account_id", "legal_name"] + forecast_cols[:3]].head(10)
                            st.dataframe(sample_forecast, use_container_width=True)
                    
                    elif analytics_type == "Anomaly Detection":
                        df_anomalies = analytics.detect_anomalies(df)
                        st.success(f"✅ Detected {df_anomalies['anomaly_flag'].sum()} anomalies")
                        
                        fig_anomalies = px.scatter(
                            df_anomalies,
                            x="turnover_90d",
                            y="lead_score_bri",
                            color="anomaly_severity",
                            size="anomaly_score",
                            hover_data=["account_id", "legal_name"],
                            title="Anomaly Detection Results"
                        )
                        st.plotly_chart(fig_anomalies, use_container_width=True)
                        
                        st.dataframe(df_anomalies[["account_id", "legal_name", "anomaly_flag", "anomaly_score", "anomaly_severity"]].head(50), use_container_width=True)
                    
                    elif analytics_type == "Risk Scoring":
                        df_risk = analytics.risk_scoring(df)
                        st.success("✅ Risk scores calculated")
                        
                        fig_risk = px.histogram(
                            df_risk,
                            x="risk_score",
                            color="risk_level",
                            nbins=30,
                            title="Risk Score Distribution"
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                        st.dataframe(df_risk[["account_id", "legal_name", "risk_score", "risk_level"]].sort_values("risk_score", ascending=False).head(50), use_container_width=True)
                    
                    elif analytics_type == "Growth Analysis":
                        df_growth = analytics.calculate_growth_metrics(df)
                        st.success("✅ Growth metrics calculated")
                        
                        if "avg_growth_rate" in df_growth.columns:
                            st.metric("Average Growth Rate", f"{df_growth['avg_growth_rate'].mean():.2f}%")
                            growth_data = df_growth[["account_id", "legal_name", "avg_growth_rate", "growth_trend"]].dropna()
                            st.dataframe(growth_data.head(50), use_container_width=True)
                        else:
                            st.info("No historical data available for growth analysis.")
                    
                    elif analytics_type == "Forecasting":
                        account_select = st.selectbox("Select Account for Forecasting", df["account_id"].tolist())
                        months = st.slider("Months Ahead", 1, 12, 3)
                        method = st.selectbox("Forecast Method", ["trend", "moving_average", "exponential"])
                        
                        forecast_result = analytics.predict_future_turnover(df, account_select, months_ahead=months, method=method)
                        
                        if "error" not in forecast_result:
                            st.success(f"✅ Forecast generated for {account_select}")
                            
                            fig_forecast = go.Figure()
                            fig_forecast.add_trace(go.Scatter(
                                y=forecast_result["predictions"],
                                mode="lines+markers",
                                name="Forecast",
                                line=dict(color="blue", width=2)
                            ))
                            fig_forecast.add_trace(go.Scatter(
                                y=forecast_result["confidence_upper"],
                                mode="lines",
                                name="Upper Bound",
                                line=dict(color="lightblue", dash="dash")
                            ))
                            fig_forecast.add_trace(go.Scatter(
                                y=forecast_result["confidence_lower"],
                                mode="lines",
                                name="Lower Bound",
                                line=dict(color="lightblue", dash="dash"),
                                fill="tonexty"
                            ))
                            fig_forecast.update_layout(
                                title=f"Turnover Forecast - {account_select}",
                                xaxis_title="Month",
                                yaxis_title="Turnover",
                                height=400
                            )
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            st.json(forecast_result)
                    
                    elif analytics_type == "Lead Quality Scoring":
                        df_lead_quality = analytics.calculate_lead_quality_score(df, G, opportunity_scores)
                        st.success("✅ Lead quality scores calculated")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("A+ Grade", len(df_lead_quality[df_lead_quality["lead_quality_grade"] == "A+"]))
                        with col2:
                            st.metric("A Grade", len(df_lead_quality[df_lead_quality["lead_quality_grade"] == "A"]))
                        with col3:
                            st.metric("B Grade", len(df_lead_quality[df_lead_quality["lead_quality_grade"] == "B"]))
                        with col4:
                            st.metric("Avg Score", f"{df_lead_quality['lead_quality_score'].mean():.1f}")
                        
                        fig_lead = px.histogram(
                            df_lead_quality,
                            x="lead_quality_score",
                            color="lead_quality_grade",
                            nbins=20,
                            title="Lead Quality Score Distribution"
                        )
                        st.plotly_chart(fig_lead, use_container_width=True)
                        
                        display_cols = ["account_id", "legal_name", "lead_quality_score", "lead_quality_grade",
                                       "financial_quality", "network_value", "relationship_potential", "behavioral_quality"]
                        st.dataframe(df_lead_quality[display_cols].sort_values("lead_quality_score", ascending=False).head(50), use_container_width=True)
                    
                    elif analytics_type == "Customer Lifetime Value":
                        df_clv = analytics.calculate_customer_lifetime_value(df)
                        st.success("✅ CLV estimates calculated")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategic Tier", len(df_clv[df_clv["clv_tier"] == "Strategic"]))
                        with col2:
                            st.metric("Enterprise Tier", len(df_clv[df_clv["clv_tier"] == "Enterprise"]))
                        with col3:
                            st.metric("Premium Tier", len(df_clv[df_clv["clv_tier"] == "Premium"]))
                        with col4:
                            avg_clv = df_clv["clv_estimate"].mean() / 1e9
                            st.metric("Avg CLV", f"Rp {avg_clv:.2f}B")
                        
                        fig_clv = px.box(
                            df_clv,
                            x="clv_tier",
                            y="clv_estimate",
                            title="CLV Distribution by Tier"
                        )
                        st.plotly_chart(fig_clv, use_container_width=True)
                        
                        st.dataframe(df_clv[["account_id", "legal_name", "clv_estimate", "clv_tier"]].sort_values("clv_estimate", ascending=False).head(50), use_container_width=True)
                    
                    elif analytics_type == "Cross-Sell Opportunities":
                        df_crosssell = analytics.identify_cross_sell_opportunities(df, G)
                        st.success("✅ Cross-sell opportunities identified")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Critical", len(df_crosssell[df_crosssell["cross_sell_priority"] == "Critical"]))
                        with col2:
                            st.metric("High", len(df_crosssell[df_crosssell["cross_sell_priority"] == "High"]))
                        with col3:
                            st.metric("Medium", len(df_crosssell[df_crosssell["cross_sell_priority"] == "Medium"]))
                        with col4:
                            st.metric("Avg Score", f"{df_crosssell['cross_sell_score'].mean():.1f}")
                        
                        st.dataframe(df_crosssell[["account_id", "legal_name", "cross_sell_score", "cross_sell_priority", "opportunity_reasons"]].sort_values("cross_sell_score", ascending=False).head(50), use_container_width=True)
                    
                    elif analytics_type == "Relationship Strength":
                        df_relationship = analytics.calculate_relationship_strength(df, G)
                        st.success("✅ Relationship strength scores calculated")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategic", len(df_relationship[df_relationship["relationship_tier"] == "Strategic"]))
                        with col2:
                            st.metric("Strong", len(df_relationship[df_relationship["relationship_tier"] == "Strong"]))
                        with col3:
                            st.metric("Moderate", len(df_relationship[df_relationship["relationship_tier"] == "Moderate"]))
                        with col4:
                            st.metric("Avg Strength", f"{df_relationship['relationship_strength'].mean():.1f}")
                        
                        fig_rel = px.histogram(
                            df_relationship,
                            x="relationship_strength",
                            color="relationship_tier",
                            nbins=20,
                            title="Relationship Strength Distribution"
                        )
                        st.plotly_chart(fig_rel, use_container_width=True)
                        
                        st.dataframe(df_relationship[["account_id", "legal_name", "relationship_strength", "relationship_tier"]].sort_values("relationship_strength", ascending=False).head(50), use_container_width=True)
                    
                    elif analytics_type == "Churn Risk Prediction":
                        df_churn = analytics.predict_churn_risk(df, G)
                        st.success("✅ Churn risk scores calculated")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Critical Risk", len(df_churn[df_churn["churn_risk_level"] == "Critical"]))
                        with col2:
                            st.metric("High Risk", len(df_churn[df_churn["churn_risk_level"] == "High"]))
                        with col3:
                            st.metric("Medium Risk", len(df_churn[df_churn["churn_risk_level"] == "Medium"]))
                        with col4:
                            st.metric("Avg Risk", f"{df_churn['churn_risk_score'].mean():.1f}")
                        
                        fig_churn = px.scatter(
                            df_churn,
                            x="turnover_90d",
                            y="churn_risk_score",
                            color="churn_risk_level",
                            size="lead_score_bri",
                            hover_data=["account_id", "legal_name"],
                            title="Churn Risk vs Portfolio Value"
                        )
                        st.plotly_chart(fig_churn, use_container_width=True)
                        
                        st.dataframe(df_churn[["account_id", "legal_name", "churn_risk_score", "churn_risk_level"]].sort_values("churn_risk_score", ascending=False).head(50), use_container_width=True)
                    
                    elif analytics_type == "Portfolio Health":
                        health_metrics = analytics.calculate_portfolio_health(df, G)
                        st.success("✅ Portfolio health analysis completed")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Health Grade", health_metrics.get("health_grade", "N/A"))
                            st.metric("Health Score", f"{health_metrics.get('overall_health_score', 0):.1f}/100")
                        with col2:
                            st.metric("Existing Rate", f"{health_metrics.get('existing_rate', 0):.1%}")
                            st.metric("NTB Rate", f"{health_metrics.get('ntb_rate', 0):.1%}")
                        with col3:
                            st.metric("Avg Lead Score", f"{health_metrics.get('avg_lead_score', 0):.1f}")
                            st.metric("High Quality Leads", health_metrics.get("high_quality_leads", 0))
                        with col4:
                            total_value = health_metrics.get("total_portfolio_value", 0) / 1e9
                            st.metric("Portfolio Value", f"Rp {total_value:.2f}B")
                            st.metric("Hub Accounts", health_metrics.get("hub_accounts", 0))
                        
                        st.markdown("##### Portfolio Health Details")
                        st.json(health_metrics)
                    
                    elif analytics_type == "RM Action Prioritization":
                        df_rm_actions = analytics.create_rm_action_prioritization(df, G, opportunity_scores)
                        st.success("✅ RM action prioritization completed")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Immediate Action", len(df_rm_actions[df_rm_actions["rm_action_priority_level"] == "Immediate Action"]))
                        with col2:
                            st.metric("Priority", len(df_rm_actions[df_rm_actions["rm_action_priority_level"] == "Priority"]))
                        with col3:
                            st.metric("Follow-up", len(df_rm_actions[df_rm_actions["rm_action_priority_level"] == "Follow-up"]))
                        with col4:
                            st.metric("Monitor", len(df_rm_actions[df_rm_actions["rm_action_priority_level"] == "Monitor"]))
                        
                        # Top priority accounts
                        st.markdown("##### Top Priority Accounts for RM Action")
                        top_priority = df_rm_actions.nlargest(20, "rm_action_priority")[[
                            "account_id", "legal_name", "rm_action_priority", "rm_action_priority_level",
                            "recommended_rm_action", "lead_quality_score", "churn_risk_score"
                        ]]
                        st.dataframe(top_priority, use_container_width=True)
                        
                        # Action distribution
                        fig_actions = px.bar(
                            df_rm_actions["rm_action_priority_level"].value_counts().reset_index(),
                            x="index",
                            y="rm_action_priority_level",
                            title="RM Action Priority Distribution"
                        )
                        st.plotly_chart(fig_actions, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error running analytics: {str(e)}")


if __name__ == "__main__":
    main()
