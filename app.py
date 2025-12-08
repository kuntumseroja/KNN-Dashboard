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
    
    for i, (d_row, idx_row) in enumerate(zip(distances, indices)):
        src_acc = account_ids[i]
        
        for dist, j in zip(d_row[1:], idx_row[1:]):
            dst_acc = account_ids[j]
            
            edge_key = tuple(sorted([src_acc, dst_acc]))
            if edge_key in seen_pairs:
                continue
            seen_pairs.add(edge_key)
            
            if dist <= max_distance:
                similarity = float(np.exp(-dist))
                edges.append({
                    "src": src_acc,
                    "dst": dst_acc,
                    "distance": float(dist),
                    "similarity": similarity
                })
    
    return pd.DataFrame(edges)


def build_graph(df, edges_df):
    G = nx.Graph()
    G.add_nodes_from(df["account_id"].values)
    
    if len(edges_df) > 0:
        for _, row in edges_df.iterrows():
            G.add_edge(row["src"], row["dst"], 
                       distance=row["distance"], 
                       similarity=row["similarity"])
    
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


def create_network_visualization(G, df, community_map, selected_account=None, color_by="ecosystem_role"):
    if len(G.nodes()) == 0:
        return go.Figure()
    
    pos = nx.spring_layout(G, seed=42, k=2)
    
    edge_x = []
    edge_y = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="rgba(150,150,150,0.4)"),
        hoverinfo="none",
        mode="lines"
    )
    
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
        fig = go.Figure(data=[edge_trace, node_trace, highlight_trace])
    else:
        fig = go.Figure(data=[edge_trace, node_trace])
    
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
    
    neighbor_data = []
    for neighbor in neighbors:
        edge_data = G.get_edge_data(account_id, neighbor)
        account_info = df[df["account_id"] == neighbor].iloc[0].to_dict()
        account_info["similarity"] = edge_data.get("similarity", 0)
        account_info["distance"] = edge_data.get("distance", 0)
        neighbor_data.append(account_info)
    
    return pd.DataFrame(neighbor_data).sort_values("similarity", ascending=False)


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
                        st.session_state.uploaded_data = result
                        st.success(f"Loaded {len(result)} accounts")
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
        st.markdown("#### Clustering")
        clustering_method = st.selectbox("Algorithm", ["Louvain", "DBSCAN"])
        
        if clustering_method == "DBSCAN":
            dbscan_eps = st.slider("DBSCAN eps", min_value=0.3, max_value=2.0, value=0.6, step=0.1)
            dbscan_min_samples = st.slider("Min samples", min_value=2, max_value=8, value=3)
        
        st.markdown("---")
        st.markdown("#### About")
        st.caption("This dashboard identifies behavioral relationships between accounts in the poultry supply chain ecosystem using K-Nearest Neighbors algorithm.")
    
    if data_source == "Upload CSV/Excel" and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data.copy()
    else:
        df = generate_poultry_ecosystem_data(n_accounts=n_accounts if data_source == "Poultry Ecosystem Demo" else 80)
    
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
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Ecosystem Overview", "Account Details", "Anchor Analysis", 
        "RM Opportunities", "Cluster Analysis", "Data Export"
    ])
    
    with tab1:
        col1, col2, col3, col4, col5 = st.columns(5)
        
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
            
            fig = create_network_visualization(G, df, community_map, color_by=color_by)
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
                display_df = neighbors_df[[
                    "account_id", "legal_name", "ecosystem_role", "anchor_group",
                    "bri_status", "similarity", "lead_score_bri"
                ]].copy()
                display_df.columns = ["Account", "Legal Name", "Role", "Anchor", "BRI Status", "Similarity", "Lead Score"]
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
            fig_selected = create_network_visualization(G, df, community_map, selected_account, color_by="ecosystem_role")
            st.plotly_chart(fig_selected, use_container_width=True)
    
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


if __name__ == "__main__":
    main()
