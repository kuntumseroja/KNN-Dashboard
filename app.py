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
    page_title="KNN Banking Account Relationships",
    page_icon="🏦",
    layout="wide"
)


NUMERIC_FEATURES = [
    "avg_txn_amount_30d",
    "txn_count_30d",
    "salary_inflow_90d",
    "cash_withdrawal_ratio_90d",
    "merchant_diversity_90d",
    "unique_devices_90d",
    "unique_ips_90d",
    "country_risk_score",
    "channel_mix_online_ratio",
]

CATEGORICAL_FEATURES = ["segment_code"]


@st.cache_data
def generate_mock_data(n_accounts=100, seed=42):
    np.random.seed(seed)
    
    segments = ["Retail", "SME", "Corporate"]
    segment_weights = [0.6, 0.3, 0.1]
    
    data = {
        "account_id": [f"ACC{str(i).zfill(5)}" for i in range(1, n_accounts + 1)],
        "avg_txn_amount_30d": np.random.lognormal(mean=6, sigma=1.5, size=n_accounts).clip(10, 50000),
        "txn_count_30d": np.random.poisson(lam=25, size=n_accounts).clip(1, 200),
        "salary_inflow_90d": np.random.lognormal(mean=8, sigma=1.2, size=n_accounts).clip(0, 100000),
        "cash_withdrawal_ratio_90d": np.random.beta(a=2, b=5, size=n_accounts),
        "merchant_diversity_90d": np.random.randint(1, 50, size=n_accounts),
        "unique_devices_90d": np.random.poisson(lam=2, size=n_accounts).clip(1, 10),
        "unique_ips_90d": np.random.poisson(lam=5, size=n_accounts).clip(1, 30),
        "country_risk_score": np.random.uniform(0, 1, size=n_accounts),
        "segment_code": np.random.choice(segments, size=n_accounts, p=segment_weights),
        "channel_mix_online_ratio": np.random.beta(a=5, b=2, size=n_accounts),
    }
    
    for i in range(0, n_accounts, 10):
        cluster_size = min(np.random.randint(3, 6), n_accounts - i)
        base_values = {
            "avg_txn_amount_30d": np.random.lognormal(mean=6, sigma=0.5),
            "cash_withdrawal_ratio_90d": np.random.beta(a=2, b=5),
            "channel_mix_online_ratio": np.random.beta(a=5, b=2),
            "unique_devices_90d": np.random.poisson(lam=2) + 1,
        }
        for j in range(cluster_size):
            if i + j < n_accounts:
                data["avg_txn_amount_30d"][i + j] = base_values["avg_txn_amount_30d"] * np.random.uniform(0.8, 1.2)
                data["cash_withdrawal_ratio_90d"][i + j] = base_values["cash_withdrawal_ratio_90d"] * np.random.uniform(0.8, 1.2)
                data["channel_mix_online_ratio"][i + j] = min(base_values["channel_mix_online_ratio"] * np.random.uniform(0.8, 1.2), 1.0)
                data["unique_devices_90d"][i + j] = base_values["unique_devices_90d"]
    
    return pd.DataFrame(data)


def validate_uploaded_data(df):
    required_cols = ["account_id"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        null_count = df[col].isna().sum()
        if null_count > 0:
            return False, f"Column '{col}' has {null_count} invalid/missing values that could not be converted to numbers"
    
    if df["account_id"].isna().any():
        return False, "Account ID column contains empty values"
    
    if df["account_id"].duplicated().any():
        return False, "Duplicate account IDs found"
    
    if len(df) < 3:
        return False, "Dataset must contain at least 3 accounts"
    
    return True, df


def build_knn_pipeline(df, n_neighbors=6):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
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
    
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    pipeline.fit(X)
    
    return pipeline


def find_neighbors_and_edges(df, pipeline, max_distance=2.0):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
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


def detect_communities_dbscan(df, eps=0.5, min_samples=3):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_processed = preprocessor.fit_transform(X)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_processed)
    
    community_map = {}
    for i, account_id in enumerate(df["account_id"].values):
        community_map[account_id] = int(labels[i])
    
    return community_map


def calculate_fraud_risk_score(G, df, community_map):
    if len(G.nodes()) == 0:
        return {}
    
    degree_centrality = nx.degree_centrality(G) if len(G.edges()) > 0 else {n: 0 for n in G.nodes()}
    betweenness_centrality = nx.betweenness_centrality(G) if len(G.edges()) > 0 else {n: 0 for n in G.nodes()}
    
    community_sizes = {}
    for node, comm in community_map.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    risk_scores = {}
    
    for account_id in df["account_id"].values:
        account_data = df[df["account_id"] == account_id].iloc[0]
        
        hub_score = degree_centrality.get(account_id, 0) * 25
        bridge_score = betweenness_centrality.get(account_id, 0) * 25
        
        community = community_map.get(account_id, 0)
        community_size = community_sizes.get(community, 1)
        cluster_score = min(community_size / 10, 1) * 15
        
        behavioral_score = 0
        behavioral_score += min(account_data["cash_withdrawal_ratio_90d"] * 10, 10)
        behavioral_score += min(account_data["country_risk_score"] * 10, 10)
        behavioral_score += min(account_data["unique_devices_90d"] / 5, 1) * 5
        behavioral_score += min(account_data["unique_ips_90d"] / 15, 1) * 5
        
        total_risk = hub_score + bridge_score + cluster_score + behavioral_score
        risk_scores[account_id] = {
            "total_risk": min(total_risk, 100),
            "hub_score": hub_score,
            "bridge_score": bridge_score,
            "cluster_score": cluster_score,
            "behavioral_score": behavioral_score,
            "risk_level": "High" if total_risk >= 60 else "Medium" if total_risk >= 30 else "Low"
        }
    
    return risk_scores


def create_network_visualization(G, df, community_map, selected_account=None, risk_scores=None):
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
        line=dict(width=1, color="rgba(150,150,150,0.5)"),
        hoverinfo="none",
        mode="lines"
    )
    
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    centrality = nx.degree_centrality(G) if len(G.edges()) > 0 else {n: 0 for n in G.nodes()}
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if risk_scores:
            risk = risk_scores.get(node, {}).get("total_risk", 0)
            node_colors.append(risk)
        else:
            community_id = community_map.get(node, 0)
            node_colors.append(community_id)
        
        degree = G.degree(node)
        node_sizes.append(10 + degree * 3)
        
        account_data = df[df["account_id"] == node].iloc[0] if len(df[df["account_id"] == node]) > 0 else None
        if account_data is not None:
            text = f"<b>{node}</b><br>"
            text += f"Segment: {account_data['segment_code']}<br>"
            text += f"Avg Txn: ${account_data['avg_txn_amount_30d']:.2f}<br>"
            text += f"Connections: {degree}<br>"
            text += f"Centrality: {centrality.get(node, 0):.3f}<br>"
            text += f"Community: {community_map.get(node, 0)}"
            if risk_scores:
                risk_info = risk_scores.get(node, {})
                text += f"<br>Risk Score: {risk_info.get('total_risk', 0):.1f}"
                text += f"<br>Risk Level: {risk_info.get('risk_level', 'Unknown')}"
        else:
            text = node
        node_text.append(text)
    
    colorbar_title = "Risk Score" if risk_scores else "Community"
    colorscale = "RdYlGn_r" if risk_scores else "Viridis"
    
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
                size=node_sizes[selected_idx] + 10,
                color="red",
                line=dict(width=3, color="darkred")
            ),
            showlegend=False
        )
        fig = go.Figure(data=[edge_trace, node_trace, highlight_trace])
    else:
        fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Account Relationship Network",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
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


def main():
    st.title("KNN Banking Account Relationship Dashboard")
    st.markdown("Discover behavioral relationships between accounts using K-Nearest Neighbors algorithm")
    
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Data Source")
        data_source = st.radio("Select data source:", ["Generate Mock Data", "Upload CSV/Excel"])
        
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
            
            with st.expander("Required columns"):
                st.markdown("**Required columns:**")
                st.code("account_id, " + ", ".join(NUMERIC_FEATURES) + ", segment_code")
                st.markdown("**Download sample template:**")
                sample_df = generate_mock_data(n_accounts=10)
                st.download_button(
                    "Download Template",
                    export_to_csv(sample_df),
                    "account_template.csv",
                    "text/csv"
                )
        
        st.markdown("---")
        
        if data_source == "Generate Mock Data":
            n_accounts = st.slider("Number of Accounts", min_value=50, max_value=500, value=100, step=50)
        
        n_neighbors = st.slider("K Neighbors", min_value=3, max_value=15, value=6)
        max_distance = st.slider("Max Distance Threshold", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        
        st.markdown("---")
        st.subheader("Clustering Algorithm")
        clustering_method = st.selectbox("Method", ["Louvain", "DBSCAN"])
        
        if clustering_method == "DBSCAN":
            dbscan_eps = st.slider("DBSCAN eps", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            dbscan_min_samples = st.slider("DBSCAN min_samples", min_value=2, max_value=10, value=3)
        
        st.markdown("---")
        st.markdown("### Feature Weights")
        st.caption("Features used for similarity calculation:")
        for feat in NUMERIC_FEATURES[:5]:
            st.caption(f"• {feat.replace('_', ' ').title()}")
        st.caption("... and more")
    
    if data_source == "Upload CSV/Excel" and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data.copy()
    else:
        df = generate_mock_data(n_accounts=n_accounts if data_source == "Generate Mock Data" else 100)
    
    pipeline = build_knn_pipeline(df, n_neighbors=n_neighbors)
    edges_df = find_neighbors_and_edges(df, pipeline, max_distance=max_distance)
    G = build_graph(df, edges_df)
    
    if clustering_method == "Louvain":
        community_map = detect_communities_louvain(G)
    else:
        community_map = detect_communities_dbscan(df, eps=dbscan_eps, min_samples=dbscan_min_samples)
    
    df["community"] = df["account_id"].map(community_map)
    
    risk_scores = calculate_fraud_risk_score(G, df, community_map)
    df["fraud_risk_score"] = df["account_id"].map(lambda x: risk_scores.get(x, {}).get("total_risk", 0))
    df["risk_level"] = df["account_id"].map(lambda x: risk_scores.get(x, {}).get("risk_level", "Unknown"))
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Network Overview", "Account Lookup", "Community Analysis", 
        "Fraud Risk Analysis", "Clustering Comparison", "Data Explorer"
    ])
    
    with tab1:
        st.subheader("Account Relationship Network")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Accounts", len(df))
        with col2:
            st.metric("Total Relationships", len(edges_df))
        with col3:
            st.metric("Communities Detected", len(set(community_map.values())))
        with col4:
            avg_connections = len(edges_df) * 2 / len(df) if len(df) > 0 else 0
            st.metric("Avg Connections", f"{avg_connections:.1f}")
        
        show_risk = st.checkbox("Color by Risk Score", value=False)
        fig = create_network_visualization(G, df, community_map, risk_scores=risk_scores if show_risk else None)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Network Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            if len(G.edges()) > 0:
                centrality = nx.degree_centrality(G)
                top_hubs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                hub_df = pd.DataFrame(top_hubs, columns=["Account", "Centrality"])
                hub_df["Centrality"] = hub_df["Centrality"].round(4)
                st.markdown("**Top Hub Accounts** (potential mule accounts)")
                st.dataframe(hub_df, use_container_width=True, hide_index=True)
        
        with col2:
            if len(edges_df) > 0:
                st.markdown("**Similarity Score Distribution**")
                fig_hist = px.histogram(edges_df, x="similarity", nbins=30, 
                                       title="Distribution of Similarity Scores")
                st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        st.subheader("Account Lookup")
        
        selected_account = st.selectbox("Select an Account", options=df["account_id"].tolist())
        
        if selected_account:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Selected Account Details**")
                account_data = df[df["account_id"] == selected_account].iloc[0]
                
                st.markdown(f"**Account ID:** {selected_account}")
                st.markdown(f"**Segment:** {account_data['segment_code']}")
                st.markdown(f"**Community:** {community_map.get(selected_account, 'N/A')}")
                
                risk_info = risk_scores.get(selected_account, {})
                risk_level = risk_info.get("risk_level", "Unknown")
                risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
                st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
                st.markdown(f"**Risk Score:** {risk_info.get('total_risk', 0):.1f}/100")
                
                st.markdown("---")
                st.markdown("**Feature Values:**")
                for feat in NUMERIC_FEATURES:
                    value = account_data[feat]
                    if "ratio" in feat or feat == "country_risk_score":
                        st.markdown(f"• {feat.replace('_', ' ').title()}: {value:.2%}")
                    elif "amount" in feat or "inflow" in feat:
                        st.markdown(f"• {feat.replace('_', ' ').title()}: ${value:,.2f}")
                    else:
                        st.markdown(f"• {feat.replace('_', ' ').title()}: {value:.1f}")
            
            with col2:
                st.markdown("**Similar Accounts (Neighbors)**")
                neighbors_df = get_account_neighbors(G, selected_account, df)
                
                if len(neighbors_df) > 0:
                    display_cols = ["account_id", "segment_code", "similarity", "distance", 
                                   "avg_txn_amount_30d", "txn_count_30d"]
                    st.dataframe(neighbors_df[display_cols], use_container_width=True, hide_index=True)
                    
                    st.markdown("**Why are these accounts similar?**")
                    if len(neighbors_df) > 0:
                        top_neighbor = neighbors_df.iloc[0]
                        explanation = f"""
                        Account **{selected_account}** is most similar to **{top_neighbor['account_id']}** 
                        (similarity score: {top_neighbor['similarity']:.3f}) because they share:
                        - Similar transaction patterns (avg amount: ${account_data['avg_txn_amount_30d']:.2f} vs ${top_neighbor['avg_txn_amount_30d']:.2f})
                        - Same segment: {account_data['segment_code']} vs {top_neighbor['segment_code']}
                        - Similar channel usage (online ratio: {account_data['channel_mix_online_ratio']:.1%} vs {top_neighbor['channel_mix_online_ratio']:.1%})
                        """
                        st.info(explanation)
                else:
                    st.warning("No similar accounts found within the distance threshold.")
            
            st.markdown("---")
            st.subheader("Account in Network Context")
            fig_selected = create_network_visualization(G, df, community_map, selected_account)
            st.plotly_chart(fig_selected, use_container_width=True)
    
    with tab3:
        st.subheader("Community Analysis")
        
        community_series = df["community"]
        community_counts = community_series.value_counts().reset_index()
        community_counts.columns = ["Community", "Account Count"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Community Size Distribution**")
            fig_comm = px.bar(community_counts, x="Community", y="Account Count",
                             title="Accounts per Community")
            st.plotly_chart(fig_comm, use_container_width=True)
        
        with col2:
            st.markdown("**Community Statistics**")
            st.dataframe(community_counts, use_container_width=True, hide_index=True)
        
        unique_communities = sorted(df["community"].unique())
        selected_community = st.selectbox("Explore Community", options=unique_communities)
        
        if selected_community is not None:
            community_accounts = df[df["community"] == selected_community]
            
            st.markdown(f"**Community {selected_community} - {len(community_accounts)} Accounts**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Segment Distribution**")
                segment_series = community_accounts["segment_code"]
                segment_counts = segment_series.value_counts()
                fig_seg = px.pie(values=segment_counts.values, names=segment_counts.index,
                                title=f"Segments in Community {selected_community}")
                st.plotly_chart(fig_seg, use_container_width=True)
            
            with col2:
                st.markdown("**Average Feature Values**")
                avg_features = community_accounts[NUMERIC_FEATURES].mean()
                feature_df = pd.DataFrame({
                    "Feature": [f.replace("_", " ").title() for f in NUMERIC_FEATURES],
                    "Average Value": list(avg_features.values)
                })
                st.dataframe(feature_df, use_container_width=True, hide_index=True)
            
            st.markdown("**Accounts in this Community**")
            display_cols = ["account_id", "segment_code", "fraud_risk_score", "risk_level"] + NUMERIC_FEATURES[:5]
            st.dataframe(community_accounts[display_cols], use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("Fraud Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        high_risk = df[df["risk_level"] == "High"]
        medium_risk = df[df["risk_level"] == "Medium"]
        low_risk = df[df["risk_level"] == "Low"]
        
        with col1:
            st.metric("High Risk Accounts", len(high_risk), delta=None)
        with col2:
            st.metric("Medium Risk Accounts", len(medium_risk), delta=None)
        with col3:
            st.metric("Low Risk Accounts", len(low_risk), delta=None)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Score Distribution**")
            fig_risk = px.histogram(df, x="fraud_risk_score", nbins=20,
                                   color="risk_level",
                                   color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"},
                                   title="Distribution of Risk Scores")
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            st.markdown("**Risk Level Breakdown**")
            risk_counts = df["risk_level"].value_counts()
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                            color=risk_counts.index,
                            color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"},
                            title="Risk Level Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("**High Risk Accounts**")
        if len(high_risk) > 0:
            display_cols = ["account_id", "segment_code", "fraud_risk_score", "community",
                           "avg_txn_amount_30d", "cash_withdrawal_ratio_90d", "country_risk_score"]
            st.dataframe(high_risk.sort_values("fraud_risk_score", ascending=False)[display_cols],
                        use_container_width=True, hide_index=True)
        else:
            st.info("No high-risk accounts detected with current thresholds.")
        
        st.markdown("---")
        st.markdown("**Risk Score Components Explained**")
        st.markdown("""
        The fraud risk score (0-100) is calculated based on:
        - **Hub Score (25%)**: How central the account is in the network (high connectivity)
        - **Bridge Score (25%)**: How often the account connects different communities
        - **Cluster Score (15%)**: Size of the account's community (larger clusters = higher risk)
        - **Behavioral Score (35%)**: Based on cash withdrawal ratio, country risk, device/IP diversity
        """)
    
    with tab5:
        st.subheader("Clustering Algorithm Comparison")
        
        louvain_map = detect_communities_louvain(G)
        dbscan_map = detect_communities_dbscan(df, eps=0.5, min_samples=3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Louvain Clustering**")
            louvain_clusters = len(set(louvain_map.values()))
            st.metric("Communities Found", louvain_clusters)
            
            louvain_sizes = pd.Series(louvain_map).value_counts().reset_index()
            louvain_sizes.columns = ["Community", "Size"]
            fig_louv = px.bar(louvain_sizes, x="Community", y="Size", title="Louvain Community Sizes")
            st.plotly_chart(fig_louv, use_container_width=True)
            
            st.caption("Louvain optimizes modularity on the graph structure")
        
        with col2:
            st.markdown("**DBSCAN Clustering**")
            dbscan_clusters = len(set(dbscan_map.values()))
            noise_points = sum(1 for v in dbscan_map.values() if v == -1)
            st.metric("Clusters Found", dbscan_clusters - (1 if -1 in dbscan_map.values() else 0))
            st.metric("Noise Points", noise_points)
            
            dbscan_sizes = pd.Series(dbscan_map).value_counts().reset_index()
            dbscan_sizes.columns = ["Cluster", "Size"]
            fig_db = px.bar(dbscan_sizes, x="Cluster", y="Size", title="DBSCAN Cluster Sizes")
            st.plotly_chart(fig_db, use_container_width=True)
            
            st.caption("DBSCAN clusters by density in feature space")
        
        st.markdown("---")
        st.markdown("**Algorithm Comparison**")
        st.markdown("""
        | Aspect | Louvain | DBSCAN |
        |--------|---------|--------|
        | **Basis** | Graph structure (edges) | Feature space density |
        | **Noise handling** | All nodes assigned | Identifies outliers (-1) |
        | **Best for** | Network communities | Behavioral clusters |
        | **Parameters** | None (automatic) | eps, min_samples |
        """)
    
    with tab6:
        st.subheader("Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Export Accounts CSV",
                export_to_csv(df),
                "accounts_data.csv",
                "text/csv"
            )
        with col2:
            if len(edges_df) > 0:
                st.download_button(
                    "Export Relationships CSV",
                    export_to_csv(edges_df),
                    "relationships.csv",
                    "text/csv"
                )
        with col3:
            risk_df = pd.DataFrame([
                {"account_id": k, **v} for k, v in risk_scores.items()
            ])
            st.download_button(
                "Export Risk Scores CSV",
                export_to_csv(risk_df),
                "risk_scores.csv",
                "text/csv"
            )
        
        st.markdown("---")
        st.markdown("**Account Features Dataset**")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("**Relationship Edges**")
        if len(edges_df) > 0:
            st.dataframe(edges_df.head(100), use_container_width=True, hide_index=True)
            st.caption(f"Showing first 100 of {len(edges_df)} relationships")
        else:
            st.warning("No relationships found. Try increasing the distance threshold.")
        
        st.markdown("---")
        st.markdown("**Feature Correlations**")
        corr_matrix = df[NUMERIC_FEATURES].corr()
        fig_corr = px.imshow(corr_matrix, 
                            labels=dict(color="Correlation"),
                            title="Feature Correlation Matrix",
                            color_continuous_scale="RdBu",
                            aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()
