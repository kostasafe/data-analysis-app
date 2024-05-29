import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score, rand_score
from matplotlib.colors import ListedColormap
from pages.TwoD_Visualization import convert_to_norm_pca

# Function to display the DBSCAN clustering plot
def show_dbscan_clustering(eps, min_samples):
    # Get the dataset from the session
    dataset = st.session_state.numeric_dataset_with_no_label

    pca_df = convert_to_norm_pca(dataset)
    true_labels = st.session_state.target_column
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pca_df)

    # Create a colormap
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    colormap = ListedColormap(colors)

    # Plot the clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in unique_labels:
        color = [0, 0, 0, 1] if label == -1 else colormap(label)
        class_member_mask = (labels == label)
        xy = pca_df[class_member_mask]
        ax.scatter(xy.iloc[:, 0], xy.iloc[:, 1], c=[color], label=f'Cluster {label}' if label != -1 else 'Noise')
    
    ax.set_title("DBSCAN Clustering")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(unique_labels), vmax=max(unique_labels)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Cluster Label")

    # Display the plot
    st.pyplot(fig)

    if len(unique_labels) > 1:
        silhouette_avg = silhouette_score(pca_df, labels)
        calinski_harabasz_avg = calinski_harabasz_score(pca_df, labels)
        st.write(f"Silhouette Score: {silhouette_avg:.3f} | Calinski-Harabasz Score: {calinski_harabasz_avg:.3f}")
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, labels)
            ami = adjusted_mutual_info_score(true_labels, labels)
            ri = rand_score(true_labels, labels)
            st.write(f"Adjusted Rand Index (ARI): {ari:.3f}")
            st.write(f"Adjusted Mutual Information (AMI): {ami:.3f}")
            st.write(f"Rand Index (RI): {ri:.3f}")
    else:
        st.write("Silhouette Score: Not applicable (only one cluster found) | Calinski-Harabasz Score: Not applicable (only one cluster found)")

# Function to display the KMeans clustering plot
def show_kmeans_clustering(n_clusters):
    # Get the dataset from the session
    dataset = st.session_state.numeric_dataset_with_no_label

    pca_df = convert_to_norm_pca(dataset)
    true_labels = st.session_state.target_column
    
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)  # Adjust the number of clusters as needed
    labels = kmeans.fit_predict(pca_df)

    # Create a colormap
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    colormap = ListedColormap(colors)

    # Plot the clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in unique_labels:
        color = colormap(label)
        class_member_mask = (labels == label)
        xy = pca_df[class_member_mask]
        ax.scatter(xy.iloc[:, 0], xy.iloc[:, 1], c=[color], label=f'Cluster {label}')
    
    ax.set_title("KMeans Clustering")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(unique_labels), vmax=max(unique_labels)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Cluster Label")

    # Display the plot
    st.pyplot(fig)

    silhouette_avg = silhouette_score(pca_df, labels)
    calinski_harabasz_avg = calinski_harabasz_score(pca_df, labels)
    st.write(f"Silhouette Score: {silhouette_avg:.3f} | Calinski-Harabasz Score: {calinski_harabasz_avg:.3f}")
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)
        ami = adjusted_mutual_info_score(true_labels, labels)
        ri = rand_score(true_labels, labels)
        st.write(f"Adjusted Rand Index (ARI): {ari:.3f}")
        st.write(f"Adjusted Mutual Information (AMI): {ami:.3f}")
        st.write(f"Rand Index (RI): {ri:.3f}")
    
def show_Clustering_Algorithms():
    st.markdown(
        """
        <style>
        .title {
            font-size: 70px;
            text-align: center;
            margin-bottom: 0px;
            margin-top: 0px;
        }
        .header {
            font-size: 40px;
            color: FireBrick;
            text-align: center;
        }
        .header-two {
            font-size: 40px;
            color: violet;
            text-align: center;
        }
        .divider {
            margin: 30px 0;
            border-bottom: 2px solid #ddd;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="title">Clustering Algorithms</div>', unsafe_allow_html=True)
    if st.session_state.uploaded_file is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('<div class="header">DBSCAN clustering</div>', unsafe_allow_html=True) 
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            sub_l_c, sub_r_c = st.columns(2)
            with sub_l_c:
                eps = st.slider('Select eps value for DBSCAN', 0.1, 10.0, 0.5, 0.1)
            with sub_r_c:
                min_samples = st.slider('Select min_samples for DBSCAN', 1, 20, 5)
            show_dbscan_clustering(eps, min_samples)
        
        with right_column:
            st.markdown('<div class="header-two">KMeans clustering</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            n_clusters = st.slider('Select number of clusters for KMeans', 2, 10, 3)
            show_kmeans_clustering(n_clusters)
    else:
        st.warning("Please upload a CSV or an Excel file on the Home Page to proceed.")
