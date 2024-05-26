import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Function to display the DBSCAN clustering plot
def show_dbscan_clustering():
    # Get the dataset from the session
    dataset = st.session_state.numeric_dataset_with_no_label

    # Normalize data
    data_norm = dataset.copy()
    sc = StandardScaler()
    data_norm = sc.fit_transform(dataset)
    
    # PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data_norm)
    pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(pca_df)

    # Create a colormap
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    colormap = ListedColormap(colors)

    # Plot the clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in unique_labels:
        if label == -1:
            # Noise points
            color = [0, 0, 0, 1]
        else:
            color = colormap(label)
        class_member_mask = (labels == label)
        xy = pca_df[class_member_mask]
        ax.scatter(xy.iloc[:, 0], xy.iloc[:, 1], c=[color], label=f'Cluster {label}' if label != -1 else 'Noise')
    
    ax.set_title("DBSCAN Clustering")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min(unique_labels), vmax=max(unique_labels)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Cluster Label")

    # Display the plot
    st.pyplot(fig)

# Main function to set up the Streamlit page
def show_First_Machine_Learning_Algorithm():
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
        .header-tsne {
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
    
    st.markdown('<div class="title">2D Visualizations</div>', unsafe_allow_html=True)
    if st.session_state.uploaded_file is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('<div class="header">DBSCAN clustering</div>', unsafe_allow_html=True) 
            st.divider()
            show_dbscan_clustering()
        
        with right_column:
            st.markdown('<div class="header-tsne">DBSCAN clustering</div>', unsafe_allow_html=True)
            st.divider()
            show_dbscan_clustering()

