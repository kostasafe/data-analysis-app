import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def convert_to_norm_pca(df):
    # Normalize data
    df = StandardScaler().fit_transform(df)
    # PCA
    pca_df = pd.DataFrame(data=PCA(n_components=2).fit_transform(df), columns=['PCA1', 'PCA2'])
    return pca_df

def convert_to_norm_tsne(df):
    # Normalize data
    df = StandardScaler().fit_transform(df)
    # TSNE
    tsne_df = pd.DataFrame(data=TSNE(learning_rate=500, n_components=2).fit_transform(df), columns=['Dim 1', 'Dim 2'])
    return tsne_df

def show_TwoD_Visualization():
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
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="title">2D Visualizations</div>', unsafe_allow_html=True)
    
    
    dataset = pd.DataFrame()
    labels = pd.Series()
    # Get the dataset from the session
    if 'numeric_dataset_with_no_label' in st.session_state:
        dataset = st.session_state.numeric_dataset_with_no_label
    if 'target_column' in st.session_state:
        labels = st.session_state.target_column
    
    if not dataset.empty and not labels.empty:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        left_column, right_column = st.columns(2)
        
        # Display PCA
        with left_column:
            st.markdown('<div class="header">Principal Component Analysis (PCA)</div>', unsafe_allow_html=True) 
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            pca_df = convert_to_norm_pca(dataset)
            unique_labels = labels.unique()
            colors = ['r', 'g', 'b'][:len(unique_labels)]
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            for lbl, color in zip(unique_labels, colors):
                ax.scatter(pca_df[(labels == lbl)].iloc[:, 0],
                           pca_df[(labels == lbl)].iloc[:, 1],
                           c=color,
                           s=50,
                           label=lbl)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_title('Two Component PCA', fontsize=20)
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        # Display t-SNE
        with right_column:
            st.markdown('<div class="header-tsne">T-Distributed Neighbor Embedding (t-SNE)</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            tsne_df = convert_to_norm_tsne(dataset)
            unique_labels = labels.unique()
            colors = ['r', 'g', 'b'][:len(unique_labels)]
            fig = plt.figure(figsize=(16, 11))
            ax = fig.add_subplot(1, 1, 1)
            for lbl, color in zip(unique_labels, colors):
                ax.scatter(tsne_df[(labels == lbl)].iloc[:, 0],
                            tsne_df[(labels == lbl)].iloc[:, 1],
                            label=lbl,
                            color=color,
                            marker='*')
            ax.set_xlabel("Dim 1", fontsize=15)
            ax.set_ylabel("Dim 2", fontsize=15)
            ax.set_title("T-SNE", fontsize=20)
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        # EDA plots
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="header">Exploratory Data Analysis (EDA) Plots</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="header">PCA Histogram</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(pca_df.iloc[:, 0], bins=10, kde=True, ax=ax)
            ax.set_title("Histogram of Principal Component 1")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with col2:
            st.markdown('<div class="header">PCA Box Plot</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=pca_df.iloc[:, 0], ax=ax)
            ax.set_title("Box Plot of Principal Component 1")
            ax.set_xlabel("Principal Component 1")
            st.pyplot(fig)

        with col3:
            st.markdown('<div class="header">PCA Scatter Plot</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=pca_df.columns[0], y=pca_df.columns[1], data=pca_df, ax=ax)
            ax.set_title('Scatter Plot of Principal Component 1 vs Principal Component 2')
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            st.pyplot(fig)

        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown('<div class="header">t-SNE Histogram</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(tsne_df.iloc[:, 0], bins=10, kde=True, ax=ax)
            ax.set_title(f'Histogram of Dim 1')
            ax.set_xlabel("Dim 1")
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        with col5:
            st.markdown('<div class="header">t-SNE Box Plot</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=tsne_df.iloc[:, 0], ax=ax)
            ax.set_title('Box Plot of Dim 1')
            ax.set_xlabel("Dim 1")
            st.pyplot(fig)

        with col6:
            st.markdown('<div class="header">t-SNE Scatter Plot</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=tsne_df.columns[0], y=tsne_df.columns[1], data=tsne_df, ax=ax)
            ax.set_title(f'Scatter Plot of Dim 1 vs Dim 2')
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            st.pyplot(fig)
    else:
        st.warning("Please upload a CSV or an Excel file in Home Page to proceed.")