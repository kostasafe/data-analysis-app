import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def show_PCA_Visualization():
    # Select all columns except the target column
    features = st.session_state.new_dataset.drop(columns=[st.session_state.target_column.name]).columns

    # Standardization
    x = st.session_state.new_dataset[features].values
    y = st.session_state.new_dataset[st.session_state.target_column.name]
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, y.reset_index(drop=True)], axis=1)

    return finalDf, y

def show_TSNE_Visualization():
    TSNE_data = st.session_state.new_dataset
    features = TSNE_data.drop(columns=[st.session_state.target_column.name]).columns
    target_column = st.session_state.target_column.name

    data_norm = TSNE_data.copy()
    sc = StandardScaler()
    data_norm[features] = sc.fit_transform(TSNE_data[features])

    tsne = TSNE(learning_rate=500, n_components=2)
    x_tsne = tsne.fit_transform(data_norm[features])
    y_tsne = TSNE_data[target_column]

    tsneDf = pd.DataFrame(data=x_tsne, columns=['Dim 1', 'Dim 2'])
    finalDf = pd.concat([tsneDf, y_tsne.reset_index(drop=True)], axis=1)

    return finalDf, y_tsne

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
    if st.session_state.uploaded_file is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Display PCA
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('<div class="header">Principal Component Analysis (PCA)</div>', unsafe_allow_html=True) 
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            pca_df, pca_targets = show_PCA_Visualization()
            unique_targets = pca_targets.unique()
            colors = ['r', 'g', 'b'][:len(unique_targets)]
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            ax.set_title('Two Component PCA', fontsize=20)
            for target, color in zip(unique_targets, colors):
                indicesToKeep = pca_df[st.session_state.target_column.name] == target
                ax.scatter(pca_df.loc[indicesToKeep, 'principal component 1'],
                           pca_df.loc[indicesToKeep, 'principal component 2'],
                           c=color,
                           s=50,
                           label=target)
            ax.legend()
            ax.grid()
            st.pyplot(fig)

        # Display t-SNE
        with right_column:
            st.markdown('<div class="header-tsne">T-Distributed Neighbor Embedding (t-SNE)</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            tsne_df, tsne_targets = show_TSNE_Visualization()
            unique_targets = tsne_targets.unique()
            colors = ['r', 'g', 'b'][:len(unique_targets)]
            fig = plt.figure(figsize=(16, 11))
            for target, color in zip(unique_targets, colors):
                indices_to_keep = tsne_df[st.session_state.target_column.name] == target
                plt.scatter(tsne_df.loc[indices_to_keep, 'Dim 1'],
                            tsne_df.loc[indices_to_keep, 'Dim 2'],
                            label=target,
                            color=color,
                            marker='*')
            plt.xlabel("Dim 1", fontsize=15)
            plt.ylabel("Dim 2", fontsize=15)
            plt.title("T-SNE", fontsize=20)
            plt.legend()
            plt.grid()
            st.pyplot(plt)

        # EDA plots
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="header">Exploratory Data Analysis (EDA) Plots</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        eda_df = st.session_state.new_dataset
        pca_selected_feature1 = 'principal component 1'
        pca_selected_feature2 = 'principal component 2'
        tsne_selected_feature1 = 'Dim 1'
        tsne_selected_feature2 = 'Dim 2'

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="header">PCA Histogram</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(pca_df[pca_selected_feature1], bins=10, kde=True, ax=ax)
            ax.set_title(f'Histogram of {pca_selected_feature1}')
            ax.set_xlabel(pca_selected_feature1)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        with col2:
            st.markdown('<div class="header">PCA Box Plot</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=pca_df[pca_selected_feature1], ax=ax)
            ax.set_title(f'Box Plot of {pca_selected_feature1}')
            ax.set_xlabel(pca_selected_feature1)
            st.pyplot(fig)

        with col3:
            st.markdown('<div class="header">PCA Scatter Plot</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=pca_selected_feature1, y=pca_selected_feature2, data=pca_df, ax=ax)
            ax.set_title(f'Scatter Plot of {pca_selected_feature1} vs {pca_selected_feature2}')
            ax.set_xlabel(pca_selected_feature1)
            ax.set_ylabel(pca_selected_feature2)
            st.pyplot(fig)

        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown('<div class="header">t-SNE Histogram</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(tsne_df[tsne_selected_feature1], bins=10, kde=True, ax=ax)
            ax.set_title(f'Histogram of {tsne_selected_feature1}')
            ax.set_xlabel(tsne_selected_feature1)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        with col5:
            st.markdown('<div class="header">t-SNE Box Plot</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=tsne_df[tsne_selected_feature1], ax=ax)
            ax.set_title(f'Box Plot of {tsne_selected_feature1}')
            ax.set_xlabel(tsne_selected_feature1)
            st.pyplot(fig)

        with col6:
            st.markdown('<div class="header">t-SNE Scatter Plot</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=tsne_selected_feature1, y=tsne_selected_feature2, data=tsne_df, ax=ax)
            ax.set_title(f'Scatter Plot of {tsne_selected_feature1} vs {tsne_selected_feature2}')
            ax.set_xlabel(tsne_selected_feature1)
            ax.set_ylabel(tsne_selected_feature2)
            st.pyplot(fig)
    else:
        st.warning("Please upload a CSV or an Excel file in Home Page to proceed.")