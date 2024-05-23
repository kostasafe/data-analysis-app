import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def show_PCA_Visualization():
    
    #select all columns except the target column
    features = st.session_state.new_dataset.drop(columns=[st.session_state.target_column.name]).columns

    #Standardization
    x = st.session_state.new_dataset[features].values

    y = st.session_state.new_dataset[st.session_state.target_column.name]

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, y.reset_index(drop=True)], axis=1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot (1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('Two Component PCA', fontsize = 20)

    #get the unique target values and colors
    unique_targets = y.unique()
    colors = ['r', 'g', 'b'][:len(unique_targets)]
    #ploting for each target with a different color
    for target, color in zip(unique_targets,colors):
        indicesToKeep = finalDf[st.session_state.target_column.name] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   ,s = 50,
                   label=target)
    ax.legend()
    ax.grid()

    st.pyplot(fig)

def show_TSNE_Visualization():

    TSNE_data = st.session_state.new_dataset
    features = TSNE_data.drop(columns=[st.session_state.target_column.name]).columns
    target_column = st.session_state.target_column.name

    data_norm = TSNE_data.copy()
    sc = StandardScaler()
    data_norm[features] = sc.fit_transform(TSNE_data[features])

    tsne = TSNE(learning_rate= 500, n_components= 2)

    x_tsne = tsne.fit_transform(data_norm[features])
    y_tsne = TSNE_data[target_column]

    # Convert target values to numerical categories for color coding
    unique_targets = y_tsne.unique()
    target_to_num = {target: num for num, target in enumerate(unique_targets)}
    y_num = y_tsne.map(target_to_num)

    plt.figure(figsize = (16,11))
    for target, color in zip(unique_targets, ['r', 'g', 'b']):
        indices_to_keep = y_num == target_to_num[target]
        plt.scatter(x_tsne[indices_to_keep, 0], x_tsne[indices_to_keep, 1], label=target, color=color, marker='*')

    plt.xlabel("Dim 1", fontsize = 15)
    plt.ylabel("Dim 2", fontsize = 15)
    plt.title("T-SNE", fontsize = 20)
    plt.legend()
    plt.grid()
    st.pyplot(plt)

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
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="title">2D Visualizations</div>', unsafe_allow_html=True)
    if st.session_state.uploaded_file is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('<div class="header">Principal Component Analysis (PCA)</div>', unsafe_allow_html=True) 
            st.divider()
            show_PCA_Visualization()
        
        with right_column:
            st.markdown('<div class="header-tsne">T-Distributed Neighbor Embedding (t-SNE)</div>', unsafe_allow_html=True)
            st.divider()
            show_TSNE_Visualization()
        
        EDA_df = st.session_state.new_dataset
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        numeric_columns = EDA_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if not numeric_columns:
            st.warning("No numeric columns found in the uploaded dataset.")
            return
        selected_feature = st.selectbox("Select a feature for analysis", numeric_columns)
        

        col1, col2, col3 = st.columns(3)
        with col1:
            generate_button1 = st.button('Histogram')
            if generate_button1:
                st.markdown('<div class="header">Histogram of Selected Feature</div>', unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(EDA_df[selected_feature], bins=10, kde=True, ax=ax)
                ax.set_title(f'Histogram of {selected_feature}')
                ax.set_xlabel(selected_feature)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        with col2:
            generate_button2 = st.button('Box Plot')
            if generate_button2:
                st.markdown('<div class="header">Box Plot of Selected Feature</div>', unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=EDA_df[selected_feature], ax=ax)
                ax.set_title(f'Box Plot of {selected_feature}')
                ax.set_xlabel(selected_feature)
                st.pyplot(fig)
        with col3:
            generate_button3 = st.button('Scatter Plot')
            if generate_button3:
                if len(numeric_columns) > 1:
                    st.markdown('<div class="header">Scatter Plot of Selected Features</div>', unsafe_allow_html=True)
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    second_feature = st.selectbox("Select another feature for scatter plot", [col for col in numeric_columns if col != selected_feature])
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=selected_feature, y=second_feature, data=EDA_df, ax=ax)
                    ax.set_title(f'Scatter Plot of {selected_feature} vs {second_feature}')
                    ax.set_xlabel(selected_feature)
                    ax.set_ylabel(second_feature)
                    st.pyplot(fig)
                else:
                    st.warning("Not enough numeric features for a scatter plot.")
    else:
        st.warning("Please upload a CSV or an Excel file in Home Page to proceed.")