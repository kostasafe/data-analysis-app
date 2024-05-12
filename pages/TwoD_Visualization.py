import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def show_TwoD_Visualization():
    
    numeric_df = st.session_state.new_dataset.select_dtypes(include=['number']) #include only numeric columns
                
    imputer = SimpleImputer(strategy='mean') #we need to exclude also the empty cells
    numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns = numeric_df.columns)
    
    
    
    #we need to standardize the data
    x = numeric_df_imputed.loc[:,numeric_df_imputed.columns].values

    y = numeric_df_imputed.loc[:, [st.session_state.target_column.name]].values

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf,numeric_df_imputed[[st.session_state.target_column.name]]], axis = 1)

    script_dir = os.path.dirname(os.path.abspath(__file__)) #fixing a path error
    save_path = os.path.join(script_dir, '..', 'img', 'PCA.png') #save_path is used to create a .png file of the plot we generated so we can project it with streanlit

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot (1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = st.session_state.target_column.unique().tolist()
    targets_str = ", ".join(str(target) for target in targets)
    
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[st.session_state.target_column.name] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   ,s = 50,
                   label=target)
    ax.legend()
    ax.grid()

    fig.savefig(save_path)

    st.image(save_path, caption='PCA 2D Projection', use_column_width=True)
