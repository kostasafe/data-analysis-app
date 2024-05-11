import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np
from sklearn.impute import SimpleImputer

def show_TwoD_Visualization():
        st.header("PCA 2D Visualization!")
        if st.session_state.uploaded_file:
                df = st.session_state.new_dataset
                numeric_df = df.select_dtypes(include=['number']) #include only numeric columns
                
                #we need to exclude also the empty cells
                imputer = SimpleImputer(strategy='mean')
                numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns = numeric_df.columns)

                script_dir = os.path.dirname(os.path.abspath(__file__)) #fixing a path error
                save_path = os.path.join(script_dir, '..', 'img', 'biplot.png') #save_path is used to create a .png file of the plot we generated so we can project it with streanlit
                
                pca =PCA(n_components=2)
                pca.fit(numeric_df_imputed)
                biplot(numeric_df_imputed, pca, labels=numeric_df_imputed.columns, save_path=save_path) #calling function with our data
                
                st.image(save_path, caption="Biplot Visualization", use_column_width=True)#load our image

def biplot(data, pca, labels=None, figsize=(8,8), save_path = None): #need further customization
        #Project data onto the principal components
        data_projected = pca.transform(data)
        components = pca.components_

        st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        st.pyplot(plt)

        st.write("Transformed Data:")
        st.write(pd.DataFrame(data_projected, columns=['PC{}'.format(i) for i in range(1, pca.n_components_+1)]))

        #data point plot
        plt.figure(figsize=figsize)
        plt.scatter(data_projected[:,0],data_projected[:,1], alpha=0.5)

        #plotting varible vectors
        for i, (component1, component2) in enumerate(zip(components[0], components[1])):
                plt.arrow(0, 0, component1, component2, color='r', alpha=0.5)
                if labels is not None:
                        plt.text(component1, component2, labels[i], fontsize='12', ha='right')

        plt.xlabel('PC1') #add labels and grid
        plt.ylabel('PCA2')
        plt.grid()

        if save_path:
                plt.savefig(save_path)
        else:
                plt.show()