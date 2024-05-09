import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def show_TwoD_Visualization():
        st.header("PCA 2D Visualization!")
        if st.session_state.uploaded_file:
                df = st.session_state.new_dataset
                numeric_df = df.select_dtypes(include=['number']) #include only numeric columns
                
                #we need to exclude also the empty cells
                imputer = SimpleImputer(strategy='mean')
                numeric_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns = numeric_df.columns)
                
                
                pca = PCA()
                pca.fit(numeric_df_imputed)
                
                st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
                plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
                plt.xlabel('Principal Component')
                plt.ylabel('Explained Variance Ratio')
                st.pyplot(plt)

                df_pca = pca.transform(numeric_df_imputed)

                st.write("Transformed Data:")
                st.write(pd.DataFrame(df_pca, columns=['PC{}'.format(i) for i in range(1, pca.n_components_+1)]))

