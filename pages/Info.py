import streamlit as st

def show_Info(): 
        st.header("Step-by-Step Guide")
        st.subheader("Step 1: Upload Your CSV File")
        st.write("""
        - Your CSV file should contain a dataset with 's' rows and 'n' columns.
        - Each column represents a feature.
        - Ensure the last column is labeled as 'label' and contains the target values for classification.
        """)
        st.write("To upload your file, use the **'Upload'** button in the home page.")
        
        st.subheader("Step 2: Perform Clustering")
        st.write("""
        - Navigate to the **'Clustering'** tab.
        - Here, you can change the parameters of the dbscan and kmeans clustering algorithms to apply to your data.
        - PCA visualizations and metrics will be generated to help you understand the clusters formed.
        - The metrics will be saved for use in the comparison tab.
        """)
        
        st.subheader("Step 3: Perform Classification")
        st.write("""
        - Next, go to the **'Classification'** tab.
        - Here, you can change the parameters of the knn and random forest classification algorithms to apply to your data.
        - Confusion matrices, PCA visualizations and metrics will be generated to help you understand the classification results.
        - The metrics will be saved for use in the comparison tab.
        """)
        
        st.subheader("Step 4: Compare Algorithms")
        st.write("""
        - Finally, switch to the **'Comparison'** tab.
        - Here, you can compare the performance of different clustering and classification algorithms side by side.
        - Visual and statistical comparisons will be provided to help you choose the best algorithm for your data.
        """)
        
        st.write("Feel free to explore the application and experiment with different parameters to see how they perform on your dataset.")
        st.write("If you have any questions or need further assistance, please refer to the documentation or contact the authors.")
        
        st.subheader("About")
        st.write("This application was made according to specifications provided by professor [Aristidis Vrahatis](https://di.ionio.gr/en/department/staff/737-vrahatis/) as part of the course [Software Engineering](https://di.ionio.gr/en/studies/undergraduate-studies/courses/614/).")
        st.write("These instructions are available on the course's [opencourses.ionio.gr page](https://opencourses.ionio.gr/modules/document/?course=DDI259).")
        st.markdown("#### Authors")
        st.markdown(
                        """ 
                         [devpersi](https://github.com/devpersi): 2D Visualization, Clustering Algorithms, Comparison, Dockerization, Information 
                         [kostasafe](https://github.com/kostasafe): Project Setup, 2D Visualization, Classification Algorithms, Readme
                         [p15kalo](https://github.com/p15kalo): Clustering Algorithms, Classification Algorithms, Information, Readme
                        """
        )
        st.write("The accompanying report was mainly authored by kostasafe. Contact the team to get access.")
        st.write("The UML diagram contained in the report was created by p15kalo.")