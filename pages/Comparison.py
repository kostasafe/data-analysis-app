import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Define a helper function to create subplots
def plot_metric(ax, metric, alg1_score, alg2_score, alg1_name, alg2_name):
    ax.bar([alg1_name, alg2_name], [alg1_score, alg2_score], color=['blue', 'orange'])
    ax.set_title(metric)
    ax.set_ylim(0, max(alg1_score, alg2_score) * 1.1)

def show_Comparison():
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
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="title">Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown('<div class="header">Clustering</div>', unsafe_allow_html=True)
        if "dbscan_scores" in st.session_state and "kmeans_scores" in st.session_state:
            dbscan_scores = st.session_state.dbscan_scores
            kmeans_scores = st.session_state.kmeans_scores
            data = {
            'Metric': ['Silhouette Score', 'Calinski-Harabasz Score', 'Adjusted Rand Index', 'Adjusted Mutual Info', 'Rand Index', 'Mutual Info'],
            'DBScan': dbscan_scores,
            'KMeans': kmeans_scores
            }
            
            # Convert the data into a DataFrame
            df = pd.DataFrame(data)

            st.subheader('Comparison Table')
            st.table(df)
            st.subheader('Metric Comparison')
            metrics = df['Metric']
            
            # Plotting the metrics comparison
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for i, metric in enumerate(metrics):
                row = i // 3
                col = i % 3
                plot_metric(axes[row, col], metric, df['DBScan'][i], df['KMeans'][i], 'DBScan', 'KMeans')

            st.pyplot(fig)
            
            # Detailed view for each metric
            st.subheader('Detailed Metrics View')

            metric_selected = st.selectbox('Select a metric to compare:', metrics)

            alg1_score = df.loc[df['Metric'] == metric_selected, 'DBScan'].values[0]
            alg2_score = df.loc[df['Metric'] == metric_selected, 'KMeans'].values[0]

            st.write(f"### {metric_selected}")
            st.write(f"**DBScan**: {alg1_score}")
            st.write(f"**KMeans**: {alg2_score}")

            if alg1_score > alg2_score:
                st.success(f"DBScan performs better in {metric_selected} with a score of {alg1_score}.")
            elif alg1_score < alg2_score:
                st.success(f"KMeans performs better in {metric_selected} with a score of {alg2_score}.")
            else:
                st.info(f"Both algorithms perform equally in {metric_selected} with a score of {alg1_score}.")

            st.write("### Summary")
            for i in range(len(metrics)):
                st.write(f"- **{metrics[i]}**: DBScan: {df['DBScan'][i]:.3f} | KMeans: {df['KMeans'][i]:.3f}")
        else:
            st.warning("No data available for comparison.")
    with right_column:
        st.markdown('<div class="header-two">Classification</div>', unsafe_allow_html=True)
        if "knn_scores" in st.session_state and "random_forest_scores" in st.session_state:
            knn_scores = st.session_state.knn_scores
            random_forest_scores = st.session_state.random_forest_scores
            data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Log Loss'],
            'KNN': knn_scores,
            'Random Forest': random_forest_scores
            }
            
            # Convert the data into a DataFrame
            df = pd.DataFrame(data)

            st.subheader('Comparison Table')
            st.table(df)
            st.subheader('Metric Comparison')
            metrics = df['Metric']
            
            # Plotting the metrics comparison
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for i, metric in enumerate(metrics):
                row = i // 3
                col = i % 3
                plot_metric(axes[row, col], metric, df['KNN'][i], df['Random Forest'][i], 'KNN', 'Random Forest')

            st.pyplot(fig)
            
            # Detailed view for each metric
            st.subheader('Detailed Metrics View')

            metric_selected = st.selectbox('Select a metric to compare:', metrics)

            alg1_score = df.loc[df['Metric'] == metric_selected, 'KNN'].values[0]
            alg2_score = df.loc[df['Metric'] == metric_selected, 'Random Forest'].values[0]

            st.write(f"### {metric_selected}")
            st.write(f"**KNN**: {alg1_score}")
            st.write(f"**Random Forest**: {alg2_score}")

            if alg1_score > alg2_score and metric_selected != 'Log Loss':
                st.success(f"KNN performs better in {metric_selected} with a score of {alg1_score}.")
            elif alg1_score < alg2_score and metric_selected != 'Log Loss':
                st.success(f"Random Forest performs better in {metric_selected} with a score of {alg2_score}.")
            elif alg1_score > alg2_score and metric_selected == 'Log Loss':
                st.success(f"Random Forest performs better in {metric_selected} with a score of {alg2_score}.")
            elif alg1_score < alg2_score and metric_selected == 'Log Loss':
                st.success(f"KNN performs better in {metric_selected} with a score of {alg1_score}.")
            else:
                st.info(f"Both algorithms perform equally in {metric_selected} with a score of {alg1_score}.")

            st.write("### Summary")
            for i in range(len(metrics)):
                st.write(f"- **{metrics[i]}**: KNN: {df['KNN'][i]:.3f} | Random Forest: {df['Random Forest'][i]:.3f}")
        else:
            st.warning("No data available for comparison.")
        