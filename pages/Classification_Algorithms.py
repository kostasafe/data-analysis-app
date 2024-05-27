import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

def show_Classification_Algorithms():
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
    data = st.session_state.new_dataset
    target_column_name = st.session_state.target_column.name


    st.markdown('<div class="title">Classification Algorithms</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.session_state.uploaded_file is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('<div class="header">k-NN Classification</div>', unsafe_allow_html=True) 
            st.divider()
            accuracy, cm, report = knn_classification(data, target_column_name)
            visualize_results(accuracy, cm, report)
        
        with right_column:
            st.markdown('<div class="header-two">SOON..</div>', unsafe_allow_html=True)
            st.divider()
            # Place your KMeans clustering code here
    else:
        st.warning("Please upload a CSV or an Excel file in Home Page to proceed.")

def knn_classification(data, target_column_name):
    # Separate features and target
    features = data.drop(columns=[target_column_name])
    target = data[target_column_name]

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return accuracy, cm, report

def visualize_results(accuracy, cm, report):
    st.write(f"Accuracy: {accuracy:.2f}")
    
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)
    
    st.write("Classification Report:")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# Ensure that this function is called in your main script
if __name__ == "__main__":
    show_Classification_Algorithms()
