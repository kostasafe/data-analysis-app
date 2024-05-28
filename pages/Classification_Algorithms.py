import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.decomposition import PCA

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

    if st.session_state.uploaded_file is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('<div class="header">k-NN Classification</div>', unsafe_allow_html=True) 
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            accuracy, cm, report, X_train_pca, X_test_pca, y_train, y_test, y_pred = knn_classification(data, target_column_name)
            visualize_results(accuracy, cm, report)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="header">k-NN 2D Visualization with PCA Dimension Reduction</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            visualize_knn_decision_boundary(X_train_pca, X_test_pca, y_train, y_test, y_pred)
        
        with right_column:
            st.markdown('<div class="header-two">SOON..</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            #2nd classific. algo
    else:
        st.warning("Please upload a CSV or an Excel file in Home Page to proceed.")

def knn_classification(data, target_column_name):
    # Separate features and target
    features = data.drop(columns=[target_column_name])
    target = data[target_column_name]

    # Encode target labels to integers
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target) 

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Create and train the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)

    # Make predictions
    y_pred = knn.predict(X_test_pca)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return accuracy, cm, report, X_train_pca, X_test_pca, y_train, y_test, y_pred

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

def visualize_knn_decision_boundary(X_train_pca, X_test_pca, y_train, y_test, y_pred):
    
    # Plot the decision boundary by assigning a color in the color map to each mesh point
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = plt.cm.RdYlBu
    cmap_bold = ['#FF0000', '#00FF00', '#0000FF']

    # Train the classifier on the PCA transformed training set
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)

    # Create a mesh to plot in
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Ensure xx and yy are of type float64
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)

    # Make predictions on the mesh grid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Ensure Z is of type float64 and has correct shape
    Z = Z.reshape(xx.shape).astype(np.float64)

    # Put the result into a color plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot also the training points
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette=cmap_bold, alpha=0.6, edgecolor='k')
    sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_pred, palette=cmap_bold, alpha=0.6, edgecolor='k', marker='X')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D Decision Boundary of k-NN')
    st.pyplot(plt)

#showing which function is the main function to be called
if __name__ == "__main__":
    show_Classification_Algorithms()
