import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, classification_report
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
    

    st.markdown('<div class="title">Classification Algorithms</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_file is not None:
        if st.session_state.output_column is not None:
            
            data = st.session_state.new_dataset
            target_column_name = st.session_state.target_column.name

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
                st.markdown('<div class="header-two">Random Forest Classification</div>', unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                accuracy, cm, report, X_train_pca, X_test_pca, y_train, y_test, y_pred = random_forest_classification(data, target_column_name)
                visualize_results(accuracy, cm, report)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="header-two">Random Forest 2D Visualization with PCA Dimension Reduction</div>', unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                visualize_random_forest_decision_boundary(X_train_pca, X_test_pca, y_train, y_test, y_pred)
        else:
            st.warning("Please select a target column in Home Page to proceed.")
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
    y_pred_proba = knn.predict_proba(X_test_pca)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    logloss = log_loss(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.session_state.knn_scores = [accuracy, precision, recall, f1, roc_auc, logloss]
    
    return accuracy, cm, report, X_train_pca, X_test_pca, y_train, y_test, y_pred

def random_forest_classification(data, target_column_name):
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

    n_estimators = st.number_input("Enter the number of estimators for Random Forest:", min_value=1, value=2, key='rf_n_estimators')

    # Create and train the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X_train_pca, y_train)

    # Make predictions
    y_pred = rf.predict(X_test_pca)
    y_pred_proba = rf.predict_proba(X_test_pca)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    logloss = log_loss(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.session_state.random_forest_scores = [accuracy, precision, recall, f1, roc_auc, logloss]

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

    # Ensure xx and yy are of type float64 for plotting
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

def visualize_random_forest_decision_boundary(X_train_pca, X_test_pca, y_train, y_test, y_pred):
    # Plot the decision boundary by assigning a color in the color map to each mesh point
    h = .02  # step size in the mesh

    # define color maps
    cmap_light = plt.cm.RdYlBu
    cmap_bold = ['#FF0000', '#00FF00', '#0000FF']

    # Train the classifier on the PCA transformed training set
    n_estimators = st.number_input("Enter the number of estimators for Random Forest:", min_value=1, value=2, key='rf_n_estimators_boundary')
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X_train_pca, y_train)

    # Get the minimum and maximum values of the principal components
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    # Generate a grid of points using numpy's meshgrid function
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Ensure xx and yy are of type float64
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)

    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions on the mesh grid
    Z = rf.predict(mesh_points)

    # Ensure Z is of type float64 and has correct shape
    Z = Z.reshape(xx.shape).astype(np.float64)

    # Put the result into a color plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot also the training points
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette=cmap_bold, alpha=0.6, edgecolor='k', ax=ax)
    sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_pred, palette=cmap_bold, alpha=0.6, edgecolor='k', marker='X', ax=ax)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2D Decision Boundary of Random Forest')
    st.pyplot(fig)