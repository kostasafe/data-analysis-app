![Ionianlogo](https://github.com/kostasafe/data-analysis-app/assets/22775121/ff493346-d82f-4f3a-9e9c-d6d388a65c7c) 

## IONIAN UNIVERSITY - DEPARTMENT OF INFORMATICS



# Data Analysis Application
This Streamlit-based application provides a comprehensive data analysis tool for performing various classification and clustering algorithms on datasets. It supports CSV and Excel file uploads and delivers intuitive visualizations and performance metrics for machine learning models.</br>

## Features

1. **File Upload**
    - Upload your dataset in CSV or Excel format.
    - The app processes and displays the dataset for further analysis.

2. **Data Preprocessing**
    - Automatically handles numeric and categorical data.
    - Standardizes the features for optimal performance.

3. **Classification Algorithms**
    - **k-NN (k-Nearest Neighbors) Classification**
        - Splits the dataset into training and testing sets.
        - Standardizes features and reduces dimensions using PCA.
        - Trains a k-NN classifier and provides accuracy, confusion matrix, and classification report.
        - Visualizes the decision boundary in a 2D plot.
    - Additional classification algorithms can be easily integrated.

4. **Clustering Algorithms**
    - **k-Means Clustering**
        - Cluster analysis and visualization.
        - Computes and displays inertia and silhouette score.
    - Placeholder for additional clustering algorithms. (DBSCAN)

5. **Results Visualization**
    - Confusion matrix and classification reports for classification models.
    - 2D decision boundary visualization using PCA.
    - Interactive and static plots using Matplotlib and Seaborn.

## Requirements

- **Python 3.7+**
- **Streamlit** for the web app framework.
- **Pandas** for data manipulation.
- **Numpy** for numerical computations.
- **Matplotlib** and **Seaborn** for plotting and visualizations.
- **scikit-learn** for machine learning algorithms.
- **OpenPyXL** for Excel file support.

Install the required packages using the following command:
  ```bash
  pip install streamlit pandas numpy matplotlib seaborn scikit-learn openpyxl
  ```

## Usage
1. Clone the repository!
  ```bash
  git clone https://github.com/yourusername/data-analysis-app.git
  cd data-analysis-app
   ```
2. Run the Streamlit app:
  ```bash
  streamlit run data_app.py
  ```
3. Open your browser and navigate to http://localhost:8501 to access the app. (Only if you are not redirected there instantly)

### Project Structure

  ```tree
   data-analysis-app/
   │
   ├── pages/
   │   ├── Home_Page.py                 # Home page for file upload
   │   ├── 2D Visualization             # 2D Visualization page
   │   ├── Clustering_Algorithms.py     # Clustering algorithms page
   │   ├── Classification_Algorithms.py # Classification algorithms page    
   │   ├── Comparison.py                # Comparison page
   │   └── Information.py               # Information page
   │
   ├── data_app.py                      # Main entry point for the Streamlit app
   ├── requirements.txt                 # List of required packages
   └── README.md                        # This readme file
   ```

### Acknowledgments

Streamlit for providing an easy-to-use web application framework.
The developers of Pandas, Numpy, Matplotlib, Seaborn, and scikit-learn for their invaluable libraries.

<details>
<summary> Contributors </summary>
Persefoni Megaliou, Afentoulis Konstantinos, Aggelos Kalocheris
</details>
