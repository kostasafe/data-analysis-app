import streamlit as st
import csv
import pandas as pd


def show_Home(): 
        st.header("Καλωσήρθατε στην Εφαρμογή Εξόρυξης και Ανάλυσης δεδομένων!")
        st.write( "Παρακαλώ φορτώστε ένα αρχείο:")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type={"csv", "excel"})
        if uploaded_file is not None:
                # To read file as bytes:
                #bytes_data = uploaded_file.getvalue()
                generate_button = st.button('Πατήστε εδώ για προβολή του dataset ως λίστα!')
                new_list = load_data(uploaded_file)
                if new_list not in st.session_state:
                        st.session_state.new_dataset = new_list #store our dataframe to session_state
                if generate_button:
                        st.write(new_list)
                        


def load_data(uploaded_file):   
        if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                return df