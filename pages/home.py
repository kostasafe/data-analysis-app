import streamlit as st
import pandas as pd
import csv
from io import StringIO

def show_Home(): 
        st.header("Καλωσήρθατε στην Εφαρμογή Εξόρυξης και Ανάλυσης δεδομένων!")
        st.write( "Παρακαλώ φορτώστε ένα αρχείο:")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file")
        if uploaded_file is not None:
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                st.write(bytes_data)








