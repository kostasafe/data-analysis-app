import streamlit as st
import csv
import pandas as pd
from io import StringIO


def show_Home(): 
        st.header("Καλωσήρθατε στην Εφαρμογή Εξόρυξης και Ανάλυσης δεδομένων!")
        st.write( "Παρακαλώ φορτώστε ένα αρχείο:")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file")
        if uploaded_file is not None:
                # To read file as bytes:
                bytes_data = uploaded_file.getvalue()
                st.write(bytes_data)




def load_data(filename):   #NOT WORKING YET
        thelist = []
        with open(filename) as test_csv:
                test_csv_data = csv.reader(test_csv, delimiter=',')
                next (test_csv_data) #skip header
                for row in test_csv_data:
                        thelist.append(row)
                return thelist



#new_list = load_data("country_full.csv")  HOW TO CALL THE FUNCTION
#    for row in new_list:
#         print(row)

