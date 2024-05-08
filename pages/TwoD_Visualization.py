import pandas as pd
import csv
import streamlit as st

def show_TwoD_Visualization():
        st.header("Εδώ θα μετατρέψουμε τον πίνακα που φορτώθηκε από το αρχείο της επιλογής σας!")
        column_names = st.session_state.new_dataset.columns.tolist() #taking column headers and convert them to a list
        column_names_str = " | ".join(column_names) #split them with |
        column_names_formatted = f"**<b>Columns:</b>** [{column_names_str}]" #make it prettier
        st.markdown(column_names_formatted, unsafe_allow_html=True)
        st.write("Ποια στήλη αντιστοιχεί στη μεταβλητή εξόδου;")
        output_column = st.text_input("Παρακαλώ γράψτε μου το ακριβές όνομα όπως αναγράφεται παραπάνω!" ,key="output_column")
        if output_column:
            fixed_dataframe = transform_dataframe(st.session_state.new_dataset , output_column)
            st.write(fixed_dataframe)
            st.balloons()


def transform_dataframe(df , column):
        if column :
                df = df[[col for col in df.columns if col != column] + [column]] #dataframe convertion so it fits the users request
        return df