import streamlit as st
import pandas as pd


def show_Home(): 
        st.title("Καλωσήρθατε στην Εφαρμογή Εξόρυξης και Ανάλυσης δεδομένων!")
        st.header( "Παρακαλώ φορτώστε ένα αρχείο:")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type={"csv", "excel"})
        if uploaded_file not in st.session_state:
                st.session_state.uploaded_file = uploaded_file
        if uploaded_file is not None:
                # To read file as bytes:
                #bytes_data = uploaded_file.getvalue()
                new_list = load_data(uploaded_file)
                if new_list not in st.session_state:
                        st.session_state.new_dataset = new_list #store our dataframe to session_state
                generate_button_1 = st.button('Πατήστε εδώ για προβολή του dataset όπως διαβάστηκε από το αρχείο!')
                if generate_button_1:
                        st.write(new_list)
                st.header("Ποια στήλη αντιστοιχεί στη μεταβλητή εξόδου;")
                column_names = st.session_state.new_dataset.columns.tolist() #taking column headers and convert them to a list
                column_names_str = " | ".join(column_names) #split them with |
                column_names_formatted = f"**<b>Columns:</b>** [{column_names_str}]" #make it prettier
                st.markdown(column_names_formatted, unsafe_allow_html=True)
                output_column = st.text_input("Παρακαλώ γράψτε μου το ακριβές όνομα όπως αναγράφεται παραπάνω!")
                generate_button_2 = st.button('Πατήστε εδώ για προβολή του διορθωμένου dataset!')
                if generate_button_2:
                        if output_column :
                                st.session_state.target = st.session_state.new_dataset.pop(output_column)
                                left_column, right_column = st.columns(2)
                                with left_column:
                                        st.write("¨Νέο Dataframe:")
                                        st.write(st.session_state.new_dataset)

                                with right_column:
                                        st.write("Μεταβλητή Εξόδου:")
                                        st.write(pd.DataFrame({output_column: st.session_state.target}))
                                        st.balloons()
                        


def load_data(uploaded_file):   
        if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                return df
        elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                return df