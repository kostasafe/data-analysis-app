import streamlit as st
import pandas as pd


def show_Home():
        st.markdown(
        """
        <style>
        .title {
            font-size: 70px;
            text-align: center;
            margin-bottom: 0px;
            margin-top: 0px;
        }
         </style>
        """, unsafe_allow_html=True
        )
        st.markdown('<div class="title">Καλωσήρθατε στην Εφαρμογή Εξόρυξης και Ανάλυσης δεδομένων!</div>', unsafe_allow_html=True)
        st.header( "Παρακαλώ φορτώστε ένα αρχείο:", divider='rainbow')
        uploaded_file = st.file_uploader("Choose a :green[CSV] or :green[Excel] file", type={"csv", "excel"})
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
                
                output_column = st.selectbox("Παρακαλώ διαλέξτε τη μεταβλητή εξόδου!", st.session_state.new_dataset.columns[::-1])
                if output_column not in st.session_state:
                        st.session_state.output_column = output_column

                #column_names = st.session_state.new_dataset.columns.tolist() #taking column headers and convert them to a list
                #column_names_str = " | ".join(column_names) #split them with |
                #column_names_formatted = f"**<b>Columns:</b>** [{column_names_str}]" #make it prettier
                #st.markdown(column_names_formatted, unsafe_allow_html=True)
                #output_column = st.text_input("Παρακαλώ γράψτε μου το ακριβές όνομα όπως αναγράφεται παραπάνω!")
                generate_button_2 = st.button('Πατήστε εδώ για προβολή του διορθωμένου dataset!')
                if output_column :
                        if generate_button_2:    
                                dataset_without_column = st.session_state.new_dataset.copy() #if we dont make a copy like this our new_dataset saved dataframe will be lost
                                st.session_state.target_column = dataset_without_column.pop(output_column)
                                st.session_state.dataset_with_no_label = dataset_without_column
                                
                                # Display the modified DataFrame in left_column
                                left_column, right_column = st.columns(2)
                                with left_column:
                                        st.write("Νέο Dataframe:")
                                        st.write(dataset_without_column)
                
                                # Display the excluded column in right_column
                                with right_column:
                                        st.write("Μεταβλητή Εξόδου:")
                                        st.write(st.session_state.target_column)
                                        st.balloons()

                        


def load_data(uploaded_file):   
        if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
        else:
                st.write("This app only works with .csv and .xlsx files.")
                return
        df.dropna(inplace=True) # Remove all rows where any column is missing
        return df