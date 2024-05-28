import streamlit as st

def show_Info(): 
        st.subheader("Οδηγίες χρήσης")
        st.markdown(
                        """
                        Test list  with bullets:
                        - Οδηγίες
                        - Κι άλλες οδηγίες
                        - Οδηγιέισον
                        """
        )
        st.subheader("Πληροφορίες")
        st.write("Αυτή η εφαρμογή δημιουργήθηκε για την εκπλήρωση του μαθήματος [Τεχνολογία Λογισμικού](https://opencourses.ionio.gr/courses/DDI259/)")
        st.subheader("Development Team")
        st.markdown(
                        """ 
                         [devpersi](https://github.com/devpersi): 2D Visualization, DBScan Clustering Algorithm         
                         [kostasafe](https://github.com/kostasafe): Project Setup, 2D Visualization, K-NN Classification Algorithm     
                         [p15kalo](https://github.com/p15kalo): KMeans Clustering Algorithm     
                        """
        )        
        
        