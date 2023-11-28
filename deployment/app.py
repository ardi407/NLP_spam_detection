import streamlit as st
import eda
import prediction

halaman = st.sidebar.selectbox('Halaman: ',('EDA', 'Prediction'))
if halaman == 'EDA':
    eda.run()
else:
    prediction.run()