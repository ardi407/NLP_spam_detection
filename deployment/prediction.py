import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np


model = tf.keras.models.load_model('spam_identifier_savedmodel')

def run():
    st.markdown("<h3 style='text-align: center; color: white;'>Prediksi Text Spam</h3>", unsafe_allow_html=True)
    st.write("Program ini dibuat untuk mengindentifikasi apakah sebuah text termasuk kategori spam atau tidak")
    with st.form('Form Car Details'):
        text = st.text_area("Masukan Teks (kalimat atau pargaraf)",height=250)
        submitted = st.form_submit_button('Predict')

    if submitted:
        text_input = {"Text":[text]}
        data_inf = pd.DataFrame(text_input).reset_index(drop=True)
        pred = model.predict(data_inf)
        pred = np.where(pred > 0.3, "Spam", "Non-Spam")
        for result in pred:
            st.write(f"#### Prediction: {result[0]}")

if __name__ == '__main__':
    run()