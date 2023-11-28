import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from collections import Counter
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def count_occurence(text):

    '''
    fungsi ini menerima text dalam bentuk kalimat atau paragraf
    '''

    # menghapus spasi dan memisahkan setiap kata yang terpisah dengan spasi
    text_list = text.str.strip().str.split()

    # container list
    total_texts=[]
    try:
    # nested loop yang akhirnya masukan setiap kata pada total_texts
        for content in text_list:
            for kata in content:
                total_texts.append(kata)
    except:
        pass

    # stopwords
    stopwords_en = list(set(stopwords.words('english')))

    # list comprehension untuk setiap kata yang tidak ada pada stopwords
    rem_stopwords = [word for word in total_texts if word not in stopwords_en]

    # mengembalikan hasil dengan Counter dari collections
    return Counter(rem_stopwords)


def run():
    st.title("EDA SPAM Text Classification")
    st.write('''
            Untuk lebih memahami model yang telah dibuat, di halaman ini disediakan beberapa eksplorasi sederhana terkait dataset yang digunakan
            ''')

    df = pd.read_csv('data_eda.csv')

    # chart 1
    st.markdown("<h4 style='text-align: center; color: white;'>Class Comparison</h4>", unsafe_allow_html=True)
    label_counts = df['Label'].value_counts()

    fig = plt.figure(figsize=(9,5)) 
    plt.pie(label_counts, autopct='%1.2f%%', labels=label_counts.index,explode=[0,0.1])
    st.pyplot(fig)
    st.write("Jumlah label non-Spam lebih sedikit dari label spam. Dengan proporsi data yang ada, data memiliki sifat yang imbalanced atau tidak seimbang.")
    st.markdown('---')

    st.write("### Before and After Text Preprocessing")
    st.write(''' 
    Text preprocessing terdiri dari beberapa proses seperti:
    - Mengubah setiap kata menjadi lowercase
    - Menghapus mention seperti @xxx
    - Menghapus new line atau ganti baris
    - Menghapus spasi di awal dengan akhir
    - Menghapus Url http: atau www.
    - Menghapus karakter-karakter spesial
    - Menghapus stopwords termasuk kata yang mengandung apostrophe seperti you're, would've dll.
    - Tokenisasi dan Lemmatization    
    ''')


    # chart 2
    all_class = 5523
    Non_Spam = 4046
    Spam = 1907
    all_class_after = 2826
    Non_Spam_after = 2275
    Spam_after = 833

    value_before = [all_class, Non_Spam, Spam]
    value_after = [all_class_after, Non_Spam_after, Spam_after]
    categories = ['All Classes', 'Non-Spam', 'Spam']
    st.markdown("<h4 style='text-align: center; color: white;'>Unique Words</h4>", unsafe_allow_html=True)

    fig = plt.figure(figsize=(15,7))

    plt.subplot(1,2,1)
    ax = sns.barplot(x=categories, y=value_before)
    plt.bar_label(ax.containers[0], label_type='center', color='white')
    plt.yticks(range(0, 6000, 500))
    plt.ylabel('Total Unique Words')
    plt.title("Before")

    plt.subplot(1,2,2)
    ax = sns.barplot(x=categories, y=value_after)
    plt.bar_label(ax.containers[0], label_type='center', color='white')
    plt.yticks(range(0, 6000, 500))
    plt.ylabel('Total Unique Words')
    plt.title("After")


    st.pyplot(fig)
    st.write(''' 
            Setelah text processing, jumlah unik data pada kedua kelas berkurang secara signifikan dan bisa membuat proses trainig model lebih cepat.
            Namun, jumlah kata unik dalam kelas spam jauh lebih sedikit dibanding kelas non-spam. Hal tersebut dapat mempengaruhi kinerja model dalam memprediksikan kelas spam.
            ''')
    
    st.markdown('---')
    st.markdown("<h4 style='text-align: center; color: white;'>Top 10 Most Common Words</h4>", unsafe_allow_html=True)

    #before
    # panggil fungsi di atas dengan memasukan series
    word_count = count_occurence(df['Message_body'])
    # panggil fungsi dengan filter
    non = count_occurence(df['Message_body'][df['Label']=='Non-Spam'])
    spam = count_occurence(df['Message_body'][df['Label']=='Spam'])
    sorted_all = word_count.most_common()[:10]
    sorted_non = non.most_common()[:10]
    sorted_spam = spam.most_common()[:10]

    #after
    word_count_after = count_occurence(df.cleaned)
    # panggil fungsi dengan filter
    non_after = count_occurence(df['cleaned'][df['Label']=='Non-Spam'])
    spam_after = count_occurence(df['cleaned'][df['Label']=='Spam'])
    sorted_all_after = word_count_after.most_common()[:10]
    sorted_non_after = non_after.most_common()[:10]
    sorted_spam_after = spam_after.most_common()[:10]

    # list hasil most_common di atas
    list_sorted = [sorted_all, sorted_non, sorted_spam]
    list_sorted_after = [sorted_all_after, sorted_non_after, sorted_spam_after]

    # list label
    name = ['All Data', 'Non-Spam Class', 'Spam Class']

    # visualisasi
    figure = plt.figure(figsize=(20,6))
    plt.suptitle("Before Text Preprocessing")
    for i, sorted in enumerate(list_sorted):
        labels, values = zip(*sorted)
        plt.subplot(1,3,i+1)
    # plot data
        plt.barh(labels, values)
        plt.xlabel('Count')
        plt.title(f'Top 10 Most Common Word {name[i]}')
        plt.gca().invert_yaxis()
    st.pyplot(figure)

    figure = plt.figure(figsize=(20,6))
    plt.suptitle("After Text Preprocessing")
    for i, sorted in enumerate(list_sorted_after):
        labels, values = zip(*sorted)
        plt.subplot(1,3,i+1)
        # plot data
        plt.barh(labels, values)
        plt.xlabel('Count')
        plt.title(f'Top 10 Most Common Word {name[i]}')
        plt.gca().invert_yaxis()
    st.pyplot(figure)
    st.write('''
            Terlihat adanya perbedaan pada top list setiap kategori. 
            Pada spam class terlihat kata yang mendominasi adalah call, free, txt, claim dan lain-lain. Kata-kata tersebut bisa menjadi alasan paling berpengaruh kenapa sebuah email bisa dikategorikan sebagai spam.
            ''')
    # chart 3
if __name__ == "__main__":
    run()