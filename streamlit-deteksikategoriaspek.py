import pickle
import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
import re
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def casefolding(opinion):
    opinion = opinion.casefold()
    return opinion

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

def stop_words(opinion):
    filteredsentence=[]
    stop_words = set(stopwords.words('english'))
    wordtokenize = word_tokenize(opinion)
    for word in wordtokenize:
        if word not in stop_words:
            filteredsentence.append(word)
    final_list=' '.join(filteredsentence)
    return final_list

def lemmatize_text(text):
    wnl = WordNetLemmatizer()
    tokens = word_tokenize(text)  # Tokenisasi kata
    pos_tags = nltk.pos_tag(tokens)  # Menentukan pos tag untuk setiap kata

    # Menggunakan pos tag untuk memberikan informasi yang tepat kepada lemmatizer
    lemmatized_words = []
    for token, pos in pos_tags:
        pos_tag = get_wordnet_pos(pos)
        lemmatized_word = wnl.lemmatize(token, pos=pos_tag)
        lemmatized_words.append(lemmatized_word)

    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenisasi(content):
    content = nltk.tokenize.word_tokenize(content)
    return content

def preprocess(content):
    content = casefolding(content)
    content = remove_punctuation(content)
    content = stop_words(content)
    content = lemmatize_text(content)
    # content = tokenisasi(content)
    return content

nltk.download('punkt')

# load save model
model_aspek = pickle.load(open('deteksikategoriaspek_model.sav', 'rb'))
tf_idf_data = pickle.load(open('tf_idf_data.sav', 'rb'))
data = pickle.load(open('data.sav', 'rb'))

# judul halaman
# st.title("Deteksi Kategori Aspek pada Ulasan Restoran dengan Metode Support Vector Machine")
st.markdown("<h2 style='text-align: center; color: system;'>Deteksi Kategori Aspek pada Ulasan Restoran dengan Metode Support Vector Machine</h2>",
            unsafe_allow_html=True)
st.markdown("<hr></hr>", unsafe_allow_html=True)

####### Prediksi #######
st.markdown("#### Prediksi Kelas Aspek pada Ulasan Restoran")
text_input = st.text_input("Enter Your Review about This Restaurant:")
review = {'Review': [text_input]}
new_data = pd.DataFrame(review)
new_data['preprocess'] = new_data['Review'].apply(preprocess)
new_data = new_data.loc[:, ['preprocess']]
new_data = new_data.rename(columns={"preprocess": "Review"})

def Tokenize(data):
    data['review_token'] = ""
    data['Review'] = data['Review'].astype('str')
    for i in range(len(data)):
        data['review_token'][i] = data['Review'][i].lower().split()
        all_tokenize = sorted(list(set([item for sublist in data['review_token'] for item in sublist])))
    return data, all_tokenize

def hitungTF(data, all_tokenize):
    from operator import truediv
    token_cal = Tokenize(data)
    data_tokenize = token_cal[0]
    for item in all_tokenize:
      data_tokenize[item] = 0
    for item in all_tokenize:
        for i in range(len(data_tokenize)):
            if data_tokenize['review_token'][i].count(item) > 0:
                a = data_tokenize['review_token'][i].count(item)
                b = len(data_tokenize['review_token'][i])
                c = a / b
                data_tokenize[item] = data_tokenize[item].astype('float')
                data_tokenize[item][i] = c
    return data_tokenize

def tfidf(data, new_data=new_data, tf_idf_data=tf_idf_data):
    tf_idf = tf_idf_data
    N = len(data)
    all_tokenize = Tokenize(data)[1]
    df = {}
    for item in all_tokenize:
        df_ = (tf_idf[item] > 0).sum()
        df[item] = df_
        idf = (np.log(N / df_))
        tf_idf[item] = tf_idf[item] * idf

    if new_data is not None:
        new_tf = hitungTF(new_data, all_tokenize)

        for item in all_tokenize:
            if item in new_tf.columns:
                df_ = df.get(item, 0)
                idf = (np.log(N / (df_)))
                new_tf[item] = new_tf[item] * idf

        new_tf.drop(columns=['Review', 'review_token'], inplace=True)

        return new_tf, df
    else:
        return tf_idf, df
    
tfidf_result, document_frequency = tfidf(data, new_data)
    
# aspect
def testing_aspek(W_aspek, data_uji_aspek):
    prediksi_aspek = np.array([])
    for i in range(data_uji_aspek.shape[0]):
        y_prediksi_aspek = np.sign(np.dot(W_aspek, data_uji_aspek.to_numpy()[i]))
        prediksi_aspek = np.append(prediksi_aspek, y_prediksi_aspek)
    return prediksi_aspek

def testing_oneagaintsall_aspek(W_aspek, data_uji_aspek):
    list_kelas_aspek = W_aspek.keys()
    hasil_aspek = pd.DataFrame(columns=W_aspek.keys())
    for kelas_aspek in list_kelas_aspek:
        hasil_aspek[kelas_aspek] = testing_aspek(W_aspek[kelas_aspek], data_uji_aspek)
    kelas_prediksi_aspek = hasil_aspek.idxmax(1)
    return kelas_prediksi_aspek

prediksi_aspek = testing_oneagaintsall_aspek(model_aspek, new_data)

prediksi = st.button("Hasil Prediksi")
if prediksi:
    for aspek in prediksi_aspek:
        st.success(f"Aspek {aspek}")
st.markdown("<hr></hr>", unsafe_allow_html=True)