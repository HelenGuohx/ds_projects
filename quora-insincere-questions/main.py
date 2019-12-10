# %%

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import string
import json
import numpy as np
import matplotlib.pyplot as plt
import re
# import nltk
# # nltk.download('stopwords')
from nltk.corpus import stopwords


# %%

import matplotlib
from sklearn.decomposition import PCA
import spacy  # tokenize text
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# sp = spacy.load("en_core_web_sm")
pd.set_option('display.max_colwidth', -1)


# under_sampling to balance data
def under_sampling(df, percent=1):
    majority = df[df['target'] == 0]
    minority = df[df['target'] == 1]
    lower_data = majority.sample(n=int(percent * len(minority)), replace=False, random_state=890, axis=0)
    return (pd.concat([lower_data, minority]))


# over sampling to balance data
def over_sampling(df, percent=1):
    # 通过numpy随机选取多数样本的采样下标
    '''
    percent:多数类别下采样的数量相对于少数类别样本数量的比例
    '''
    most_data = df[df['label'] == 1]  # 多数类别的样本
    minority_data = df[df['label'] == 0]  # 少数类别的样本
    index = np.random.randint(len(most_data), size=int(percent * len(minority_data)))
    # 下采样后数据样本
    lower_data = most_data.iloc[list(index)]  # 下采样


# replace unicode space character with space ' '
spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0']

def replace_space(text):
    for s in spaces:
        text = text.replace(s, ' ')
    return text


# clean rare words
with open('rare_words.json') as f:
    rare_words_mapping = json.load(f)
    # print(rare_words_mapping)


def clean_rare_words(text):
    for w in rare_words_mapping:
        if text.count(w) > 0:
            text = text.replace(w, rare_words_mapping[w])
    return text


# decontracted
def clean_decontracted(text):
    # specific
    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    # general
    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)
    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)s ", " is ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)
    return text


# misspelling
with open('misspell_words.json') as f:
    misspell_words_mapping = json.load(f)


def clean_misspell(text):
    for w in misspell_words_mapping:
        if text.count(w) > 0:
            text = text.replace(w, misspell_words_mapping[w])
    return text


# replace punctuation with space
def replace_punctuation(text):
    punct = str.maketrans('', '', string.punctuation)
    return text.translate(punct)


# clean repeated letters
def clean_repeat_words(text):
    text = text.replace("img", "ing")

    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
    text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
    text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
    text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
    text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
    text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
    text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
    text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
    text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
    text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
    text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
    text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
    text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
    text = re.sub(r"(Q|q)(Q|q)+", "q", text)
    text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
    text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
    text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
    text = re.sub(r"(V|v)(V|v)+", "v", text)
    text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
    text = re.sub(r"plzz+", "please", text)
    text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)
    return text


# make text lower case
def lower_words(text):
    return text.lower()


stop_words = stopwords.words('english')


def remove_stopwords(text):
    """
    remove stop words and extra space
    params: string
    return: list
    """
    words = text.split()
    new_words = []
    for w in words:
        if w not in stop_words and w != ' ':
            new_words.append(w)
    return ' '.join(new_words)


def stemming(text):
    pass


# apply all the clean methods
def text_cleaning(text):
    text = replace_space(text)
    text = clean_rare_words(text)
    text = clean_decontracted(text)
    text = clean_misspell(text)
    text = replace_punctuation(text)
    text = clean_repeat_words(text)
    text = lower_words(text)
    text = remove_stopwords(text)
    return text


# %%

# data after cleaning
# clean_text = sample.question_text.apply(lambda x: text_cleaning(x))


# text_cleaning('i like apples, just like hahahhhhh?')
#
#
# itext = 'i like apples, just like hahahhhhh?'
# for i in itext.split():
#     if i in stop_words:
#         print(i)


# %%


# %%

# keras is on top of TensorFlow
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# tokenize text
# embed_size = 300 #词向量维度
# max_features = 95000 #设置词典大小
# max_len = 70 #设置输入的长度

# tokenizer = Tokenizer(num_words = max_features)
# tokenizer.fit_on_texts(list(train_X))
# train_X = tokenizer.texts_to_sequences(train_X)
# test_X = tokenizer.texts_to_sequences(test_X)

# # pad the sentences
# train_X = pad_sequences(train_X, maxlen = max_len)
# test_X = pad_sequences(test_X, maxlen = max_len)


# %% md

# word embedding

# %%

def load_embed(typeToLoad):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float16')

    if typeToLoad == "glove":
        file = 'embeddings/glove.840B.300d/glove.840B.300d.txt'
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o) > 100)
    elif typeToLoad == "word2vec":
        # file = 'embeddings⁩/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin⁩'
        file = 'embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
        embeddings_index = KeyedVectors.load_word2vec_format(file, binary=True)  # query word vector from the file
    elif typeToLoad == "fasttext":
        # file = "⁨embeddings⁩/wiki-news-300d-1M⁩/wiki-news-300d-1M.vec"
        file = 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    return embeddings_index



def vectorize(text, embeddings_index):
    text_list = text.split()
    vectors = []
    for word in text_list:
        if word in embeddings_index:
            vector = embeddings_index[word]
            vectors.append(vector)
    avg_vectors = np.mean(np.array(vectors), axis=0) if vectors else [0] * 300
    return avg_vectors


# %%

def compute_oov_rate(text, embeddings_index):
    text_list = text.split()
    num_of_words = len(text_list)
    num_of_known_words = 0
    for word in text_list:
        if word in embeddings_index:
            num_of_known_words += 1
    oov_rate = 1 - num_of_known_words / num_of_words if num_of_words else None
    return oov_rate

# %%

embed_glove = load_embed('glove')

# %%

# embed_word2vec = load_embed('word2vec')
# embed_fasttext = load_embed('fasttext')

# %%

# vector = vectorize(itext, embed_word2vec)


# word coverage
# def build_vocab(texts):
#     sentences = texts.apply(lambda x: x.split()).values
#     vocab = {}
#     for sentence in sentences:
#         for word in sentence:
#             try:
#                 vocab[word] += 1
#             except KeyError:
#                 vocab[word] = 1
#     return vocab
#

# def check_coverage(vocab, embeddings_index):
#     known_words = {}
#     unknown_words = {}
#     num_known_words = 0
#     num_unknown_words = 0
#     for word in vocab.keys():
#         if word in embeddings_index:
#             known_words[word] = embeddings_index[word]
#             num_known_words += vocab[word]
#
#         else:
#             unknown_words[word] = vocab[word]
#             num_unknown_words += vocab[word]
#
#     print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
#     print('Found embeddings for  {:.2%} of all text'.format(num_known_words / (num_known_words + num_unknown_words)))
#     #     unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
#
#     return unknown_words


# def vocab_check_coverage(texts, embed_method):
#     vocab = build_vocab(texts)
#     oov_words = check_coverage(vocab, embed_method)
#     oov = {"oov_rate": len(oov_words) / len(vocab), 'oov_words': oov_words}
#     print("oov_rate", oov["oov_rate"])
#
#     return oov




# concate_features
def concate_features(df):
    feature_matrix = []
    cnt = 0
    for row in df.iterrows():
        x = row[1]
        new_vectors = x['word_vector']
        new_vectors = np.append(new_vectors, x["oov_rate"])
        new_vectors = np.append(new_vectors, x["text_len"])
        new_vectors = np.append(new_vectors, x["clean_text_len"])
        feature_matrix.append(new_vectors)
    return feature_matrix



# PCA

def reduce_demension(X, n):
    """
    X: features matrix
    n: number of compoments or total explained ratio we want
    return:
    ev: explained variance of each component
    evr: explained variance ratio of each component
    """
    pca = PCA(n_components=n)
    pca.fit(X)
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    return ev, evr


# %%
dataset = pd.read_csv("train.csv")
# test using small sample to do
sample = dataset.sample(frac=0.001, random_state=100)  # 261224 rows

print("sample",np.shape(sample))
# split the data set into train and test
train_set, test_set = train_test_split(sample, test_size=0.2, random_state=42)

print("train_set",train_set.target.value_counts())
print("test_set",test_set.target.value_counts())



sample_train_set = under_sampling(train_set, percent=5)
print('sample_train_set', np.shape(sample_train_set))
# %%

sample_train_set['text_len'] = sample_train_set.question_text.apply(lambda x: len(x.split()))
print("sample_train_set['text_len']", np.shape(sample_train_set['text_len'][:2]))
# %%

sample_train_set["clean_text"] = sample_train_set.question_text.apply(lambda x: text_cleaning(x))
print("sample_train_set['clean_text']", np.shape(sample_train_set['clean_text']))
# %%

sample_train_set['clean_text_len'] = sample_train_set.clean_text.apply(lambda x: len(x.split()))
print("sample_train_set['clean_text_len']", np.shape(sample_train_set['clean_text_len']))

# %%

sample_train_set["word_vector"] = sample_train_set.clean_text.apply(lambda x: vectorize(x, embed_glove))
print("word_vector")
# %%

sample_train_set["oov_rate"] = sample_train_set.clean_text.apply(lambda x: compute_oov_rate(x, embed_glove))
print("oov_rate")
# %%

# standardlize
# text_len_mean = np.mean(sample_train_set["text_len"])
# text_len_std = np.std(sample_train_set["text_len"])
# sample_train_set["text_len_standard"] = sample_train_set['text_len'].apply(lambda x: x - text_len_mean/ text_len_std)
# sample_train_set["clean_text_len_standard"] = sample_train_set['clean_text_len'].apply(lambda x: x - np.mean(x)/ np.std(x))


# %%

# train feature matrix
train_matrix = concate_features(sample_train_set)
print("train_matrix")

# export
train_output = pd.DataFrame(train_matrix)
train_output['target'] = sample_train_set.target
train_output.to_csv("train_output.csv")

# %%

######################Word Embedding############

# %%

# word coverage
# oov_words_glove = vocab_check_coverage(sample_train_set.clean_text, embed_glove)
#
# # %%
#
# % % time
# oov_words_word2vec = vocab_check_coverage(sample_train_set.clean_text, embed_word2vec)
#
# # %%
#
# % % time
# oov_words_fasttext = vocab_check_coverage(sample_train_set.clean_text, embed_fasttext)
#
# # %%

###########################

# %%

