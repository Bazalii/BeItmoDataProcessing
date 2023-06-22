import nltk
import numpy as np
import pymorphy2

from nltk.corpus import stopwords

morph = pymorphy2.MorphAnalyzer()

def lemmatize(text):
    bad = "\"\-();,."
    for elem in bad:
        if elem in bad:
            text = text.replace(elem, " ")
    words = text.split() # разбиваем текст на слова
    res = list()
    my_stopwords = ["также", "ещё", "тут", "в"]
    for word in words:
        if word in stopwords.words('russian') or word in my_stopwords or word in stopwords.words('english'):
            continue
        p = morph.parse(word)[0]
        res.append(p.normal_form)
    return res

def preprocess(text):
    return " ".join(lemmatize(text))

def e_step(X, probs):
    # z = np.matmul( X, np.log(probs).T )
    z = X @ np.log(probs).T
    z = z - np.logaddexp.reduce(z, axis=1).reshape(-1, 1)
    return np.exp(z)

def classify(text, vectorizers, probs):
    LABELS = ["BePro", "BeFriendly", "BeHealthy"]
    LABELS_NUMBER = [0, 0, 0, 1, 2]
    text = preprocess(text)
    vector = vectorizers.transform(np.array([text])).toarray()
    if len(vector[0]) == 0:
        return LABELS[0]
    probabilities = e_step(vector, probs)
    return LABELS[LABELS_NUMBER[probabilities.argmax()]]
    



