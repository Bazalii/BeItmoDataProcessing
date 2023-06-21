import numpy as np

LABELS = ['no_emotion', 'joy', 'sadness', 'surprise', 'fear', 'anger']

def softmax(x):
    exponential = np.exp(x)
    probabilities = exponential / np.sum(exponential)
    return probabilities

def emotion_rating(probabilitie, emotion_type):
    if emotion_type == 'no_emotion':
        return probabilitie * 60
    elif emotion_type == 'joy':
        return probabilitie * 100
    elif emotion_type == 'sadness':
        return probabilitie * 30
    elif emotion_type == 'surprise':
        return probabilitie * 80
    elif emotion_type == 'fear':
        return probabilitie * 20
    elif emotion_type == 'anger':
        return probabilitie * 50

def predict_emotion_for(text, session, tokenizer):
    text_tokens = dict(tokenizer(text, return_tensors='np'))
    for key in text_tokens:
        text_tokens[key] = text_tokens[key].astype(np.int64)
    session_output = session.run(None, text_tokens)
    probabilities = softmax(session_output[0])[0]
    emotion_number = probabilities.argmax()
    emotion = LABELS[emotion_number]
    return max(min(emotion_rating(probabilities[emotion_number], emotion), 100), 0), emotion


