import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle5 as pickle

def train_classifier():
    df = pd.read_json('./models/base/classification_data.json')
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['intent'], random_state=0)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    model = LinearSVC(dual="auto").fit(X_train_tfidf, y_train)

    # Save the vectorizer 
    vec_file = './models/base/vectorizer.pickle'
    pickle.dump(count_vect, open(vec_file, 'wb'))

    # Save the model
    mod_file = './models/base/classifier/classifier.model'
    pickle.dump(model, open(mod_file, 'wb'))


def classify_text_base(text):
    # Load the vectorizer
    vec_file = './models/base/classifier/vectorizer.pickle'
    count_vect = pickle.load(open(vec_file, 'rb'))

    # Load the model
    mod_file = './models/base/classifier/classifier.model'
    model = pickle.load(open(mod_file, 'rb'))

    # Classify the text
    text_counts = count_vect.transform([text])
    predicted = model.predict(text_counts)
    scores = model.decision_function(text_counts)

    result = {
        'intent': predicted[0],
        'confidence': max(scores[0]),
        'text': text
    }

    return result

def classify_text_pro(text):
    # Load the vectorizer
    vec_file = './models/pro/classifier/vectorizer.pickle'
    count_vect = pickle.load(open(vec_file, 'rb'))

    # Load the model
    mod_file = './models/pro/classifier/classifier.model'
    model = pickle.load(open(mod_file, 'rb'))

    # Classify the text
    text_counts = count_vect.transform([text])
    predicted = model.predict(text_counts)
    scores = model.decision_function(text_counts)

    result = {
        'intent': predicted[0],
        'confidence': max(scores[0]),
        'text': text
    }

    return result
