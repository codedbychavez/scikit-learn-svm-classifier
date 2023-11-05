# scikit-learn-svm-classifier
A simple svm text classifier


## Code blocks


### Imports

```python
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle5 as pickle
```

### Training the classifier

```python
df = pd.read_json('./classifier/classification_data.json')
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['intent'], random_state=0)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = LinearSVC(dual="auto").fit(X_train_tfidf, y_train)

# Save the vectorizer 
vec_file = './classifier/vectorizer.pickle'
pickle.dump(count_vect, open(vec_file, 'wb'))

# Save the model
mod_file = './classifier/classifier.model'
pickle.dump(model, open(mod_file, 'wb'))
```

### Making a prediction

```python
text = "Whats the weather like today?"
```

```python
# Load the vectorizer
vec_file = './classifier/vectorizer.pickle'
count_vect = pickle.load(open(vec_file, 'rb'))

# Load the model
mod_file = './classifier/classifier.model'
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

print(result)
```

#### Output

```json
{
  'intent': 'get_weather_in_location',
  'confidence': 0.32104388592482813,
  'text': 'Whats the weather like today?'
}
```

### Training data

```json
[
    {
        "intent": "goodbye",
        "text": "later gater"
    },
    {
        "intent": "goodbye",
        "text": "goodbye"
    },
    {
        "intent": "goodbye",
        "text": "bye"
    },
    {
        "intent": "goodbye",
        "text": "see yah"
    },
    {
        "intent": "get_weather_in_location",
        "text": "get me some weather data"
    },
    {
        "intent": "get_weather_in_location",
        "text": "will it rain?"
    },
    {
        "intent": "get_weather_in_location",
        "text": "I wanna know if its snowing"
    },
    {
        "intent": "get_weather_in_location",
        "text": "can you tell me about the weather"
    },
    {
        "intent": "get_weather_in_location",
        "text": "tell me about the weather"
    },
    {
        "intent": "get_weather_in_location",
        "text": "is it raining right now"
    },
    {
        "intent": "get_weather_in_location",
        "text": "what is the weather in france"
    },
    {
        "intent": "greeting",
        "text": "sup"
    },
    {
        "intent": "greeting",
        "text": "you "
    },
    {
        "intent": "greeting",
        "text": "heyyy"
    },
    {
        "intent": "greeting",
        "text": "hello joe"
    },
    {
        "intent": "greeting",
        "text": "hii"
    },
    {
        "intent": "greeting",
        "text": "whats up"
    },
    {
        "intent": "greeting",
        "text": "how are you"
    },
    {
        "intent": "greeting",
        "text": "hello"
    }
]
```



