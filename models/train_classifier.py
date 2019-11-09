import sys
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import pickle

nltk.download(['stopwords','punkt','wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    features = list(df.columns[4:])
    return X, Y, features


def tokenize(text):
    # normalize remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    # split text into words
    words = word_tokenize(text)
    
    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    return lemmed

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if(len(pos_tags)==0): 
                continue
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    return Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),

                ('starting_verb', StartingVerbExtractor())
            ])),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("category ", category_names[i], "\n\tclassification ", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]), 
              "\n\taccuracy ", accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()