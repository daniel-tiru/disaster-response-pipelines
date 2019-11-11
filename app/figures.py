import pandas as pd
import plotly
import plotly.graph_objs as go

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin

from sqlalchemy import create_engine

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

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

def index_figures():
    """Creates the vizualizations from the index page


    Returns:
        List of graphs

    """  
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graph_one = []
    graph_one.append(
        go.Bar(
            x = genre_names,
            y = genre_counts
        )
    )  
    layout_one = dict(title = 'Distribution of Message Genres',
                      yaxis = dict(title = 'Count'),
                      xaxis = dict(title = 'Genre')
                )
    
    category_values = df.iloc[:,4:].sum().sort_values(ascending=False).head()
    category_names = list(category_values.index)
    
    graph_two = []
    graph_two.append(
        go.Pie(
            values=category_values,
            labels=category_names
        )
    )
    layout_two = dict(title = 'Top Categories',
                      yaxis = dict(title = 'Count'),
                      xaxis = dict(title = 'Category')
                )
    
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    return graphs

def results(query, model):
    """Uses the model to predict the classification

    Args:
        query (str): the text to classify
        model (sklearn.model): the model

    Returns:
        The classification results

    """  
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    return classification_results
    
    