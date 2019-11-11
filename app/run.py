import json
import plotly
import pandas as pd

import nltk

from flask import Flask
from flask import render_template, request, jsonify

from sklearn.externals import joblib

from figures import StartingVerbExtractor
from figures import tokenize, index_figures, results

nltk.download(['stopwords','punkt','wordnet', 'averaged_perceptron_tagger'])

app = Flask(__name__)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    graphs = index_figures()
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=results(query, model)
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()