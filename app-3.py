from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data and preprocess as in your script
a = pd.read_csv('tmdb_5000_credits.csv')
b = pd.read_csv('tmdb_5000_movies.csv')
b = b[['id', 'title', 'overview', 'genres', 'keywords']]
ab = pd.merge(b, a[['id', 'cast']])
ab.dropna(inplace=True)

def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

ab['genres'] = ab['genres'].apply(convert)
ab['keywords'] = ab['keywords'].apply(convert)

def convertr(obj):
    l = []
    t = 0
    for i in ast.literal_eval(obj):
        if t != 3:
            l.append(i['name'])
            t = t + 1
        else:
            break
    return l

ab['cast'] = ab['cast'].apply(convertr)
ab['overview'] = ab['overview'].apply(lambda x: x.split())
ab['genres'] = ab['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
ab['keywords'] = ab['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
ab['cast'] = ab['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
ab['tags'] = ab['overview'] + ab['genres'] + ab['keywords'] + ab['cast']
new = ab[['id', 'title', 'tags']]
new['tags'] = new['tags'].apply(lambda x: " ".join(x)).apply(lambda x: x.lower())

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new['tags'] = new['tags'].apply(stem)

# Text vectorization - Bag of words technique
cv = CountVectorizer(max_features=500, stop_words="english")
vectors = cv.fit_transform(new['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    mov_index = new[new['title'] == movie].index[0]
    dist = similarity[mov_index]
    movie_list = sorted(list(enumerate(dist)), reverse=True, key=lambda x: x[1])[1:6]
    return [new.iloc[i[0]].title for i in movie_list]

# Route for home page
@app.route('/', methods=['GET','POST'])
def index():
    return render_template('front.html',options=new['title'].unique().tolist())

# Route for recommendations
@app.route('/front', methods=['GET','POST'])
def get_recommendations():
    
    movie = request.form.get('Movie')
    recommendations = recommend(movie)
    movies = new['title'].values
    return render_template('front.html', movies=movies, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
