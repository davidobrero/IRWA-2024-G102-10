from flask import Flask, render_template, request, redirect, url_for, session
import json
import math
from collections import defaultdict, Counter
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'my_secret_key'  # Required to use sessions

# In-memory data for analytics
user_queries = []  # List to store performed searches
click_data = []    # List to store click data

# Load processed tweets and build the inverted index
with open('processed_tweets.json', encoding='utf-8') as f:
    processed_tweets = json.load(f)

inverted_index = defaultdict(list)
tweet_lengths = {}
tfidf_scores = defaultdict(dict)
total_tweets = len(processed_tweets)

# Build the inverted index and preprocess data
for tweet in processed_tweets:
    doc_id = tweet["doc_id"]
    terms = tweet["tweet"]
    tweet_lengths[doc_id] = len(terms)
    
    # Build index
    term_counts = Counter(terms)
    for term, count in term_counts.items():
        inverted_index[term].append(doc_id)

    # Calculate initial TF-IDF
    for term, count in term_counts.items():
        tfidf_scores[doc_id][term] = count / len(terms)  # tf

# Ranking algorithm
def rank_tweets_tfidf(terms, tweet_ids):
    query_vector = [0] * len(terms)
    tweet_vectors = defaultdict(lambda: [0] * len(terms))
    query_counts = Counter(terms)
    query_norm = np.linalg.norm(list(query_counts.values()))

    # Calculate query vector
    for i, term in enumerate(terms):
        if term in inverted_index:
            query_vector[i] = (query_counts[term] / query_norm) * math.log(total_tweets / (1 + len(inverted_index[term])))
            for tweet_id in inverted_index[term]:
                if tweet_id in tweet_ids:
                    tweet_vectors[tweet_id][i] = tfidf_scores[tweet_id][term]
    
    # Calculate cosine similarity
    tweet_scores = []
    for tweet_id, tweet_vec in tweet_vectors.items():
        cosine_similarity = np.dot(tweet_vec, query_vector) / (tweet_lengths[tweet_id] * query_norm)
        if cosine_similarity > 0:
            tweet_scores.append((cosine_similarity, tweet_id))
    
    return [x[1] for x in sorted(tweet_scores, reverse=True, key=lambda x: x[0])]

# Main page
@app.route('/')
def search_page():
    return render_template('search.html')

# Search action
@app.route('/search', methods=['GET', 'POST'])
def search_action():
    if request.method == 'GET':
        # Use the last query stored in the session
        query = session.get('query', '')
        if not query:
            return redirect(url_for('search_page'))  # Redirect to the search page if no query in session
        terms = query.lower().split()
    else:  # POST method
        query = request.form.get('query', '').strip()
        if not query:
            return render_template('results.html', query=query, results=[])
        terms = query.lower().split()  # Simple tokenization

    user_queries.append({
        'query': query,
        'num_terms': len(terms),
        'timestamp': datetime.now().isoformat(),
        'user_agent': request.headers.get('User-Agent'),
        'ip_address': request.remote_addr
    })

    matched_tweets = set()
    for term in terms:
        if term in inverted_index:
            matched_tweets.update(inverted_index[term])

    ranked_tweets = rank_tweets_tfidf(terms, matched_tweets)

    results = []
    for tweet_id in ranked_tweets:
        tweet_data = next((tweet for tweet in processed_tweets if tweet["doc_id"] == tweet_id), None)
        if tweet_data:
            results.append({
                'title': ' '.join(tweet_data["tweet"][:3]) + '...',
                'summary': ' '.join(tweet_data["tweet"]),
                'date': tweet_data["date"],
                'url': tweet_data["url"],
                'hashtags': ', '.join(tweet_data["hashtags"]),
                'doc_id': tweet_data["doc_id"]
            })

    session['query'] = query  # Save the query in the session
    session['query_timestamp'] = datetime.now().isoformat()  # Register the time
    return render_template('results.html', query=query, results=results)

# Document details
@app.route('/doc_details')
def document_details():
    doc_id = request.args.get('id')
    query = session.get('query')
    query_timestamp = session.get('query_timestamp')

    dwell_time = None
    if query_timestamp:
        dwell_time = (datetime.now() - datetime.fromisoformat(query_timestamp)).total_seconds()

    tweet_data = next((tweet for tweet in processed_tweets if tweet["doc_id"] == doc_id), None)
    if tweet_data:
        click_data.append({
            'doc_id': doc_id,
            'query': query,
            'dwell_time': dwell_time,
            'timestamp': datetime.now().isoformat(),
            'user_agent': request.headers.get('User-Agent'),
            'ip_address': request.remote_addr
        })
        return render_template('document_details.html', tweet=tweet_data)
    return "Document not found", 404

# Dashboard analytics
@app.route('/analytics')
def analytics_dashboard():
    total_queries = len(user_queries)
    total_clicks = len(click_data)
    most_clicked_docs = Counter([click['doc_id'] for click in click_data]).most_common(5)

    return render_template('analytics.html', 
                           total_queries=total_queries,
                           total_clicks=total_clicks,
                           most_clicked_docs=most_clicked_docs,
                           user_queries=user_queries,
                           click_data=click_data)

if __name__ == '__main__':
    app.run(debug=True, port=8088)