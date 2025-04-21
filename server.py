import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
from functools import lru_cache
import math
import time
import requests
import os
import pickle
import pickle

# Global cache for images
image_cache = {}
# Global cache for info
info_cache = {}


# Load data files and fix index
anime_df = pd.read_csv("anime.csv")
ratings_df = pd.read_csv("ratings.csv")
users_df = pd.read_csv("users.csv")

anime_df = anime_df.reset_index(drop=True)
ratings_df = ratings_df.reset_index(drop=True)

# Fetch the image of the given anime
def get_mal_image(mal_id):
    if mal_id in image_cache:
        return image_cache[mal_id]

    try:
        url = f"https://api.jikan.moe/v4/anime/{mal_id}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            image_url = data['data']['images']['jpg']['image_url']
            image_cache[mal_id] = image_url
            return image_url
        elif response.status_code == 429:
            print(f"Rate limited for mal_id {mal_id}, retrying...")
            time.sleep(2)
            return get_mal_image(mal_id)
    except Exception as e:
        print(f"Error fetching image for {mal_id}: {str(e)}")

    return None


# Fetch the info of the given anime
def get_mal_info(mal_id):
    if mal_id in info_cache:
        return info_cache[mal_id]

    try:
        url = f"https://api.jikan.moe/v4/anime/{mal_id}"
        response = requests.get(url)
        info = {}
        if response.status_code == 200:
            data = response.json()
            info["en_name"] = data["data"]["title_english"]
            info["jp_name"] = data["data"]["title_japanese"]
            info["date"] = data["data"]["aired"]["string"]
            info["abstract"] = data["data"]["synopsis"]
            return info
        elif response.status_code == 429:
            print(f"Rate limited for mal_id {mal_id}, retrying...")
            time.sleep(2)
            return get_mal_info(mal_id)
    except Exception as e:
        print(f"Error fetching info for {mal_id}: {str(e)}")

    return None


app = Flask(__name__, static_folder="static")
app.secret_key = '123'
CORS(app, supports_credentials=True)

# Load the local data
anime_df = pd.read_csv('anime.csv')
anime_data_df = pd.read_csv('anime_data.csv')
rating_df = pd.read_csv('ratings.csv')
users_df = pd.read_csv('users.csv')
users = {}

for i in range(len(users_df)):
    users[str(users_df["user_id"][i])] = str(users_df["password"][i])

# Fill the missing values
anime_df = anime_df.fillna({
    'name': 'Unknown',
    'genre': 'Unknown',
    'type': 'Unknown',
    'rating': 0.0
})

# Compute the average score and overall score
anime_stats = rating_df.groupby('anime_id')['rating'].agg(['mean', 'count'])
anime_stats = anime_stats.rename(columns={'mean': 'avg_rating', 'count': 'rating_count'})

anime_df = anime_df.merge(anime_stats, left_on='anime_id', right_index=True, how='left')

global_avg = anime_df['avg_rating'].mean()
min_members = 100

# Fill the missing values
anime_df['avg_rating'] = anime_df['avg_rating'].fillna(global_avg)
anime_df['rating_count'] = anime_df['rating_count'].fillna(0)
anime_df['members'] = anime_df['members'].fillna(0)

# Bayesian Average
m = min_members
C = global_avg

anime_df['weighted_score'] = (
    (anime_df['rating_count'] / (anime_df['rating_count'] + m)) * anime_df['avg_rating'] +
    (m / (anime_df['rating_count'] + m)) * C
)

# The weighted score
anime_df['final_score'] = anime_df['weighted_score']

# Load the model
#model = pickle.load(open('animeSVCBoost_3.pkl','rb'))
with open("animeSVCBoost_clean.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")


# Save the popular animes to cache
@lru_cache(maxsize=100)
def get_popular_animes(page=1, per_page=20):
    sorted_animes = anime_df.sort_values(by=['final_score'], ascending=False)
    start = (page - 1) * per_page
    end = start + per_page
    return sorted_animes.iloc[start:end].to_dict('records')


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user_id = request.form["user_id"]
        password = request.form["password"]

        # Ensure the ID is an integer
        if not user_id.isdigit():
            return "User ID must be an integer."

        user_id = user_id

        # Check whether the ID has been taken
        if user_id in users:
            return "User ID already exists."

        # Add new user
        users[user_id] = password
        user_df = pd.concat([users_df, pd.DataFrame([{"user_id": user_id, "password": password}])])
        user_df.to_csv("users.csv", index=False)

        return render_template("index.html")

    return render_template("register.html")


@app.route('/check_login', methods=['GET'])
def check_login():
    return jsonify({
        'logged_in': 'user_id' in session,
        'user_id': session.get('user_id')
    })


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON'}), 400

        user_id = str(data.get('user_id'))
        password = data.get('password', '')

        # Debug log
        print(f"Login in Session - User ID: {user_id}, Password: {password}")

        # Simplify the verification --- ALL account's password default by 123456
        if user_id in users.keys() and password == users[user_id]:
            session['user_id'] = user_id
            print(f"{user_id} login successfully!")
            return jsonify({'success': True, 'user_id': user_id})

        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    except Exception as e:
        print(f"Login Error: {str(e)}")
        return jsonify({'success': False, 'error': 'Server error'}), 500


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})


@app.route('/popular', methods=['GET'])
def popular_animes():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 20

        # Compute how to show the page indexes
        sorted_animes = anime_df.sort_values('final_score', ascending=False)
        start = (page - 1) * per_page
        end = start + per_page
        page_data = sorted_animes.iloc[start:end]

        # Attach the image link to each anime data
        animes_with_images = []
        for _, row in page_data.iterrows():
            anime_data = {
                'anime_id': row['anime_id'],
                'name': row['name'],
                'genre': row['genre'],
                'type': row['type'],
                'avg_rating': float(row['rating']),
                'members': int(row['members']),
                'image_url': get_mal_image(row['anime_id'])
            }
            animes_with_images.append(anime_data)

        return jsonify({
            'animes': animes_with_images,
            'current_page': page,
            'total_pages': math.ceil(len(anime_df) / per_page)
        })

    except Exception as e:
        print(f"Error in /popular: {str(e)}")
        return jsonify({"error": "Server error"}), 500


@app.route('/anime_count', methods=['GET'])
def get_anime_count():
    return jsonify({'count': len(anime_df)})


@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    global rating_df

    user_id = session['user_id']
    rated_ids = rating_df[rating_df['user_id'] == int(user_id)]['anime_id'].tolist()
    print(rated_ids)

    # Rated animes
    watched_features_df = anime_data_df[anime_df['anime_id'].isin(rated_ids)]
    if watched_features_df.empty:
        return jsonify({'recommendations': []})

    # Average feature of the user
    features = watched_features_df.values.astype(np.float32)
    weights = watched_features_df['rating'].values.astype(np.float32)

    user_profile = np.average(features, axis=0, weights=weights)

    # Unrated animes
    unrated_df = anime_df[~anime_df['anime_id'].isin(rated_ids)]
    unrated_features_df = anime_data_df[~anime_df['anime_id'].isin(rated_ids)]

    if unrated_features_df.empty:
        return jsonify({'recommendations': []})

    # Find the 50-nearest anime first
    features = unrated_features_df.values.astype(np.float32)
    distances = np.linalg.norm(features - user_profile, axis=1)
    closest_indices = np.argsort(distances)[:50]

    top50_features = features[closest_indices]
    top50_anime_ids = unrated_features_df.iloc[closest_indices]['anime_id'].values
    top50_df = unrated_df[unrated_df['anime_id'].isin(top50_anime_ids)]

    # Predict the anime by model
    scores = model.predict_proba(top50_features)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(top50_features)

    # Build the recommendations
    recommendations = []
    top50_df = top50_df.assign(score=scores)
    for _, row in top50_df.sort_values('score', ascending=False).head(8).iterrows():
        recommendations.append({
            'anime_id': row['anime_id'],
            'name': row['name'],
            'genre': row['genre'],
            'type': row['type'],
            'avg_rating': float(row['rating']),
            'members': int(row['members']),
            'image_url': get_mal_image(row['anime_id'])
        })

    return jsonify({'recommendations': recommendations})


@app.route('/test')
def test_route():
    return jsonify({"message": "Server worked.", "status": "success"})


@app.route('/anime/<int:anime_id>')
def anime_detail(anime_id):
    anime = anime_df[anime_data_df['anime_id'] == anime_id].to_dict(orient='records')
    anime[0]["image_url"] = get_mal_image(anime_id)
    info = get_mal_info(anime_id)
    for key in info:
        anime[0][key] = info[key]
    if not anime:
        return "Anime not found", 404
    return render_template('anime_detail.html', anime=anime[0])


@app.route('/rate', methods=['POST'])
def rate_anime():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    anime_id = data.get('anime_id')
    rating = data.get('rating')

    user_id = int(session['user_id'])
    print(f"User {user_id} rated anime {anime_id} with {rating}")

    global rating_df

    # Check whether exist
    existing_idx = rating_df[
        (rating_df['user_id'] == user_id) & (rating_df['anime_id'] == anime_id)
    ].index

    if len(existing_idx) > 0:
        rating_df.loc[existing_idx, 'rating'] = rating
        print("Rating updated.")
    else:
        new_row = pd.DataFrame([{
            'user_id': int(user_id),
            'anime_id': int(anime_id),
            'rating': rating
        }])
        rating_df = pd.concat([rating_df, new_row], ignore_index=True)
        header = not os.path.exists('ratings.csv') or os.path.getsize('ratings.csv') == 0
        new_row.to_csv('ratings.csv', mode='a', header=header, index=False)
        print("New rating added.")

    return jsonify({'message': 'Rating received!'})


#if __name__ == '__main__':
    #app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render 会传入 PORT 环境变量
    app.run(host="0.0.0.0", port=port, debug=True)
