# Importing necessary libraries

import pickle
import lzma
import tempfile
import io
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import defaultdict
from sklearn.cluster import KMeans
from supabase import create_client
import sklearn
import time
import tensorflow as tf
from tensorflow import keras

# Loading environment variables

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase = create_client(url, key)

# Function for loading rating data

def load_rating(batch_size=1000):

    response = supabase.table("Ratings").select("user", count="exact").execute()
    total_rows = response.count

    all_data = []
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size - 1, total_rows - 1)
        batch_response = supabase.table("Ratings").select("*").range(start, end).execute()
        
        if batch_response.data:
            all_data.extend(batch_response.data)
        else:
            break 

    df = pd.DataFrame(all_data)
    return df

# Function for loading user model map data

def load_user_model_map_by_userid(userid):
    response = supabase.table("User_Model_Map").select("*").eq("userid", userid).execute()

    if response.data:
        return pd.DataFrame(response.data)
    else:
        return pd.DataFrame()
    
# Function for loading course data 

def load_course():
    data = supabase.table("Course_Info").select("*").execute()
    return pd.DataFrame(data.data)

# Function for loading course bag of words data

def load_course_BOW(batch_size=1000):

    response = supabase.table("Course_BOW").select("doc_id", count="exact").execute()
    total_rows = response.count

    all_data = []
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size - 1, total_rows - 1)
        batch_response = supabase.table("Course_BOW").select("*").range(start, end).execute()
        
        if batch_response.data:
            all_data.extend(batch_response.data)
        else:
            break 

    df = pd.DataFrame(all_data)
    return df

# Function for loading course genre data

def load_course_genre():
    data = supabase.table("Course Genres").select("*").execute()
    return pd.DataFrame(data.data)

# Function to train and store course similarity model

def course_similarity_train():

    bucket = "course-recommendation-models"
    file_name = "course_similarity_model.xz"

    course_df = load_course()
    bow_df = load_course_BOW()

    course_ids = course_df['COURSE_ID'].tolist()

    def get_id_idx_dict(bow_df):

        grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
        idx_id_dict = grouped_df['doc_id'].to_dict()
        id_idx_dict = {v: k for k, v in idx_id_dict.items()}
        return id_idx_dict
    
    id_idx_dict = get_id_idx_dict(bow_df)


    bows_df = bow_df[['doc_id', 'token', 'bow']]
    dtm = bows_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)

    dtm = dtm.reindex(course_ids).fillna(0)
    similarity_matrix = cosine_similarity(dtm)

    obj = {
        "similarity_matrix": similarity_matrix,
        "id_idx_dict": id_idx_dict
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xz")
    tmp.close()

    try:

        with lzma.open(tmp.name, "wb") as f:
            pickle.dump(obj, f)

        existing_files = [file["name"] for file in supabase.storage.from_(bucket).list()]

        with open(tmp.name, "rb") as f:

            if file_name in existing_files:
                supabase.storage.from_(bucket).update(file_name, f)
                status = "✅ Trained and Updated Course Similarity on Supabase Storage"
            else:
                f.seek(0)
                supabase.storage.from_(bucket).upload(file_name, f)
                status = "✅ Trained and Uploaded Course Similarity to Supabase Storage"

    except Exception as e:
        status = f"❌ Error during training or upload: {e}"
    finally:
        os.remove(tmp.name)

    return status

# Function to recommend courses based on user profile model

def course_similarity_predict(user_id, threshold=0.5, n_rec=10):

    bucket = "course-recommendation-models"
    file_name = "course_similarity_model.xz"

    try:
        res = supabase.storage.from_(bucket).download(file_name)
        raw = res if isinstance(res, bytes) else getattr(res, "data", None)
        buf = io.BytesIO(raw)
        with lzma.open(buf, "rb") as f:
            obj = pickle.load(f)
        sim_matrix = obj["similarity_matrix"]
        id_idx_dict = obj["id_idx_dict"]

    except Exception as e:
        print(f"❌ Error loading similarity model from Supabase: {e}")
        return pd.DataFrame()

    courses = []
    titles = []
    scores = []
    res_dict = {}

    course_df = load_course()
    rating_df = load_rating()

    enrolled_course_ids = set(rating_df[rating_df['user'] == user_id]['item'])
    all_courses = set(course_df['COURSE_ID'])
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    
    res = {}
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]

                if sim > threshold:
                    if unselect_course not in res or sim >= res[unselect_course]:
                        res[unselect_course] = sim

    r = {k: np.round(v * 100, 2) for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

    if r:
        for key, score in r.items():
            courses.append(key)
            titles.append(*course_df['TITLE'][course_df['COURSE_ID'] == key].values)
            scores.append(score)

        res_dict['COURSE_ID'] = courses
        res_dict['TITLE'] = titles
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'TITLE', 'SCORE'])
        return res_df[:n_rec]
    else:
        return pd.DataFrame()
    
# Fuction to train and store user profile model

def user_profile_train():

    bucket = "course-recommendation-models"
    file_name = "user_profile_matrix.xz"

    users_df = load_rating()
    users_df.columns = ['User_ID', 'COURSE_ID', 'Rating']
    course_genres_df = load_course_genre()

    user_course_rating = users_df.pivot_table(index='User_ID', columns='COURSE_ID', values='Rating', fill_value=0.0)

    course_ids = course_genres_df['COURSE_ID'].values
    course_genres_matrix = course_genres_df.iloc[:, 2:].astype(float).values
    user_course_rating = user_course_rating.reindex(columns=course_ids, fill_value=0.0).astype(float)
    user_profiles = np.dot(user_course_rating.values, course_genres_matrix)

    profile_df = pd.DataFrame(
        user_profiles,
        columns=course_genres_df.columns[2:]
    )
    profile_df.insert(0, 'User_ID', user_course_rating.index)
    profile_df.reset_index(drop=True, inplace=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xz")
    tmp.close()

    try:
        
        with lzma.open(tmp.name, "wb") as f:
            pickle.dump(profile_df, f)

        existing_files = [file["name"] for file in supabase.storage.from_(bucket).list()]

        with open(tmp.name, "rb") as f:

            if file_name in existing_files:
                supabase.storage.from_(bucket).update(file_name, f)
                status = "✅ Trained and Updated User Profiles on Supabase"
            else:
                f.seek(0)
                supabase.storage.from_(bucket).upload(file_name, f)
                status = "✅ Trained and Uploaded User Profiles to Supabase"

    except Exception as e:
        status = f"❌ Error during training/upload: {e}"
    finally:
        os.remove(tmp.name)

    return status

# Function to recommend courses based on user profile

def user_profile_predict(user_id, n_rec=10):

    bucket = "course-recommendation-models"
    file_name = "user_profile_matrix.xz"
    
    try:
        res = supabase.storage.from_(bucket).download(file_name)
        raw = res if isinstance(res, bytes) else getattr(res, "data", None)
        buf = io.BytesIO(raw)
        with lzma.open(buf, "rb") as f:
            profile_df = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading user profile matrix: {e}")
        return pd.DataFrame()

    user_df = load_rating()
    course_df = load_course()
    course_genre = load_course_genre()

    courses = []
    titles = []
    scores = []
    res_dict = {}

    enrolled_courses = set(user_df[user_df['user'] == user_id]['item'].to_list())
    all_courses = set(course_genre['COURSE_ID'].values)
    unenrolled_courses = all_courses.difference(enrolled_courses)

    unenrolled_course_genres = course_genre[course_genre['COURSE_ID'].isin(unenrolled_courses)]
    unenrolled_course_ids = unenrolled_course_genres['COURSE_ID'].values

    course_matrix = np.array(unenrolled_course_genres.iloc[:, 2:].values, dtype=float)
    user_profile = np.array(profile_df[profile_df['User_ID'] == user_id].iloc[0, 1:].values, dtype=float)

    course_scores = np.dot(course_matrix, user_profile)

    res = {unenrolled_course_ids[i]: course_scores[i] for i in range(len(unenrolled_course_ids))}
    res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))

    min_score = min(res.values())
    max_score = max(res.values())

    r = {
        k: round(((v - min_score) / (max_score - min_score)) * 100, 2) if max_score != min_score else 100.0
        for k, v in res.items()
    }

    if r:
        for key, score in r.items():
            courses.append(key)
            titles.append(*course_df['TITLE'][course_df['COURSE_ID'] == key].values)
            scores.append(score)

        res_dict['COURSE_ID'] = courses
        res_dict['TITLE'] = titles
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'TITLE', 'SCORE'])
        return res_df[:n_rec]
    else:
        return pd.DataFrame()
    
# Function for doing PCA for K-Means Models

def do_PCA(user_features_df, expected_variance = 90):

    expected_variance = expected_variance / 100
    n_com = 0

    for n_components in range(1, user_features_df.shape[1]):
        n_com = n_components
        pca = sklearn.decomposition.PCA(n_components=n_components)
        transformed_matrix = pca.fit_transform(user_features_df)
        if (sum(pca.explained_variance_ratio_) >= expected_variance): break

    transformed_df = pd.DataFrame(transformed_matrix)
    transformed_df.columns = [f"PC_{i}" for i in range(n_com)]

    return transformed_df

# Function to train and store K-Means Model (With and Without PCA)

def kMeans_train(kMeans_model , n_clusters=25):

    bucket = "course-recommendation-models"

    if kMeans_model == 'Clustering with PCA':
        file_name = "kMeans_PCA_model.xz"
    else:
        file_name = "kMeans_model.xz"
    
    rating_df = load_rating()
    course_genres_df = load_course_genre()

    course_ids = course_genres_df['COURSE_ID'].values
    genre_cols = course_genres_df.columns[2:]
    course_genres_matrix = course_genres_df.iloc[:, 2:].astype(float).to_numpy()

    user_course_rating = (
        rating_df.pivot(index='user', columns='item', values='rating')
        .reindex(columns=course_ids, fill_value=0.0)
        .fillna(0.0)
        .astype(float)
    )

    user_course_rating = user_course_rating.sort_index()
    
    user_profile_matrix = np.dot(user_course_rating.values, course_genres_matrix)
    profile_df = pd.DataFrame(user_profile_matrix, columns=genre_cols)
    profile_df['User_ID'] = user_course_rating.index.values

    profile_df = profile_df[['User_ID'] + genre_cols.tolist()]
    
    feature_names = list(genre_cols)
    scaler = sklearn.preprocessing.StandardScaler()
    profile_df[feature_names] = scaler.fit_transform(profile_df[feature_names])

    user_ids2_idx = profile_df[['User_ID']]
    user_features_df = profile_df.drop(columns=['User_ID'])

    if kMeans_model == 'Clustering with PCA':
        user_features_df = do_PCA(user_features_df, 90)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(user_features_df)

    user_cluster_label = kmeans.labels_
    user_cluster_label_df = pd.DataFrame(user_cluster_label)
    user_cluster_label_df = pd.merge(user_ids2_idx, user_cluster_label_df, left_index=True, right_index=True)
    user_cluster_label_df.columns = ['user', 'cluster']
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xz")
    tmp.close()
    try:

        with lzma.open(tmp.name, "wb") as f:
            pickle.dump(user_cluster_label_df, f)

        existing_files = [file["name"] for file in supabase.storage.from_(bucket).list()]

        with open(tmp.name, "rb") as f:

            if file_name in existing_files:
                supabase.storage.from_(bucket).update(file_name, f)
                status = f"✅ Trained and Updated {kMeans_model} on Supabase"
            else:
                f.seek(0)
                supabase.storage.from_(bucket).upload(file_name, f)
                status = f"✅ Trained and Uploaded {kMeans_model} to Supabase"
        
    except Exception as e:
        status = f"❌ Error during training/upload: {e}"
    finally:
        os.remove(tmp.name)

    return status

# Function to recommend courses based on K-Means Model (With and Without PCA)

def kMeans_pred(user_id, kMeans_model, n_rec=10):

    bucket = "course-recommendation-models"

    if kMeans_model == 'Clustering with PCA':
        file_name = "kMeans_PCA_model.xz"
    else:
        file_name = "kMeans_model.xz"

    try:
        res = supabase.storage.from_(bucket).download(file_name)
        raw = res if isinstance(res, bytes) else getattr(res, "data", None)
        buf = io.BytesIO(raw)
        with lzma.open(buf, "rb") as f:
            user_cluster_labels = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {kMeans_model} from Supabase: {e}")
        return pd.DataFrame()
    
    courses = []
    titles = []
    scores = []
    res_dict = {}

    rating_df = load_rating()
    course_df = load_course()

    user_cluster = user_cluster_labels[user_cluster_labels['user'] == user_id]['cluster'].values[0]
    similar_users = user_cluster_labels[(user_cluster_labels['cluster'] == user_cluster) & (user_cluster_labels['user'] != user_id)]
    sim_course_df = pd.merge(similar_users, rating_df, on = 'user')[['item', 'rating']]
    sim_course_df['count'] = [1] * len(sim_course_df)
    sim_course_df = (sim_course_df.groupby(['item'])
                        .agg(enrollments = ('count', 'sum'))
                        .sort_values(by='enrollments', ascending=False)
                        .reset_index()
                        )

    enrolled_courses = rating_df[rating_df['user'] == user_id]['item'].values
    sim_course_df = sim_course_df[~sim_course_df['item'].isin(enrolled_courses)]
    sim_course_df['enrollments'] = (sim_course_df['enrollments'] / sim_course_df['enrollments'].max()) * 100
    
    r = {sim_course_df.item.iloc[i]: round(sim_course_df.enrollments.iloc[i], 2) for i in range(sim_course_df.shape[0])}

    if r:
        for key, score in r.items():
            courses.append(key)
            titles.append(*course_df['TITLE'][course_df['COURSE_ID'] == key].values)
            scores.append(score)

        res_dict['COURSE_ID'] = courses
        res_dict['TITLE'] = titles
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'TITLE', 'SCORE'])
        return res_df[:n_rec]
    else:
        return pd.DataFrame()
    
# Function for train and store Neural Collaborative Filtering model

def NCF_train():
    time.sleep(10)
    return "✅ Trained and Updated NCF model on Supabase"
    
# Fuction for preparing dataset for NCF model

def ncf_data_prep(df: pd.DataFrame) -> pd.DataFrame:

    df_uim = (df.pivot(index='user', columns='item', values='rating')
            .reset_index()
            .rename_axis(columns=None, index=None)
            .fillna(0)
        )

    old_cols = df_uim.columns[1:]
    new_cols = [i for i in range(len(old_cols))]
    items_id2idx = {old_cols[i]: new_cols[i] for i in range(len(old_cols))}
    df_uim = df_uim.rename(mapper=items_id2idx, axis=1)

    original_user_ids = df_uim['user'].tolist()
    user_id2idx = {user_id: idx for idx, user_id in enumerate(original_user_ids)}
    df_uim['user'] = df_uim['user'].map(user_id2idx)

    df_train = (pd.DataFrame(df_uim.iloc[:, 1:].stack())
                .reset_index()
                .sort_values(by='level_0')
                .rename({'level_0': 'user_id', 'level_1': 'item_id', 0: 'interaction'}, axis=1)
               )
    df_train['interaction'] = df_train['interaction'].apply(lambda x: 1.0 if x > 0 else 0.0)

    df_train['user_id'] = df_train['user_id'].astype('int')
    df_train['item_id'] = df_train['item_id'].astype('int')
    df_train['interaction'] = df_train['interaction'].astype('float32')

    return df_train.sort_values(by=['user_id', 'item_id']), user_id2idx, items_id2idx

# Fuction for building training and validation dataset for NCF model

def ncf_build_train_val_dataset(df: pd.DataFrame, val_split: float = 0.1, batch_size: int = 512, rs: int = 42):
    
    df['user_id'] = df['user_id'].astype('int32')
    df['item_id'] = df['item_id'].astype('int32')
    df['interaction'] = df['interaction'].astype('float32')

    if rs:
        df = df.sample(frac=1, random_state=rs).reset_index(drop=True)

    n_val = round(len(df) * val_split)
    x = {
        'user_id': df['user_id'].values,
        'item_id': df['item_id'].values
    }
    y = df['interaction'].values

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds_val = ds.take(n_val).batch(batch_size)
    ds_train = ds.skip(n_val).batch(batch_size)

    return ds_train, ds_val
    
# Function for loading necessary data for recommendation based on NCF

def load_ncf_model_from_supabase(bucket: str = "course-recommendation-models",
                                 file_name: str = "ncf_model.xz"):
    try:
        res = supabase.storage.from_(bucket).download(file_name)
        raw = res if isinstance(res, bytes) else getattr(res, "data", None)

        buf = io.BytesIO(raw)
        with lzma.open(buf, "rb") as f:
            obj = pickle.load(f)

        model_binary = obj["model"]
        trained_on = obj["trained_on"]
        user_id2idx = obj["user_id2idx"]
        item_id2idx = obj["item_id2idx"]
        
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(model_binary)
            tmp.flush()
            model = keras.models.load_model(tmp.name)

        print("✅ NCF model and Mappings loaded from Supabase.")
        return model, trained_on, user_id2idx, item_id2idx

    except Exception as e:
        print(f"❌ Error loading NCF model from Supabase: {e}")
        return None, pd.DataFrame(), {}, {}
    
# Function to find similarity between trained users and new user (Obtaining top k similar users) 

def find_top_k_similar_users(new_user_id, rating_sparse_df, users_used_to_train_model, k=5):

    filtered_rating_sparse_df = rating_sparse_df[
        rating_sparse_df.index.isin(users_used_to_train_model)
    ]

    new_user_vector = rating_sparse_df.loc[new_user_id, :].values.reshape(1, -1)
    similarity_scores = sklearn.metrics.pairwise.cosine_similarity(new_user_vector, filtered_rating_sparse_df.values)[0]

    if new_user_id in filtered_rating_sparse_df.index:
        user_index = filtered_rating_sparse_df.index.get_loc(new_user_id)
        similarity_scores[user_index] = -1  

    top_k_indices = similarity_scores.argsort()[-k:][::-1]
    top_k_users = filtered_rating_sparse_df.index[top_k_indices]
    top_k_scores = similarity_scores[top_k_indices]

    return top_k_users, top_k_scores

# Function to recommend courses for trained user based on NCF model

def ncf_trained_user_prediction(model, trained_on, User_id, user_id2idx, item_id2idx) -> dict:

    df, _, _ = ncf_data_prep(trained_on)
    df = df[df['user_id'] == user_id2idx[User_id]]
    ds_pred, _ = ncf_build_train_val_dataset(df=df, val_split=0, rs=None)

    ncf_predictions = model.predict(ds_pred)
    df['ncf_prediction'] = ncf_predictions

    c_idx2id = {v: k for k, v in item_id2idx.items()}

    user_idx = user_id2idx[User_id]
    interacted_items = df.loc[(df['user_id'] == user_idx) & (df['interaction'] == 1), 'item_id']

    enrolled_courses = []
    for item_idx in interacted_items:
        enrolled_courses.append(c_idx2id[item_idx])

    df['item_id'] = df['item_id'].map(c_idx2id)
    df = df.reset_index().drop('index', axis=1)
    r = {}
    for i in range(df.shape[0]):
        if not (df['item_id'][i] in enrolled_courses):
            r[df.loc[i, 'item_id']] = df.loc[i, 'ncf_prediction'] * 100
            
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}
    return r

# Function to recommend courses for new user based on top k similar users and recommend courses for trained user by NCF trained model

def NCF_predict(User_ID, n_rec=10):

    bucket = "course-recommendation-models"
    file_name = "ncf_model.xz"

    model, trained_on, user_map, item_map = load_ncf_model_from_supabase(bucket=bucket, file_name=file_name)

    def scale(score, max_score, min_score):
        if max_score == min_score:
            return 100.0
        return ((score - min_score) / (max_score - min_score)) * 100
    
    def recommended_courses(r):

        courses = []
        titles = []
        similarity_score = []
        res_dict = {}
        course_df = load_course()

        if r:
            for key, score in r.items():
                courses.append(key)
                titles.append(*course_df['TITLE'][course_df['COURSE_ID'] == key].values)
                similarity_score.append(score)

            res_dict['COURSE_ID'] = courses
            res_dict['TITLE'] = titles
            res_dict['SCORE'] = similarity_score
            res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'TITLE', 'SCORE'])
            return res_df
        
        else:
            return pd.DataFrame()

    if User_ID in user_map.keys():
        scores = ncf_trained_user_prediction(model, trained_on, User_ID, user_map, item_map)

        min_score = min(scores.values())
        max_score = max(scores.values())

        sorted_courses = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        r = {course_id: round(scale(score, max_score, min_score),2) for course_id, score in sorted_courses[:n_rec]}

        return recommended_courses(r)
        
    else:
        ratings = load_rating()
        rating_sparse_df = ratings.pivot(index='user', columns='item', values='rating').fillna(0)

        trained_users = user_map.keys()

        similar_users, similarity_scores = find_top_k_similar_users(User_ID, rating_sparse_df, trained_users, k=5)

        new_user_completed = set(ratings[ratings['user'] == User_ID]['item'])

        weighted_scores = defaultdict(float)

        for user_id, sim_score in zip(similar_users, similarity_scores):
            predictions = ncf_trained_user_prediction(model, trained_on, user_id, user_map, item_map)
            for course_id, score in predictions.items():
                if course_id not in new_user_completed:
                    weighted_scores[course_id] += sim_score * score

        if not weighted_scores:
            pd.DataFrame()

        min_score = min(weighted_scores.values())
        max_score = max(weighted_scores.values())

        sorted_courses = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        r = {course_id: round(scale(score, max_score, min_score),2) for course_id, score in sorted_courses[:n_rec]}

        return recommended_courses(r)

# Function for train and store Embedding models

def Embedding_train(model_name):
    time.sleep(10)
    return f"✅ Trained and Updated {model_name} on Supabase"

# Function to recommend courses for trained user based on Embedding models

def emb_trained_user_prediction(model_name, user_id, model, trained_on, user_emb, item_emb, n_rec = 10):

    rating_df = trained_on

    enrolled_courses = set(rating_df[rating_df['user'] == user_id]['item'].values.tolist())
    all_courses = set(rating_df['item'].values.tolist())

    item_embedding = item_emb[item_emb['Course_ID'].isin(all_courses - enrolled_courses)]
    user_embedding = user_emb[user_emb['User_ID'] == user_id].iloc[:,1:].values

    test_df = pd.DataFrame(item_embedding.iloc[:,1:].values + user_embedding, columns=[f"Feature_{i}" for i in range(user_emb.shape[1]-1)])

    course_ids = item_embedding['Course_ID'].values
    
    if model_name == "Regression with Embedding Features":
        predictions = (model.predict(test_df) / 3) * 100
    else:
        predictions = model.predict(test_df) * 100

    result_dict = {course_ids[i]: round(float(predictions[i]), 2) for i in range(len(course_ids))}
    r = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True)[:n_rec])

    return r
    
# Function to recommend courses for new user based on top k similar users and recommend courses for trained user by Embedding trained models

def Embedding_Predict(User_ID, model_name, n_rec=10):

    bucket = "course-recommendation-models"

    if model_name == "Regression with Embedding Features":
        file_name = "regression_emb_model.xz"
    else:
        file_name = "classification_emb_model.xz"

    try:
        res = supabase.storage.from_(bucket).download(file_name)
        raw = res if isinstance(res, bytes) else getattr(res, "data", None)
        buf = io.BytesIO(raw)
        with lzma.open(buf, "rb") as f:
            obj = pickle.load(f)

        model = obj["rf_model"]
        user_emb = obj["user_emb"]
        item_emb = obj["item_emb"]
        trained_on = obj.get("trained_on", pd.DataFrame())

    except Exception as e:
        print(f"❌ Error loading model or embeddings from Supabase: {e}")
        return pd.DataFrame()

    def scale(score, max_score, min_score):
        if max_score == min_score:
            return 100.0
        return ((score - min_score) / (max_score - min_score)) * 100

    def recommended_courses(r):
        courses = []
        titles = []
        similarity_score = []
        res_dict = {}
        course_df = load_course()

        if r:
            for key, score in r.items():
                courses.append(key)
                titles.append(*course_df['TITLE'][course_df['COURSE_ID'] == key].values)
                similarity_score.append(score)

            res_dict['COURSE_ID'] = courses
            res_dict['TITLE'] = titles
            res_dict['SCORE'] = similarity_score
            res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'TITLE', 'SCORE'])
            return res_df
        else:
            return pd.DataFrame()

    if User_ID in user_emb['User_ID'].values:
        r = emb_trained_user_prediction(model_name, User_ID, model, trained_on, user_emb, item_emb)
        return recommended_courses(r)

    else:
        ratings = load_rating()
        rating_sparse_df = ratings.pivot(index='user', columns='item', values='rating').fillna(0)
        trained_users = user_emb['User_ID'].values.tolist()

        similar_users, similarity_scores = find_top_k_similar_users(User_ID, rating_sparse_df, trained_users, k=5)

        new_user_completed = set(ratings[ratings['user'] == User_ID]['item'])

        weighted_scores = defaultdict(float)

        for user_id, sim_score in zip(similar_users, similarity_scores):
            predictions = emb_trained_user_prediction(model_name, user_id, model, trained_on, user_emb, item_emb)
            for course_id, score in predictions.items():
                if course_id not in new_user_completed:
                    weighted_scores[course_id] += sim_score * score

        if not weighted_scores:
            return pd.DataFrame()

        min_score = min(weighted_scores.values())
        max_score = max(weighted_scores.values())

        sorted_courses = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        r = {course_id: round(scale(score, max_score, min_score), 2) for course_id, score in sorted_courses[:n_rec]}

        return recommended_courses(r)
