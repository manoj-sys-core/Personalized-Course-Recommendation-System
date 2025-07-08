import pandas as pd
import os
import io
import lzma
import pickle
import tempfile
from supabase import create_client
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow import keras
import sklearn

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase = create_client(url, key)


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

# ======= Neural Collaborative Filtering =======

def ncf_create(n_users: int, n_items: int,                           
               latent_dim_mf: int = 32, latent_dim_mlp: int = 32,   
               reg_mf: int = 0, reg_mlp: int = 0.001,                
               dense_layers: list = [16, 8, 4],                       
               reg_layers: list = [0.01, 0.01, 0.01],                    
               activation_dense: str = 'relu'                     
) -> keras.Model:

    user = keras.Input(shape=(), dtype='int32', name='user_id')
    item = keras.Input(shape=(), dtype='int32', name='item_id')

    mf_user_embedding = keras.layers.Embedding(input_dim = n_users,
                                  output_dim = latent_dim_mf,
                                  name = 'mf_user_embedding',
                                  embeddings_initializer = 'RandomNormal',
                                  embeddings_regularizer = keras.regularizers.l2(reg_mf)
                                 )
    
    mf_item_embedding = keras.layers.Embedding(input_dim = n_items,
                                  output_dim = latent_dim_mf,
                                  name = 'mf_item_embedding',
                                  embeddings_initializer = 'RandomNormal',
                                  embeddings_regularizer = keras.regularizers.l2(reg_mf)
                                 )

    mlp_user_embedding = keras.layers.Embedding(input_dim = n_users,
                                   output_dim = latent_dim_mlp,
                                   name = 'mlp_user_embedding',
                                   embeddings_initializer = 'RandomNormal',
                                   embeddings_regularizer = keras.regularizers.l2(reg_mlp)
                                  )
    mlp_item_embedding = keras.layers.Embedding(input_dim = n_items,
                                  output_dim = latent_dim_mlp,
                                  name = 'mlp_item_embedding',
                                  embeddings_initializer = 'RandomNormal',
                                  embeddings_regularizer = keras.regularizers.l2(reg_mlp)
                                 )

    mf_user_latent = keras.layers.Flatten()(mf_user_embedding(user))
    mf_item_latent = keras.layers.Flatten()(mf_item_embedding(item))

    mlp_user_latent = keras.layers.Flatten()(mlp_user_embedding(user))
    mlp_item_latent = keras.layers.Flatten()(mlp_item_embedding(item))

    mf_cat_latent = keras.layers.Multiply()([mf_user_latent, mf_item_latent])
    mlp_cat_latent = keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent
    for i in range(len(dense_layers)):
        layer = keras.layers.Dense(
                      units = dense_layers[i],
                      activation = activation_dense,
                      activity_regularizer = keras.regularizers.l2(reg_layers[i]),
                      name = 'layer%d' % i,
                     )
        mlp_vector = layer(mlp_vector)
    
    predict_layer = keras.layers.Concatenate()([mf_cat_latent, mlp_vector])
    result = keras.layers.Dense(
                   units = 1, 
                   activation = 'sigmoid',
                   kernel_initializer = 'lecun_uniform',
                   name = 'interaction' 
                  )

    output = result(predict_layer)

    model = keras.Model(inputs = [user, item],
                  outputs = [output]
                 )

    return model


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


def ncf_train_model(ds_train, ds_val, n_epochs: int = 10):

    n_users, n_items = (load_rating()
                        .pivot(index='user', columns='item', values='rating')
                        .reset_index()
                        .rename_axis(index=None, columns=None)
                        .shape)
    
    ncf_model = ncf_create(n_users=n_users, n_items=n_items)
    ncf_model.compile(optimizer = "adam",
                    loss = 'binary_crossentropy',
                    metrics = [
                                tf.keras.metrics.TruePositives(name="tp"),
                                tf.keras.metrics.FalsePositives(name="fp"),
                                tf.keras.metrics.TrueNegatives(name="tn"),
                                tf.keras.metrics.FalseNegatives(name="fn"),
                                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                                tf.keras.metrics.Precision(name="precision"),
                                tf.keras.metrics.Recall(name="recall"),
                                tf.keras.metrics.AUC(name="auc"),
                                ]
                    )

    ncf_model._name = 'neural_collaborative_filtering'
    ncf_hist = ncf_model.fit(x=ds_train, 
                             validation_data=ds_val,
                             epochs=n_epochs,
                             verbose=1
                            )
    return ncf_model, ncf_hist


def NCF_train():

    df = load_rating()
    df_train, user_id2idx, item_id2idx = ncf_data_prep(df)
    ds_train, ds_val = ncf_build_train_val_dataset(df=df_train, val_split=0.1, rs=42)

    model, _ = ncf_train_model(ds_train=ds_train, ds_val=ds_val, n_epochs=10)

    return model, df, user_id2idx, item_id2idx


def emb_create(n_user, n_item, n_user_latent_dim: int = 16, n_item_latent_dim: int = 16, reg_users: int = 1e-6, reg_items: int = 1e-6) -> keras.Model:

    user_input = keras.layers.Input(shape=(), dtype='int32', name='user')
    item_input = keras.layers.Input(shape=(), dtype='int32', name='item')

    # USER
    user_embedding = keras.layers.Embedding(input_dim=n_user,
                    output_dim=n_user_latent_dim,
                    name='user_embedding',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(reg_users)
                    )(user_input)
    
    user_vec = keras.layers.Flatten(name='user_flat')(user_embedding)

    user_bias = keras.layers.Embedding(input_dim=n_user,
                    output_dim=1,
                    name='user_bias',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(reg_users)
                    )(user_input)
    
    user_model = keras.models.Model(inputs=user_input, outputs=user_vec)

    # ITEM
    item_embedding = keras.layers.Embedding(input_dim=n_item,
                    output_dim=n_item_latent_dim,
                    name='item_embedding',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(reg_items)
                    )(item_input)

    item_bias = keras.layers.Embedding(input_dim=n_user,
                    output_dim=1,
                    name='item_bias',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(reg_users)
                    )(item_input)

    merged = keras.layers.Dot(name='dot', normalize=True, axes=1)([user_embedding, item_embedding])
    merged_dropout = keras.layers.Dropout(0.2)(merged)

    #hidden layers
    dense_1 = keras.layers.Dense(units=64, name='Dense_1')(merged_dropout)
    do_1 = keras.layers.Dropout(0.2, name='Dropout_1')(dense_1)

    dense_2 = keras.layers.Dense(units=32, name='Dense_2')(do_1)
    do_2 = keras.layers.Dropout(0.2, name='Dropout_2')(dense_2)

    dense_3 = keras.layers.Dense(units=16, name='Dense_3')(do_2)
    do_3 = keras.layers.Dropout(0.2, name='Dropout_3')(dense_3)

    dense_4 = keras.layers.Dense(units=8, name='Dense_4')(do_3)

    result = keras.layers.Dense(1, name='rating', activation='relu')(dense_4)

    model = keras.models.Model(inputs=[user_input, item_input], outputs=[result])
    model._name = 'embedding_extraction_model'
    
    return model


def emb_data_prep(raw_data):
    
    encoded_data = raw_data.copy()

    user_id2idx = {x: i for i, x in enumerate(encoded_data["user"].unique())}
    item_id2idx = {x: i for i, x in enumerate(encoded_data["item"].unique())}

    encoded_data['user'] = encoded_data['user'].map(user_id2idx)
    encoded_data['item'] = encoded_data['item'].map(item_id2idx)

    return encoded_data, user_id2idx, item_id2idx


def emb_ds_create(df, val_split = 0.1, batch_size = 512, rs = 42):
    
    if rs: 
        df = df.sample(frac=1, random_state=rs).reset_index(drop=True)

    n_val = round(len(df) * val_split)

    x = {
        'user': df['user'].values,
        'item': df['item'].values
    }
    y = df['rating'].values

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds_val = ds.take(n_val).batch(batch_size)
    ds_train = ds.skip(n_val).batch(batch_size)
    
    return ds_train, ds_val


def emb_train(ds_train, ds_val, num_users, num_items, epochs = 10, embedding_size = 16):

    emb_model = emb_create(n_user=num_users, n_item=num_items, n_item_latent_dim=embedding_size, n_user_latent_dim=embedding_size)
    emb_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(), metrics=[tf.keras.metrics.MeanSquaredError()])

    emb_model_hist = emb_model.fit(x = ds_train, validation_data = ds_val, epochs=epochs, verbose=1)

    return emb_model, emb_model_hist


def Embedding_train(model_name):

    ratings_df = load_rating()

    num_users = len(ratings_df['user'].unique())
    num_items = len(ratings_df['item'].unique())

    encoded_data, user_id2idx, course_id2idx = emb_data_prep(ratings_df)
    ds_train, ds_val = emb_ds_create(encoded_data)

    emb, _ = emb_train(ds_train=ds_train, ds_val=ds_val, num_users=num_users, num_items=num_items)

    user_latent_features = emb.get_layer('user_embedding').get_weights()[0]
    item_latent_features = emb.get_layer('item_embedding').get_weights()[0]

    user_emb = pd.DataFrame(user_latent_features, columns=[f'User_Feature_{i}' for i in range(user_latent_features.shape[1])])
    user_emb.insert(0, 'User_ID', list(user_id2idx.keys()))

    item_emb = pd.DataFrame(item_latent_features, columns= [f'Course_Feature_{i}' for i in range(item_latent_features.shape[1])])
    item_emb.insert(0, 'Course_ID', list(course_id2idx.keys()))

    train_df = ratings_df.copy()

    user_emb_train_merged = pd.merge(train_df, user_emb, how='left', left_on='user', right_on='User_ID').fillna(0)
    merged_train_df = pd.merge(user_emb_train_merged, item_emb, how='left', left_on='item', right_on='Course_ID').fillna(0)

    u_features = [f"User_Feature_{i}" for i in range(user_emb.shape[1] - 1)]
    c_features = [f"Course_Feature_{i}" for i in range(item_emb.shape[1] - 1)]

    user_train_embeddings = merged_train_df[u_features]
    course_train_embeddings = merged_train_df[c_features]
    ratings_train = merged_train_df['rating']

    x_train = user_train_embeddings + course_train_embeddings.values
    x_train.columns = [f"Feature_{i}" for i in range(item_emb.shape[1] - 1)]
    y_train = ratings_train

    if model_name == "Regression with Embedding Features":
        rf_model = sklearn.ensemble.RandomForestRegressor(
            n_estimators=100,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='log2',
            max_depth=20,
            random_state=42
        )
    else:
        label_encoder = sklearn.preprocessing.LabelEncoder()
        y_train = label_encoder.fit_transform(y_train.values.ravel())
        rf_model = sklearn.ensemble.RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            max_features='sqrt',
            min_samples_split=2,
            random_state=42
        )

    rf_model.fit(x_train, y_train)

    return rf_model, ratings_df, user_emb, item_emb


def check_file_exists(bucket: str, file_path: str) -> bool:
    
    try:
        if "/" in file_path:
            folder, file_name = file_path.rsplit("/", 1)
        else:
            folder, file_name = "", file_path
            
        file_list = supabase.storage.from_(bucket).list(path=folder)

        if not isinstance(file_list, list):
            return False

        file_names = [file['name'] for file in file_list]
        return file_name in file_names

    except Exception as e:
        return False
    
    
def load_model_metadata_from_supabase(bucket, file_name):

    if file_name == "ncf_model.xz":
        try:
            res = supabase.storage.from_(bucket).download(file_name)
            raw = res if isinstance(res, bytes) else getattr(res, "data", None)
            buf = io.BytesIO(raw)
            with lzma.open(buf, "rb") as f:
                obj = pickle.load(f)

            return obj.get("trained_on", pd.DataFrame())

        except Exception:
            return pd.DataFrame()
    else:
        try:
            res = supabase.storage.from_(bucket).download(file_name)
            raw = res if isinstance(res, bytes) else getattr(res, "data", None)
            buf = io.BytesIO(raw)
            with lzma.open(buf, "rb") as f:
                obj = pickle.load(f)
            return obj.get("trained_on", pd.DataFrame())
        
        except Exception:
            return pd.DataFrame()
        
def upload_NCF_model_to_supabase(model, df, user_id2idx, item_id2idx):

    bucket = "course-recommendation-models"
    file_name = "ncf_model.xz"

    status = ""

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_model:
        model.save(tmp_model.name) 
        model_path = tmp_model.name

    with open(model_path, "rb") as f_model:
        model_binary = f_model.read()

    obj = {
        "model": model_binary,
        "trained_on": df,
        "user_id2idx": user_id2idx,
        "item_id2idx": item_id2idx
    }

    tmp_bundle = tempfile.NamedTemporaryFile(delete=False, suffix=".xz")
    tmp_bundle.close()

    try:
        
        with lzma.open(tmp_bundle.name, "wb") as f:
            pickle.dump(obj, f)

        existing_files = [file["name"] for file in supabase.storage.from_(bucket).list()]

        with open(tmp_bundle.name, "rb") as f:

            if file_name in existing_files:
                supabase.storage.from_(bucket).update(file_name, f)
                status = f"✅ Trained and Updated NCF model to `{file_name}`"
            else:
                f.seek(0)
                supabase.storage.from_(bucket).upload(file_name, f)
                status = f"✅ Trained and Uploaded NCF model to `{file_name}`"

    except Exception as e:
        status = f"❌ Error during saving or upload: {e}"
    finally:
        os.remove(model_path)
        os.remove(tmp_bundle.name)

    return status


def upload_Embedding_model_to_supabase(model_name, rf_model, df, user_emb, item_emb):

    bucket = "course-recommendation-models"

    if model_name == "Regression with Embedding Features":
        file_name = "regression_emb_model.xz"
    else:
        file_name = "classification_emb_model.xz"

    status = ""

    obj = {
        "rf_model": rf_model,
        "trained_on": df,
        "user_emb": user_emb,
        "item_emb": item_emb
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
                status = f"✅ Trained and Updated {model_name} on Supabase"
            else:
                f.seek(0)
                supabase.storage.from_(bucket).upload(file_name, f)
                status = f"✅ Trained and Uploaded {model_name} to Supabase"

    except Exception as e:
        status = f"❌ Error during training/upload: {e}"
    finally:
        os.remove(tmp.name)

    return status


def retrain_models():

    bucket = "course-recommendation-models"
    file_names = ["ncf_model.xz", "regression_emb_model.xz", "classification_emb_model.xz"]

    def train_and_upload(file_name):

        if file_name == "ncf_model.xz":
            model, df, user_id2idx, item_id2idx = NCF_train()
            return upload_NCF_model_to_supabase(model, df, user_id2idx, item_id2idx)
        elif file_name == "regression_emb_model.xz":
            model, df, user_emb, item_emb = Embedding_train("Regression with Embedding Features")
            return upload_Embedding_model_to_supabase("Regression with Embedding Features", model, df, user_emb, item_emb)
        else:
            model, df, user_emb, item_emb = Embedding_train("Classification with Embedding Features")
            return upload_Embedding_model_to_supabase("Classification with Embedding Features", model, df, user_emb, item_emb)

    results = []

    for file_name in file_names:

        if check_file_exists(bucket, file_name):

            ratings_df = load_rating()
            trained_on = load_model_metadata_from_supabase(bucket, file_name)

            if ratings_df.shape == trained_on.shape :
                results.append(f"✅ No new users. Model is already up-to-date at `{file_name}`")
            else:
                results.append(train_and_upload(file_name))
        else:
            results.append(train_and_upload(file_name))
        
    return "\n".join(results)


if __name__ == "__main__":
    result = retrain_models()
    print(result)
