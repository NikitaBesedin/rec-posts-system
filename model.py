import os
from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("Here was the address")

app = FastAPI()


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем, где выполняется код: в лмс или локально
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("/my/super/path")
    from catboost import CatBoostClassifier
    from_file = CatBoostClassifier()
    model = from_file.load_model(model_path)
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    chunksize = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=chunksize):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM n_besedin_14_features_lesson_22", con=engine)
    return df


def load_liked_posts() -> pd.DataFrame:
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action='like' """
    liked_posts = batch_load_sql(liked_posts_query)
    return liked_posts

# Загрузка модели
model = load_models()

# Загрузка таблиц
post_text_df = pd.read_sql("SELECT * FROM public.post_text_df", con=engine)
vectorized_text_df = pd.read_sql("SELECT * FROM n_besedin_14_lesson_10_vecs", con=engine)
df_with_user_features = load_features()
liked_posts = load_liked_posts()


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:

    # Создание датафрейма с id всех постов и одним юзером для дальнейшего предикта
    df_predict = pd.DataFrame({'post_id': vectorized_text_df['post_id'], 'user_id': id})

    # Merge фичей к датафрейму для предикта
    df_predict = df_predict.merge(df_with_user_features, on='user_id', how='left') # Merge фичей юзера
    df_predict = df_predict.merge(vectorized_text_df, on='post_id', how='left') # Merge фичей постов

    # Добавление времени к датафрейму
    df_predict['hour'] = time.hour
    df_predict['month'] = time.month

    # Предикт вероятностей
    prediction = model.predict_proba(df_predict)[:, 1]

    # Добавление в датафрейм вероятностей и текстов постов с темами
    df_predict['prediction'] = prediction
    df_predict = df_predict.merge(post_text_df, on='post_id', how='left')

    # Удаление постов, которые пользователь уже лайкал
    liked_posts_for_user = liked_posts[liked_posts.user_id == id].post_id.values
    df_predict = df_predict[~df_predict.index.isin(liked_posts_for_user)]

    # Сортировка вероятностей по убыванию
    df_predict = df_predict.sort_values(by='prediction', ascending=False).reset_index(drop=True)

    # Отбор постов
    df_predict = df_predict.loc[:limit - 1]

    return [PostGet(**{
        'id': post_id,
        'text': df_predict[df_predict['post_id'] == post_id]['text'].values[0],
        'topic': df_predict[df_predict['post_id'] == post_id]['topic'].values[0]
        }) for post_id in df_predict['post_id']]
