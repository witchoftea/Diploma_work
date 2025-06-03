import pickle
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
import scipy.sparse
from collections import defaultdict

def run(save_dir="data/models"):
    ratings = pd.read_csv("data/inputs/ratings.csv")
    books = pd.read_csv("data/inputs/books.csv")
    book_tags = pd.read_csv("data/inputs/book_tags.csv")
    tags = pd.read_csv("data/inputs/tags.csv")

    book_tags = book_tags.merge(tags, on="tag_id", how="left")
    book_tags = book_tags.merge(books[["book_id", "goodreads_book_id"]], on="goodreads_book_id", how="inner")
    book_tags = book_tags[book_tags["count"] > 10]

    df = ratings.merge(books, on='book_id', how='inner')

    # якщо довго працює цей скрипт або evaluate_models, то виставити меншу кількість користувачів
    df = df[df['user_id'] < 10000]

    authorDf = df[['authors', 'book_id']].drop_duplicates()
    itemFeatureAssignments = []
    itemFeatureList = []
    for ii in range(len(authorDf)):
        itemFeatureAssignments.append((authorDf['book_id'].iloc[ii],
                              authorDf['authors'].iloc[ii].split(", ")))
        itemFeatureList.extend(authorDf['authors'].iloc[ii].split(", "))
    itemFeatureList = set(itemFeatureList)

    dataset = Dataset()
    dataset.fit(users = df['user_id'],
                items = df['book_id'],
                item_features = itemFeatureList)
    item_features = dataset.build_item_features(itemFeatureAssignments)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train, _ = dataset.build_interactions(list(zip(train_df['user_id'], train_df['book_id'])))
    test, _ = dataset.build_interactions(list(zip(test_df['user_id'], test_df['book_id'])))

    model = LightFM(loss='warp', learning_rate=0.05, random_state=1)
    model.fit(train, epochs=5, item_features=item_features)

    # Зберігаємо все необхідне
    with open(f"{save_dir}/lightfm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{save_dir}/lightfm_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    scipy.sparse.save_npz(f"{save_dir}/lightfm_item_features.npz", item_features)

    # Підготовка мапінгів та допоміжних структур
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    inv_user_id_map = {v: k for k, v in user_id_map.items()}
    inv_item_id_map = {v: k for k, v in item_id_map.items()}
    book_id_to_title = dict(zip(books['book_id'], books['title']))
    tag_to_books = defaultdict(list)
    for _, row in book_tags.iterrows():
        tag = row['tag_name'].lower()
        tag_to_books[tag].append((row['book_id'], row['count']))
    top_genres = (
        book_tags["tag_name"]
        .value_counts()
        .loc[lambda s: s > 50]
        .head(30)
        .index
        .tolist()
    )

    with open(f"{save_dir}/lightfm_mappings.pkl", "wb") as f:
        pickle.dump({
            "user_id_map": user_id_map,
            "item_id_map": item_id_map,
            "inv_user_id_map": inv_user_id_map,
            "inv_item_id_map": inv_item_id_map,
            "book_id_to_title": book_id_to_title,
            "df": df,
            "top_genres": top_genres,
            "tag_to_books": tag_to_books
        }, f)

    return model, dataset, item_features, df