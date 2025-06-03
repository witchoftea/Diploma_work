import pickle
from surprise import Dataset as SurpriseDataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split as surprise_train_test_split
import pandas as pd

def run(df, save_dir="data/models"):
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    surprise_data = SurpriseDataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
    train_surp, test_surp = surprise_train_test_split(surprise_data, test_size=0.2, random_state=42)

    train_df_surp = pd.DataFrame(train_surp.all_ratings(), columns=['uid', 'iid', 'rating'])
    train_df_surp['uid'] = train_df_surp['uid'].astype(int)
    train_df_surp['iid'] = train_df_surp['iid'].astype(int)
    test_df_surp = pd.DataFrame(test_surp, columns=['uid', 'iid', 'rating'])

    # Збереження train/test датасетів для подальшої оцінки
    train_df_surp.to_pickle(f"{save_dir}/train_df_surp.pkl")
    test_df_surp.to_pickle(f"{save_dir}/test_df_surp.pkl")

    # --- SVD ---
    svd = SVD()
    svd.fit(train_surp)
    with open(f"{save_dir}/svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)

    # --- KNN ---
    knn = KNNBasic(sim_options={'user_based': False})
    knn.fit(train_surp)
    with open(f"{save_dir}/knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)

    return svd, knn, train_df_surp, test_df_surp