import numpy as np
import pandas as pd
from lightfm.evaluation import precision_at_k, recall_at_k
from utils.load_trained_models import load_lightfm_model, load_svd_model, load_knn_model

from collections import defaultdict


def precision_recall_at_k_surprise(model, train_df, test_df, k=10, threshold=4.0):
    user_rated_items = defaultdict(set)
    for _, row in train_df.iterrows():
        user_rated_items[int(row['uid'])].add(int(row['iid']))

    user_metrics = defaultdict(list)
    for uid in test_df['uid'].unique():
        seen = user_rated_items.get(uid, set())
        all_items = set(train_df['iid'].unique())
        unseen_items = list(all_items - seen)
        preds = [model.predict(uid, iid) for iid in unseen_items]
        preds.sort(key=lambda x: x.est, reverse=True)
        top_k = preds[:k]

        relevant = set(test_df[(test_df['uid'] == uid) & (test_df['rating'] >= threshold)]['iid'])
        recommended = set([int(p.iid) for p in top_k])
        hits = len(recommended & relevant)

        if len(recommended) > 0:
            prec = hits / k
            rec = hits / len(relevant) if len(relevant) > 0 else 0
            user_metrics[uid] = [prec, rec]

    precisions = [v[0] for v in user_metrics.values()]
    recalls = [v[1] for v in user_metrics.values()]
    return np.mean(precisions), np.mean(recalls)


def evaluate_all():
    print("📦 Завантаження моделей...")
    lfm_model, dataset, item_features, mappings = load_lightfm_model()
    svd_model = load_svd_model()
    knn_model = load_knn_model()

    print("🔍 Обчислення метрик LightFM...")
    train, _ = dataset.build_interactions(list(zip(mappings['df']['user_id'], mappings['df']['book_id'])))
    test = train  # Спрощено: тест = весь набір для демонстрації

    lfm_precision = precision_at_k(lfm_model, test, item_features=item_features, k=10).mean()
    lfm_recall = recall_at_k(lfm_model, test, item_features=item_features, k=10).mean()

    print("🔍 Обчислення метрик SVD...")
    train_df_surp = pd.read_pickle("data/models/train_df_surp.pkl")
    test_df_surp = pd.read_pickle("data/models/test_df_surp.pkl")
    svd_precision, svd_recall = precision_recall_at_k_surprise(svd_model, train_df_surp, test_df_surp)

    print("🔍 Обчислення метрик KNN...")
    knn_precision, knn_recall = precision_recall_at_k_surprise(knn_model, train_df_surp, test_df_surp)

    print("\n📊 Порівняння моделей:")
    print(f"LightFM: Precision@10 = {lfm_precision:.4f}, Recall@10 = {lfm_recall:.4f}")
    print(f"SVD:      Precision@10 = {svd_precision:.4f}, Recall@10 = {svd_recall:.4f}")
    print(f"KNN:      Precision@10 = {knn_precision:.4f}, Recall@10 = {knn_recall:.4f}")


if __name__ == "__main__":
    evaluate_all()