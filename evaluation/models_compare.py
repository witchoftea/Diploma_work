import numpy as np
from collections import defaultdict

def precision_recall_at_k(model, train_df, test_df, k=10, threshold=4.0):
    """
    Обчислює середні значення Precision@k та Recall@k для Surprise-моделей.
    """
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


def print_comparison_results(lfm_prec, lfm_rec, svd_prec, svd_rec, knn_prec, knn_rec):
    print("=== Порівняльні метрики Precision@k / Recall@k ===")
    print(f"LightFM: precision = {lfm_prec:.4f}, recall = {lfm_rec:.4f}")
    print(f"SVD:      precision = {svd_prec:.4f}, recall = {svd_rec:.4f}")
    print(f"KNN:      precision = {knn_prec:.4f}, recall = {knn_rec:.4f}")
