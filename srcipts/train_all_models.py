from models.train_lightfm import run as run_lightfm
from models.train_svd_knn import run as run_svd_knn

def main():
    print("🔄 Навчання LightFM...")
    model, dataset, item_features, df = run_lightfm()

    print("\n🔄 Навчання SVD та KNN...")
    svd_model, knn_model, train_df_surp, test_df_surp = run_svd_knn(df)

    print("\n✅ Усі моделі успішно натреновані та збережені у каталозі 'data/model'")

if __name__ == "__main__":
    main()