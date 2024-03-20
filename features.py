import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=12)

def remove_outliers(column):
    column = column.copy()  # явное создание копии
    median = np.median(column)
    mad = np.median(np.abs(column - median))
    threshold = 3 * mad
    # Заменяем вбросы на ближайшие к границам данные
    column.loc[column < median - threshold] = median - threshold
    column.loc[column > median + threshold] = median + threshold
    return column

def features_cluster(file_path = 'features.csv'):
    df = pd.read_csv(file_path)

    for column in df.columns[2:]:
        df[column] = remove_outliers(df[column])

    # Выбор колонок с широтой и долготой
    coordinates = df[['lat', 'lon']]

    # Пример использования KMeans для кластеризации
    kmeans = KMeans(n_clusters=12)  # Укажите количество кластеров

    kmeans.fit(coordinates)
    joblib.dump(kmeans, 'kmeans.pkl')

    # Создаем DataFrame с метками кластеров
    cluster_df = pd.DataFrame({'cluster': kmeans.labels_})

    # Добавляем новый DataFrame к исходному
    df = pd.concat([df, cluster_df], axis=1)

    cluster_means = df.groupby('cluster').mean()

    cluster_means.to_csv('cluster_means.csv')

    return {"kmeans_model": kmeans, "new_cluster_df": cluster_means, "features_cluster" : df}

def concat_df_and_features_with_preproc(file_path = 'train.csv',
                                        kmeans = 'kmeans.pkl',
                                        cluster_means = 'cluster_means.csv',
                                        end_file_name = "train_preprocessing.csv",):
    if type(kmeans) == str:
        kmeans = joblib.load('kmeans.pkl')
    if type(cluster_means) == str:
        cluster_means = pd.read_csv('cluster_means.csv')

    df = pd.read_csv(file_path)

    coordinates = df[['lat', 'lon']]

    # Создаем DataFrame с метками кластеров
    cluster_df = pd.DataFrame({'cluster': kmeans.predict(coordinates)})

    # Добавляем новый DataFrame к исходному
    df = pd.concat([df, cluster_df], axis=1)

    merged_df = pd.merge(df, cluster_means, on='cluster', suffixes=('', '_y'), how='left')
    merged_df.drop([col for col in merged_df.columns if col.endswith('_y')], axis=1, inplace=True)

    for column in merged_df.columns[2:]:
        merged_df[column] = remove_outliers(merged_df[column])

    merged_df = merged_df.drop(columns=['id'])
    merged_df.to_csv(end_file_name)

    return merged_df

if __name__ == '__main__':
    features_cluster = features_cluster()

    df = features_cluster["features_cluster"]
    # Визуализация кластеров
    plt.figure(figsize=(10, 8))
    plt.scatter(df['lon'], df['lat'], s=50, c=df['cluster'], cmap='tab10', alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clustered Locations')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

    print(concat_df_and_features_with_preproc('train.csv',
                                              features_cluster["kmeans_model"],
                                              features_cluster["new_cluster_df"]))