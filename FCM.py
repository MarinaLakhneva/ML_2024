import numpy as np
import pandas as pd

# k - количество кластеров 1<j<k
# d - размерность вектора данных 1<l<d
# n - мощность выборки


def calculating_the_degree_of_affiliation(k, n, m, dist):
    u_ij = np.zeros((k, n))
    extent = 1/(1-m)

    for i in range(0, n):
        for j in range(0, k):
            sum = 0.0
            for t in range(0, k):
                sum += (dist[j][i]/dist[t][i])**2
            u_ij[j][i] = sum**extent

    return u_ij

def calculating_the_coordinates_of_the_cluster_center(k, n, data, d, table, m):
    c_jl = np.zeros((d, k))

    for j in range(0, k):
        for l in range(0, d):
            nomerstor = 0.0
            denomerstor = 0.0
            for i in range(0, n):
                nomerstor += table[j][i]**m * data[l][i]
                denomerstor += table[j][i]**m
            c_jl[l][j] = nomerstor/denomerstor

    cluster_center_for_PCA = pd.DataFrame(c_jl)
    cluster_center_for_PCA.to_csv("cluster_center.csv", index=False, header=False)
    return c_jl

# евклидово расстояние
def distance_calculation(k, n,  data, d, c_ij, m):
    dist = np.zeros((k, n))

    for j in range(0, k):
        for i in range(0, n):
            sum = 0.0
            for l in range(0, d):
                sum += (data[l][i] - c_ij[l][j])**m
            dist[j][i] = np.sqrt(sum)
    return dist

def solution(k, n, data, d, table, m, E):
    max = 1000

    while(max > E):
        coordinates = 0
        distance = 0
        affiliation = 0

        coordinates = calculating_the_coordinates_of_the_cluster_center(k, n, data, d, table, m)
        distance = distance_calculation(k, n, data, d, coordinates, m)
        affiliation = calculating_the_degree_of_affiliation(k, n, m, distance)

        max = 0.0
        for j in range(0, k):
            for i in range(0, n):
                difference = affiliation[j][i]-table[j][i]
                if (difference > max):
                    max = difference
        table = affiliation


    dist = pd.DataFrame(distance)
    dist.to_csv("distance.csv", index=False, header=False)
    return table


def fcm(k):
    print(k)
    # # удалила шипики которых нет в 0.025 0.025 0.1 dataset
    # import glob
    #
    # metrics_d = pd.read_csv("data/metrics.csv")
    # print(len(metrics_d))
    # to_delete = []
    # for indexClusterization, p in enumerate(metrics_d['Spine File'].to_numpy()):
    #     if p.replace("/", "\\") not in glob.glob('0.025 0.025 0.1 dataset/*/*.off', recursive=True):
    #         to_delete.append(indexClusterization)
    #
    # for n_drop in range(0, len(to_delete)):
    #     metrics_d = metrics_d.drop(to_delete[n_drop])
    # metrics_d.to_csv("data/metrics_update.csv", indexClusterization=False)


    eps = 0.001
    degree_of_fuzziness = 2
    metrics = pd.read_csv("data/metrics_update.csv")
    OldChordDistribution_metric = metrics['OldChordDistribution']

    dataset = np.zeros((len(OldChordDistribution_metric[0].split()), len(OldChordDistribution_metric)))

    for i in range(0, len(OldChordDistribution_metric)):
        for j in range(0, len(OldChordDistribution_metric[0].split())):
            dataset[j][i] = list(map(float, OldChordDistribution_metric[i][1:-1].split()))[j]

    dataset_for_PCA = pd.DataFrame(dataset)
    dataset_for_PCA.to_csv("dataset.csv", index=False, header=False)

    n = len(OldChordDistribution_metric)
    d = len(OldChordDistribution_metric[0].split())

    # датасет для проверки метрик достоверности кластеризации
    # from ucimlrepo import fetch_ucirepo
    #
    # iris = fetch_ucirepo(id=53)
    #
    # dataset = iris.data.features.values.transpose()
    # n = 150
    # d = 4

    # 11-ть классических метрик
    # metrics = pd.read_csv("data/metrics_for_classic.csv", usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).transpose()
    # dataset_m = pd.DataFrame(metrics)
    # dataset_m.to_csv(path_FCM+str(k)+"/dataset.csv", index=False, header=False)
    # dataset = pd.read_csv(path_FCM+str(k)+'/dataset.csv', header=None, index_col=None).values
    #
    # n = dataset.shape[1]
    # print(n)
    # d = dataset.shape[0]
    # print(d)

    #заполняем таблицу принадлежности случайными значениями
    table_of_accessories = np.random.rand(k, n)
    table_of_accessories /= np.sum(table_of_accessories, axis=0)
    frame_accessories = pd.DataFrame(table_of_accessories)
    frame_accessories.to_csv("accessories.csv", index=False, header=False)

    result = solution(k, n, dataset, d, table_of_accessories, degree_of_fuzziness, eps)
    frame_result = pd.DataFrame(result)
    frame_result.to_csv("FCM.csv", index=False, header=False)