import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def charts(k):
    metrics = pd.read_csv("data/metrics_update.csv")
    property = metrics['OldChordDistribution']

    dataset = np.zeros((len(property[0].split()), len(property)))
    for i in range(0, len(property)):
        for j in range(0, len(property[0].split())):
            dataset[j][i] = list(map(float, property[i][1:-1].split()))[j]
    n = len(property)
    result = pd.read_csv("FCM.csv", header=None, index_col=None).values

    from sklearn.preprocessing import StandardScaler

    data = dataset.transpose()
    data = StandardScaler().fit_transform(data)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)

    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(0, n):
        x[i] = principalComponents[i][0]
        y[i] = principalComponents[i][1]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Clusters ', fontsize=20)
    ax.set_xlabel('PC 1', fontsize=15)
    ax.set_ylabel('PC 2', fontsize=15)
    colors = ['#EE0000', '#FFFF00', '#006400']
    for t in range(0, k):
        probability = np.zeros(len(result[0]))
        for j in range(0, len(result[0])):
            probability[j] = result[t][j]
        ax.scatter(x, y, s=100, alpha=probability, color=colors[t], edgecolors="black")
    ax.legend(["1", "2", "3"])
    for t in range(0, k):
        ax.get_legend().legend_handles[t].set_alpha(1)
    plt.show()