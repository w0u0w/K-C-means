import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt


def Euclidean_distance(feat_one, feat_two):
    square_distance = 0

    for i in range(len(feat_one)):
        square_distance += (feat_one[i] - feat_two[i]) ** 2

    ed = sqrt(square_distance)
    return ed


class K_Means:
    def __init__(self, k, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}

        #  Инициализация центроид. первые К в датасете
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iterations):
            self.classes = {}
            for j in range(self.k):
                self.classes[j] = []

            # Находим расстояние между точками и кластеры; выбираем ближайшую центроиду
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroids]) for centroids in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            #  Пересчет центроид
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            isOptimal = True  # Флаг оптимального значения

            # Проверка на разницу между новыми и старыми центроидами
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid)/original_centroid * 100) > self.tolerance:
                    isOptimal = False

            # Выход из общего цикла, если результат оптимальный(т.е центроиды меняют свои позиции незначительно)
            if isOptimal:
                break


def kmeans():
    players = pd.read_csv(r".\Cust_Segmentation.csv")
    players = players[['Age', 'Income']]
    dataset = players.astype(int).values.tolist()

    X = players.values
    km = K_Means(3)
    km.fit(X)



    colors = 10*["r", "g", "c", "b", "k"]

    for centroid in km.centroids:
        plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")

    for classification in km.classes:
        color = colors[classification]
        for features in km.classes[classification]:
            plt.scatter(features[0], features[1], color=color, s=30)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Income', fontsize=14)
    plt.show()

    players = pd.read_csv(r".\Cust_Segmentation.csv")
    players2 = players[['Edu', 'Age', 'Income']]
    X2 = players2.values
    km = K_Means(3)
    km.fit(X2)
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, rect=(0.0, 0.0, .95, 1.0), elev=48, azim=134)
    ax.set_xlabel('Education')
    ax.set_ylabel('Age')
    ax.set_zlabel('Income')
    # ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(float))
    for centroid in km.centroids:
        ax.scatter(km.centroids[centroid][0], km.centroids[centroid][1], km.centroids[centroid][2], s=130, marker="x")
    for classification in km.classes:
        color = colors[classification]
        for features in km.classes[classification]:
            ax.scatter(features[0], features[1], features[2], color=color, s=30)
    plt.show()


if __name__ == '__main__':
    kmeans()


