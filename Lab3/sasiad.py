from scipy.spatial import distance
from sklearn.base import ClassifierMixin, BaseEstimator


class NearestNeighborClassifier(ClassifierMixin,BaseEstimator):
    def fit(self, features, labels):
        self.features_train = features
        self.labels_train = labels

    def predict(self, features_test):
        predictions = []

        for row in features_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = distance.cityblock(row, self.features_train[0])
        best_index = 0

        for i in range(0, len(self.features_train)):
            dist = distance.cityblock(row, self.features_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.labels_train[best_index]