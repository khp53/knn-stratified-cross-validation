import q4
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNClasssifier:
    def __init__(self, dataA, dataB, k, splits):
        self.dataA = dataA
        self.dataB = dataB
        self.k = k
        self.splits = splits
        self.combinedData = pd.concat([self.dataA, self.dataB])

    def getAccuracy(self):
        X = self.combinedData[['X1', 'X2', 'X3', 'X4', 'X5']]
        y = self.combinedData['class']
        # initialize KNN classifier, k gets passed in the constructor
        knn_classifier = KNeighborsClassifier(n_neighbors = self.k)
        # initialize Stratified K-Fold, folds (splits) get passed in the constructor
        stratified_kfold = StratifiedKFold(n_splits = self.splits)
        # perform 10-fold stratified cross-validation
        accuracy_scores = [] # add accuracy scores to this list
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            knn_classifier.fit(X_train, y_train)
            y_pred = knn_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
        # display the accuracy scores for each fold
        for i, accuracy in enumerate(accuracy_scores, start=1):
            print(f'Fold {i}: Accuracy = {accuracy:.4f}')
        average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        print(f'\nAverage Accuracy: {average_accuracy:.4f}')

    # initialize a object named curve and call GenerateCurve class from q4.py
    def getPrecisionRecallCurve(self, whichDataSet):
        curve = q4.GenerateCurve(self.dataA, self.dataB, self.k, self.splits)
        curve.generateCurve(whichDataSet)