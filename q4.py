import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score, average_precision_score, confusion_matrix

class GenerateCurve:
    def __init__(self, dataA, dataB, k, splits):
        self.dataA = dataA
        self.dataB = dataB
        self.k = k
        self.splits = splits
        # extract features (X1 to X5) and labels (class) from dataset A
        self.xA = self.dataA[['X1', 'X2', 'X3', 'X4', 'X5']]
        self.yA = self.dataA['class']
        # from dataset B
        self.xB = self.dataB[['X1', 'X2', 'X3', 'X4', 'X5']]
        self.yB = self.dataB['class']
        # init KNN classifier and Stratified K-Fold
        self.knn_classifier = KNeighborsClassifier(n_neighbors = self.k)
        self.stratified_kfold = StratifiedKFold(n_splits = self.splits)
        # init empty lists to store metrics and precision-recall curves
        self.all_precision_curve = []
        self.all_recall_curve = []
        self.all_accuracy = []
        self.all_f1 = []
        self.all_precision = []
        self.all_recall = []
        self.all_avg_precision = []

    def calculate_metrics(self, y_true, y_pred_proba):
        y_pred = (y_pred_proba[:, 1]).astype(int)

        # calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        # precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])

        # calc average precision scorew
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])

        return accuracy, f1, precision, recall, precision_curve, recall_curve, avg_precision

    def generateCurve(self, whichDataSet):
        # create figure and adjust size
        plt.figure(figsize=(10, 8))
        # based on which dataset, get the x and y
        if whichDataSet == 'A':
            x = self.xA
            y = self.yA
        else:
            x = self.xB
            y = self.yB
        # iterate through each fold
        for fold_num, (train_index, test_index) in enumerate(self.stratified_kfold.split(x, y), start=1):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # train the KNN classifier
            self.knn_classifier.fit(x_train, y_train)

            # predict probabilities on the test set
            y_pred_proba = self.knn_classifier.predict_proba(x_test)

            # calculate metrics
            accuracy, f1, precision, recall, precision_curve, recall_curve, avg_precision = self.calculate_metrics(y_test, y_pred_proba)

            # store metrics for each fold
            self.all_accuracy.append(accuracy)
            self.all_f1.append(f1)
            self.all_precision.append(precision)
            self.all_recall.append(recall)
            self.all_avg_precision.append(avg_precision)

            # store precision-recall curves for each fold
            self.all_precision_curve.append(precision_curve)
            self.all_recall_curve.append(recall_curve)

            # plot Precision-Recall curve for each fold on the same subplot
            plt.plot(recall_curve, precision_curve, label=f'Fold {fold_num}')

        # display average metrics across all folds
        avg_accuracy = np.mean(self.all_accuracy)
        avg_f1 = np.mean(self.all_f1)
        avg_precision = np.mean(self.all_precision)
        avg_recall = np.mean(self.all_recall)
        avg_avg_precision = np.mean(self.all_avg_precision)

        print(f'Metrics for Data set {whichDataSet}:')
        print(f'Average Accuracy: {avg_accuracy:.4f}')
        print(f'Average F1-score: {avg_f1:.4f}')
        print(f'Average Precision: {avg_precision:.4f}')
        print(f'Average Recall: {avg_recall:.4f}')
        print(f'Average Average Precision: {avg_avg_precision:.4f}')

        # set labels and title
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if whichDataSet == 'A':
            plt.title('Precision-Recall Curve for Data A - All Folds')
        else:    
            plt.title('Precision-Recall Curve for Data B - All Folds')
        plt.legend()
        plt.show()