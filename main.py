import pandas as pd
import q1
import q3

# from the root directory read both data sets
dataSetOne = pd.read_csv('A1_dataA.tsv', delimiter='\t')
dataSetTwo = pd.read_csv('A1_dataB.tsv', delimiter='\t')

# call the class and plot the density
# plot the density for dataset A, change the 'A' to 'B' to get
# uncomment the next two lines to get the density plot
q1Obj = q1.DataDensityPlotter(dataSetOne, dataSetTwo)
q1Obj.plotDensity('A')

# call the class to perform 10-fold stratified cross-validation
# and display the accuracy scores for each fold
# here k = 3 and splits = 10, meaning 10 folds
# uncomment the next two lines to get the accuracy scores
# q3Obj = q3.KNNClasssifier(dataSetOne, dataSetTwo, 3, 10)
# print(q3Obj.getAccuracy())

# calling this method will plot the precision-recall curve using the
# 10-fold stratified cross-validation and k = 3, change the 'A' to 'B' to get
# the precision-recall curve for dataset B
# uncomment the next line to get the precision-recall curve
# **also line 18 should have to be uncommented**
# q3Obj.getPrecisionRecallCurve('B')