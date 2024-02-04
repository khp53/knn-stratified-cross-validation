import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataDensityPlotter:
    def __init__(self, dataA, dataB):
        self.dataA = dataA
        self.dataB = dataB

    # concat them, meaning take the datasets as array and merge them
    def concatData(self):
        return pd.concat([self.dataA, self.dataB])

    def plotDensity(self, whichDataSet):
        # set up subplots to view better, note: fig size can be adjusted
        # based on prof.'s/TA's laptop screen size
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
        # using density plots to compare the data
        # the hue parameter is used to group the data by class
        # iterate through X 1 to 5 and create density plots
        for i, column in enumerate(['X1', 'X2', 'X3', 'X4', 'X5']):
            row, col = divmod(i, 3)
            if whichDataSet == 'A':
                sns.kdeplot(
                    data = self.dataA, 
                    x = column, 
                    hue = 'class', 
                    common_norm = False, 
                    fill = True,
                    alpha = 0.2, # for transparency 
                    ax = axes[row, col],
                )
            else:
                sns.kdeplot(
                    data = self.dataB, 
                    x = column, 
                    hue = 'class', 
                    common_norm = False, 
                    fill = True,
                    alpha = 0.2, # for transparency 
                    ax = axes[row, col],
                )
            axes[row, col].set_title(f'For {column} column')
            axes[row, col].set_xlabel(column)
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend(title='Class', labels=['0', '1'])

        if whichDataSet == 'A':
            plt.suptitle('Density Plots for Data A')
        else:
            plt.suptitle('Density Plots for Data B')
        plt.tight_layout()
        plt.show()