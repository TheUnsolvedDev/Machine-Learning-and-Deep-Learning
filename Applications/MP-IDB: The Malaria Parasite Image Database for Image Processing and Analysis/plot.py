import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    files = os.listdir('Histories/CSV')
    files = [file for file in files if file.endswith('.csv')]
    for file in files:
        data = pd.read_csv('Histories/CSV/'+file)
        plot_items = [item for item in data.columns if 'val' in item]
        for item in plot_items:
            plt.plot(data[item.replace('val_', '')])
            plt.plot(data[item])
            plt.title(file.split('_')[1] + ' '+item.replace('val_', ''))
            plt.ylabel(item.replace('val_', ''))
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig('diagrams/stats/'+file.split('_')
                        [1] + ' '+item.replace('val_', '')+'.png')
            plt.show()