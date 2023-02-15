import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns

from dataset import *
from model import *
from params import *

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    dataset = Dataset()
    train, val = dataset.generators()
    models = [MobileNet, DenseNet,  ResNext, XCeption, SE_ResNet]
    model_wieghts_path = ['models/model_' +
                          model.__name__+'_.h5' for model in models]
    print(model_wieghts_path)

    for ind, model in tqdm.tqdm(enumerate(models)):
        y_true = []
        y_pred = []
        model = model()
        model.load_weights(model_wieghts_path[ind])
        for data, labels in val:
            y_true.extend(labels.numpy())
            y_pred_temp = model.predict(data.numpy(), verbose=0)
            for pred in y_pred_temp:
                if pred > 0.5:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

        confusion_mat = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        sns.heatmap(confusion_mat, annot=True,
                    annot_kws={"size": 16})  # font size
        plt.title(models[ind].__name__+'_confusion_matrix')
        plt.savefig(models[ind].__name__+'_confusion_matrix.png')
        plt.show()
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=models[ind].__name__+"_auc="+str(auc))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc=4)
        plt.savefig(models[ind].__name__+'_AUC_curve.png')
        plt.show()

'''
[[3465  902]
 [1372 2998]]
0.7687179487179487
0.6860411899313501

[[1739 2628]
 [ 819 3551]]
0.5746884609160058
0.8125858123569794
'''
