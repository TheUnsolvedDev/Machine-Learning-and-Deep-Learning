from silence_tensorflow import silence_tensorflow
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from params import *
from model import *
from dataset import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
silence_tensorflow()


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    dataset = Dataset()
    train, val, test = dataset.generators()
    models = models = [MobileNet, DenseNet, InceptionNet,
                       ResNext, XCeption, SE_ResNet, Alexnet, ZFnet,  VGG, lenet5_model, PyramidalNet]
    model_wieghts_path = ['models/model_' +
                          model.__name__+'_.h5' for model in models]
    print(model_wieghts_path)

    for ind, model in tqdm.tqdm(enumerate(models)):
        y_true = []
        y_pred = []
        model = model()
        model.load_weights(model_wieghts_path[ind])
        for data, labels in test:
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
        plt.savefig('diagrams/stats/' +
                    models[ind].__name__+'_confusion_matrix.png')
        plt.show()
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, label=models[ind].__name__+"_auc="+str(auc))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc=4)
        plt.savefig('diagrams/stats/'+models[ind].__name__+'_AUC_curve.png')
        plt.show()

        print(models[ind].__name__, 'has', len(model.layers), 'layers')
