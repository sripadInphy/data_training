from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from joblib import dump,load
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import os
import ast

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# label_name = ['U', 'Tc', 'Pu', 'Cs', 'Co', 'K', 'U', 'Co57', 'Th', 'Am', 
#                     'Pu239', 'I', 'Ra', 'Ga', 'Ir', 'Ba', 'U235', 'Cf252']
label_name = ['Cs137','Co60','K40','Co57','Am241','I131','Ir192','Ba133']
# label_name = ['Co57', 'Co60', 'Cr51', 'Cs137', 'F18', 'Ga67', 'I123', 'I125', 'I131', 'In111', 'Ir192', 'Se75', 'Sm153', 'Xe133']
label_number = LabelEncoder().fit_transform(label_name).tolist()


no_of_bins = 1500
no_of_class = 9
pos_weight = 1
neg_weight = 1

def normalize(arr):
    """
    Normalizes an array by dividing each element by the sum of all elements.

    Args:
        arr (list): A list of numbers to normalize.

    Returns:
        list: A normalized list of numbers.
    """
    arr = [float(x) for x in arr]
    arr_sum = sum(arr)
    return [float(i) / arr_sum for i in arr]



def generate_predictions(model, generator, reshape=False):
    X = []
    y = []
    for batch in generator:
        features, label = batch
        X.append(features)
        label = np.argmax(label, axis=1)
        y.append(label)
        break
    X = np.concatenate(X)
    y = np.concatenate(y)
    axis = 1
    print("Shape : ", X.shape)
    if reshape is False:
        print("called?")
        X = X.reshape(len(X), -1)
        axis = None
    features = model.predict(X)
    predictions = features
    predictions = np.argmax(predictions, axis=axis)
    print(predictions)
    return predictions, y


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        count = 0

        for line in reader:
            features = [np.float64(n) for n in line[:no_of_bins]]
            label = ast.literal_eval(line[-1])  # safely evaluate string as Python expression
            label = np.array(label, dtype=np.float32)
            features = np.array(features)
            features = features.reshape(no_of_bins,-1)
            # features = features*1000  #scale it up cuz the model can't recognize small values
            # yield features, tf.keras.utils.to_categorical(int(label), num_classes=no_of_class)
            yield features, label



def plot_confusion_matrix(matrix, title):
    labels = [i for i in range(14)]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    plt.show()



def model_field_test(folder_path, models,cnn_model):
    files = os.listdir(folder_path)
    results = []
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        data = df.iloc[:, 1]
        spectrum = np.append(data.values, [0])
        spectrum = normalize(spectrum)
        spectrum = [float(val)*10000 for val in spectrum]
        spectrum = np.array(spectrum)
        label = file.split('.')[0]
        ind_result = [label]
        ind_result.append(label_name[label_number.index(cnn_model.predict(spectrum.reshape(1,1500,1)).argmax())])
        for model in models:
            result = model.predict(spectrum.reshape(1, 1500))
            ind_result.append(label_name[label_number.index(result[0])])
        results.append(ind_result)
    # print("Isotope    |  CNN   |   SVM   |   NB   |   LR   | KNN")
    print("Isotope    |  CNN ")
    for ind in results:
        # print(f"{ind[0]}  |  {ind[1]}  |  {ind[2]}  |  {ind[3]}  |  {ind[4]}  |  {ind[5]}")
        print(f"{ind[0]}  |  {ind[1]} ")






def main():
    print("*********Model Evaluation****************")
    cnn_model = load('C:/theCave/ISO-ID/train/trained_models/cnn.joblib')
    svm_model = load('C:/theCave/ISO-ID/train/trained_models/svm.joblib')
    knn = load('C:/theCave/ISO-ID/train/trained_models/knn.joblib')
    nb = load('C:/theCave/ISO-ID/train/trained_models/nb.joblib')
    lr = load('C:/theCave/ISO-ID/train/trained_models/lr.joblib')
    # rnd_model = load('C:/theCave/ISO-ID/train/trained_models/cnn_rf.joblib')
    val_ds = lambda: read_csv('C:/theCave/ISO-ID/train/validation_select.csv')
    # val_ds = tf.data.Dataset.from_generator(val_ds, output_types = (tf.float32, tf.int64), output_shapes = (tf.TensorShape([3000,1]),tf.TensorShape([18])))
    val_ds = tf.data.Dataset.from_generator(val_ds, output_signature=(tf.TensorSpec(shape=(no_of_bins,1),dtype=tf.dtypes.float64), tf.TensorSpec(shape=(no_of_class,))))
    val_ds = val_ds.batch(256).prefetch(3)
   
    test_data_folder = 'C:/theCave/ISO-ID/captures/captures_for_test'

    list_of_model = [svm_model,knn,nb,lr]
    if False:                  #Field Test
        model_field_test(test_data_folder,list_of_model, cnn_model)



    if True:                  #Model Evaluate
        models = [cnn_model]
        model_names = ['CNN']
        results = []
        confusion_matrices = []
        reshape = False
        for test_model, name in zip(models, model_names):
            if name == 'CNN':
                reshape = True
            predictions, labels = generate_predictions(test_model, val_ds,reshape)
            # predictions = np.argmax(predictions, axis=1)
            # labels = np.argmax(labels, axis=1)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            results.append([name, accuracy, precision, recall, f1_score])
            confusion_matrices.append(confusion_matrix(labels, predictions))
            reshape = False


        results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
        print(results_df)


        # Plot confusion matrices
        fig, axs = plt.subplots(1, len(confusion_matrices), figsize=(15, 5))
        for i, cm in enumerate(confusion_matrices):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)
            disp.plot(ax=axs[i], cmap='Blues', values_format='d')
            axs[i].set_title(model_names[i])
            axs[i].grid(False)
            axs[i].xaxis.tick_top()
            axs[i].set_xlabel('Predicted Label')
            axs[i].set_ylabel('True Label')

        # # Plot differences between confusion matrices
        # fig, ax = plt.subplots(figsize=(7, 5))
        # diff_cm = confusion_matrices[0]
        # for i in range(1, len(confusion_matrices)):
        #     diff_cm -= confusion_matrices[i]
        # disp = ConfusionMatrixDisplay(confusion_matrix=diff_cm, display_labels=label_name)
        # disp.plot(ax=ax, cmap='coolwarm', values_format='d')
        # ax.set_title('Difference between Confusion Matrices')
        # ax.grid(False)
        # ax.xaxis.tick_top()
        # ax.set_xlabel('Predicted Label')
        # ax.set_ylabel('True Label')
        plt.show()



if __name__ == "__main__":
    main()