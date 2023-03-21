from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from joblib import dump,load
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

label_name = ['U', 'Tc', 'Pu', 'Cs', 'Co', 'K', 'U', 'Co57', 'Th', 'Am', 
                    'Pu239', 'I', 'Ra', 'Ga', 'Ir', 'Ba', 'U235', 'Cf252']
label_number = LabelEncoder().fit_transform(label_name).tolist()

def generate_predictions(cnn_model, generator, sec_model=None):
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
    
    # Extract features using CNN model
    features = cnn_model.predict(X)
    predictions = features
    predictions = np.argmax(predictions, axis=1)
    if sec_model != None:
        # Reshape features for SVM model
        features = features.reshape(len(features), -1)
        
        predictions = sec_model.predict(features)
    
    return predictions, y


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        count = 0

        glob_labels  = []
        glob_features = []
        for line in reader:
            # print('line : ' ,line)
            if count == 1 : 
                glob_features = []
                glob_labels = []
        # record = line.rstrip().split(',').astype(int)
            features = [int(float(n)) for n in line[:3000]]
            label = line[-1]
            features = np.array(features)
            features = features.reshape(3000,-1)
  
            yield features, tf.keras.utils.to_categorical(int(label), num_classes=18)



def plot_confusion_matrix(matrix, title):
    labels = [i for i in range(18)]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    plt.show()





def main():
    print("*********Model Evaluation****************")
    model = load('C:/theCave/ISO-ID/train/trained_models/cnn.joblib')
    svm_model = load('C:/theCave/ISO-ID/train/trained_models/cnn_svm.joblib')
    rnd_model = load('C:/theCave/ISO-ID/train/trained_models/cnn_rf.joblib')
    val_ds = lambda: read_csv('C:/theCave/ISO-ID/train/validation_select.csv')
    val_ds = tf.data.Dataset.from_generator(val_ds, output_types = (tf.float32, tf.int64), output_shapes = (tf.TensorShape([3000,1]),tf.TensorShape([18])))
    val_ds = val_ds.batch(256).prefetch(3)
   


    models = [None, rnd_model, svm_model]
    model_names = ['CNN', 'Random Forest', 'SVM']
    results = []
    confusion_matrices = []

    for sec_model, name in zip(models, model_names):
        predictions, labels = generate_predictions(model, val_ds,sec_model)
        # predictions = np.argmax(predictions, axis=1)
        # labels = np.argmax(labels, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        results.append([name, accuracy, precision, recall, f1_score])
        confusion_matrices.append(confusion_matrix(labels, predictions))


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
    # plt.show()



if __name__ == "__main__":
    main()