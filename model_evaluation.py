from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump,load
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




def main():
    print("*********Model Evaluation****************")
    model = load('C:/theCave/ISO-ID/train/trained_models/cnn.joblib')
    svm_model = load('C:/theCave/ISO-ID/train/trained_models/cnn_svm.joblib')
    rnd_model = load('C:/theCave/ISO-ID/train/trained_models/cnn_rf.joblib')
    val_ds = lambda: read_csv('C:/theCave/ISO-ID/train/validation_select.csv')
    val_ds = tf.data.Dataset.from_generator(val_ds, output_types = (tf.float32, tf.int64), output_shapes = (tf.TensorShape([3000,1]),tf.TensorShape([18])))
    val_ds = val_ds.batch(256).prefetch(3)
    pred,labels = generate_predictions(model,val_ds,rnd_model)
    print("Legends : ")
    print(label_name)
    print(label_number)

    print(" Confusion Matrix:")
    cm = confusion_matrix(labels, pred)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_name)

    disp.plot()
    plt.show()
    print(" Classification Report:")
    print(classification_report(labels, pred))



if __name__ == "__main__":
    main()