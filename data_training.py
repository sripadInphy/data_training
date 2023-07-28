### Import Requirements ###
# Import the required libraries and modules
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from joblib import dump, load
import keras.backend as K
import csv
import os
import pickle as pkl
import random
import datetime
import glob
import re
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters


from custom_loss import class_loss, conc_loss

no_of_bins = 1500
no_of_class = 6
pos_weight = 5
neg_weight = 1

"""
Model Creation
"""


def build_classification_model(hp):
    # Build a CNN model for classification with tunable hyperparameters
    CNN_Classifier = Sequential()
    # FIRST LAYER
    CNN_Classifier.add(
        Conv1D(
            filters=hp.Int("conv1d_filters", min_value=16, max_value=64, step=16),
            kernel_size=hp.Int("conv1d_kernel", min_value=2, max_value=5, step=1),
            input_shape=(no_of_bins, 1),
        )
    )
    CNN_Classifier.add(Activation("relu"))
    CNN_Classifier.add(MaxPooling1D(pool_size=2))
    CNN_Classifier.add(Flatten())

    # SECOND LAYER
    CNN_Classifier.add(
        Dense(
            units=hp.Int("dense_units", min_value=32, max_value=128, step=32),
            activation="relu",
        )
    )
    CNN_Classifier.add(
        Dropout(rate=hp.Float("dense_dropout", min_value=0.2, max_value=0.5, step=0.1))
    )

    # THIRD LAYER
    CNN_Classifier.add(
        Dense(
            units=hp.Int("dense_units_2", min_value=16, max_value=64, step=16),
            activation="relu",
        )
    )
    CNN_Classifier.add(
        Dropout(
            rate=hp.Float("dense_dropout_2", min_value=0.2, max_value=0.5, step=0.1)
        )
    )

    # OUTPUT LAYER
    CNN_Classifier.add(Dense(no_of_class, activation="sigmoid"))

    # Compile the models with custom loss functions
    CNN_Classifier.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, decay=1e-6),
        loss=class_loss,
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    return CNN_Classifier


def build_regression_model(hp):
    # Build a CNN model for concentration regression with tunable hyperparameters
    CNN_Concentration = Sequential()
    # FIRST LAYER
    CNN_Concentration.add(
        Conv1D(
            filters=hp.Int("conv1d_filters", min_value=16, max_value=64, step=16),
            kernel_size=hp.Int("conv1d_kernel", min_value=2, max_value=5, step=1),
            input_shape=(no_of_bins, 1),
        )
    )
    CNN_Concentration.add(Activation("relu"))
    CNN_Concentration.add(MaxPooling1D(pool_size=2))
    CNN_Concentration.add(Flatten())

    # SECOND LAYER
    CNN_Concentration.add(
        Dense(
            units=hp.Int("dense_units", min_value=32, max_value=128, step=32),
            activation="relu",
        )
    )
    CNN_Concentration.add(
        Dropout(rate=hp.Float("dense_dropout", min_value=0.2, max_value=0.5, step=0.1))
    )

    # THIRD LAYER
    CNN_Concentration.add(
        Dense(
            units=hp.Int("dense_units_2", min_value=16, max_value=64, step=16),
            activation="relu",
        )
    )
    CNN_Concentration.add(
        Dropout(
            rate=hp.Float("dense_dropout_2", min_value=0.2, max_value=0.5, step=0.1)
        )
    )

    # OUTPUT LAYER
    CNN_Concentration.add(Dense(no_of_class, activation="linear"))

    CNN_Concentration.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, decay=1e-6),
        loss=conc_loss,
        metrics=["mae", "mse"],
    )

    return CNN_Concentration


def hyperparameter_tuning(train_dataset, val_dataset):
    # Perform hyperparameter tuning using Keras Tuner

    # Initialize a random search tuner for classification
    class_tuner = RandomSearch(
        build_classification_model,
        objective="val_loss",
        max_trials=10,  # You can adjust the number of trials
        directory="classification_tuning",
        project_name="cnn_class_tuner",
    )

    # Initialize a random search tuner for regression
    conc_tuner = RandomSearch(
        build_regression_model,
        objective="val_loss",
        max_trials=10,  # You can adjust the number of trials
        directory="concentration_tuning",
        project_name="cnn_conc_tuner",
    )

    # Search for the best hyperparameters for classification
    class_tuner.search(train_dataset, epochs=10, validation_data=val_dataset)

    # Search for the best hyperparameters for concentration
    conc_tuner.search(train_dataset, epochs=10, validation_data=val_dataset)

    # Get the best hyperparameters found during tuning for classification
    best_class_hps = class_tuner.get_best_hyperparameters(num_trials=1)[0]

    # Get the best hyperparameters found during tuning for concentration
    best_conc_hps = conc_tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_class_hps, best_conc_hps


def train_with_best_hyperparameters(
    train_dataset, val_dataset, class_best_hps, conc_best_hps
):
    # Train the model using the best hyperparameters

    # Build the classification model with the best hyperparameters
    class_model = build_classification_model(class_best_hps)

    # Build the concentration regression model with the best hyperparameters
    conc_model = build_regression_model(conc_best_hps)

    # Define the callbacks
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
    fname = os.path.sep.join(
        [
            "C:/theCave/ISO-ID/data_prep/weights/",
            "weights-{epoch:03d}-{val_loss:.4f}.hdf5",
        ]
    )
    log_dir = "C:/theCave/ISO-ID/train/logs/fit/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fname,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    # Train the models
    class_history = class_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[model_checkpoint_callback, tensorboard_callback],
        verbose=2,
    )
    conc_history = conc_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[model_checkpoint_callback, tensorboard_callback],
        verbose=2,
    )

    return class_model, conc_model, class_history, conc_history


"""
Model History 
evaluate the model
"""


def plot_model_history(model_name, history, epochs):
    print(model_name)
    plt.figure(figsize=(15, 5))

    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(history["mae"])), history["mae"], "r")
    plt.plot(
        np.arange(1, len(history["val_accuracy"]) + 1), history["val_accuracy"], "g"
    )
    plt.xticks(np.arange(0, epochs + 1, epochs / 10))
    plt.title("Training Accuracy vs. Validation Accuracy")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(history["loss"]) + 1), history["loss"], "r")
    plt.plot(np.arange(1, len(history["val_loss"]) + 1), history["val_loss"], "g")
    plt.xticks(np.arange(0, epochs + 1, epochs / 10))
    plt.title("Training Loss vs. Validation Loss")
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="best")

    plt.show()


"""
Data Processing.

- Combines individual isotope csv files to create training csv file and add
a column to label the spectrum.

- From consolidated csv file split the data set to traininf and validation
set.
"""

# label_name = ['Cs137','Co60','K40','Co57','Am241','I131','Ir192','Ba133']
label_name = []
label_number = LabelEncoder().fit_transform(label_name)


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


elements = set()
# Regular expression to extract element names from filename
regex = r"\(\((.*?)\),"


def extract_element_names(filename, output_list):
    pattern = r"combos_\(\((.*?)\)"
    match = re.search(pattern, filename)
    if match:
        elements = match.group(1).split(", ")
        element_names = [element.strip("'") for element in elements]
        for element in element_names:
            if element not in output_list:
                output_list.append(element)
    return output_list


# Function to create consolidated dataset
def consolidated_data_pkl(folder_path, output_file):
    count = 0
    folders = os.listdir(folder_path)
    global label_name
    for folder in folders:
        label_name = extract_element_names(folder, label_name)
    with open(output_file, "w", newline="") as f_out:
        csv_writer = csv.writer(f_out)
        for file in folders:
            with open(os.path.join(folder_path, file), "rb") as f_in:
                print(file)
                data = pkl.load(f_in)
                for row in data:
                    if len(row) > 0:
                        output_row = normalize(row[0])
                        output_row = [float(val) * 100000 for val in output_row]
                        if isinstance(row[1], int):
                            output_row.append("[" + str(row[1]) + "]")
                        else:
                            output_row.append(np.array2string(row[1], separator=","))
                        csv_writer.writerow(output_row)
                        count = count + 1


# Function to splitting the dataset to training and validation set.
# input : consolidated csv file containing all spectrum with labels
def split_dataset(input_file):
    line_offsets = list()
    with open(input_file, "r") as f:
        title = f.readline()
        # store offset of the first
        while True:
            # store offset of the next line start
            line_offsets.append(f.tell())
            line = f.readline()
            if line == "":
                break

        # now shuffle the offsets
        random.shuffle(line_offsets)
        print("len :", len(line_offsets))

        # and write the output file
        with open("train_select.csv", "w") as fw:
            fw.write(title)
            for offset in line_offsets[
                0 : int(0.8 * len(line_offsets))
            ]:  # for offset in line_offsets[x*0.8]:
                f.seek(offset)
                fw.write(f.readline())

        with open("validation_select.csv", "w") as fw:
            for offset in line_offsets[int(0.8 * len(line_offsets)) : -1]:
                f.seek(offset)
                fw.write(f.readline())


"""
Data training : 
- Create mechanism to deal with large dataset. 
- Train and test model.
"""


def read_csv(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        count = 0

        for line in reader:
            features = [np.float64(n) for n in line[:no_of_bins]]
            label = ast.literal_eval(
                line[-1]
            )  # safely evaluate string as Python expression
            label = np.array(label, dtype=np.float32)
            features = np.array(features)
            features = features.reshape(no_of_bins, -1)
            # features = (
            #     features * 1000
            # )  # scale it up cuz the model can't recognize small values
            # yield features, tf.keras.utils.to_categorical(int(label), num_classes=no_of_class)

            yield features, label


def main():
    if True:
        folder_path = "C:/theCave/ISO-ID/data_prep/output_data/hamamatsu_capture_dataset_output/data"
        output_file = "C:/theCave/ISO-ID/data_prep/output_data/hamamatsu_capture_dataset_output/output.csv"

        # Merge data
        # consolidated_data_pkl(folder_path, output_file)
        print(label_name)
        print("....Finished merging dataset......")

        # Split data
        # split_dataset(output_file)
        print("....Finished splitting dataset......")

        # Read training dataset
        tf_ds_train = lambda: read_csv("C:/theCave/ISO-ID/train/train_select.csv")
        tf_ds_conc = lambda: read_csv("C:/theCave/ISO-ID/train/train_select.csv")

        # Read Validation dataset
        tf_ds_val = lambda: read_csv("C:/theCave/ISO-ID/train/validation_select.csv")
        val_ds_conc = lambda: read_csv("C:/theCave/ISO-ID/train/validation_select.csv")

        # Create Datasets using dataset generators
        dataset_train = tf.data.Dataset.from_generator(
            tf_ds_train,
            output_signature=(
                tf.TensorSpec(shape=(no_of_bins, 1), dtype=tf.dtypes.float64),
                tf.TensorSpec(shape=(no_of_class,)),
            ),
        )
        dataset_val = tf.data.Dataset.from_generator(
            tf_ds_val,
            output_signature=(
                tf.TensorSpec(shape=(no_of_bins, 1), dtype=tf.dtypes.float64),
                tf.TensorSpec(shape=(no_of_class,)),
            ),
        )
        dataset_train = dataset_train.shuffle(1000)
        dataset_train = dataset_train.batch(256).prefetch(3)
        dataset_val = dataset_val.batch(256).prefetch(3)

        # Perform hyperparameter tuning
        best_class_hps, best_conc_hps = hyperparameter_tuning(
            dataset_train, dataset_val
        )

        # Train the models using the best hyperparameters
        (
            trained_class_model,
            trained_conc_model,
            class_history,
            conc_history,
        ) = train_with_best_hyperparameters(
            dataset_train, dataset_val, best_class_hps, best_conc_hps
        )

        # Save the trained models
        dump(
            trained_class_model,
            "C:/theCave/ISO-ID/train/trained_models/cnn_class.joblib",
        )
        dump(
            trained_conc_model, "C:/theCave/ISO-ID/train/trained_models/cnn_conc.joblib"
        )

        # Plot model history
        # plot_model_history("Classification Model", class_history.history, 100)
        # plot_model_history("Concentration Model", conc_history.history, 100)

        print("Training finished")

    """
    Other classifiers 
    """
    if False:
        # initialize the classifiers
        knn = KNeighborsClassifier(n_neighbors=50)
        nb = GaussianNB()
        lr = LogisticRegression()
        svm_clf = SVC(kernel="linear")

        train_dataset = list(read_csv("C:/theCave/ISO-ID/train/train_select.csv"))
        train_features = np.array(
            [
                x[0].reshape(
                    no_of_bins,
                )
                for x in train_dataset
            ]
        )
        train_labels = np.array([x[1] for x in train_dataset])

        knn.fit(train_features, np.argmax(train_labels, axis=1))
        dump(knn, "C:/theCave/ISO-ID/train/trained_models/knn.joblib")
        nb.fit(train_features, np.argmax(train_labels, axis=1))
        dump(nb, "C:/theCave/ISO-ID/train/trained_models/nb.joblib")
        lr.fit(train_features, np.argmax(train_labels, axis=1))
        dump(lr, "C:/theCave/ISO-ID/train/trained_models/lr.joblib")
        svm_clf.fit(train_features, np.argmax(train_labels, axis=1))
        dump(svm_clf, "C:/theCave/ISO-ID/train/trained_models/svm.joblib")

        val_dataset = list(read_csv("C:/theCave/ISO-ID/train/validation_select.csv"))
        val_features = np.array(
            [
                x[0].reshape(
                    no_of_bins,
                )
                for x in val_dataset
            ]
        )
        val_labels = np.array([x[1] for x in val_dataset])

        knn_score = knn.score(val_features, np.argmax(val_labels, axis=1))
        nb_score = nb.score(val_features, np.argmax(val_labels, axis=1))
        lr_score = lr.score(val_features, np.argmax(val_labels, axis=1))
        svm_score = svm_clf.score(val_features, np.argmax(val_labels, axis=1))

        print("KNN score: ", knn_score)
        print("Naive Bayes score: ", nb_score)
        print("Logistic Regression score: ", lr_score)
        print("SVM score: ", svm_score)


if __name__ == "__main__":
    main()
