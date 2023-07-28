import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pandas as pd


@tf.function
def create_grad_cam_map(model, input_data, class_index):
    # Convert input_data to tensor
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Watch the input_data to compute gradients
    input_data = tf.Variable(input_data, dtype=tf.float32, trainable=True)

    # Get the target class score from the model's output
    class_output = model(input_data)
    class_score = class_output[0][class_index]

    # Get the gradient of the target class score with respect to the convolutional layer output
    conv_output = model.get_layer(
        "dense_2"
    ).output  # Replace "conv1d" with the correct layer name
    with tf.GradientTape() as tape:
        grads = tape.gradient(class_score, conv_output)

    # Check if gradients are not None
    if grads is None:
        raise ValueError(
            "Gradient is None. Ensure that the model architecture is correct."
        )

    # Calculate the channel importance weights (global average pooling of gradients)
    channel_importance = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)

    # Perform the weighted sum of convolutional layer output using channel importance weights
    grad_cam_map = tf.reduce_sum(tf.multiply(conv_output, channel_importance), axis=-1)

    # Normalize the Grad-CAM map
    grad_cam_map = tf.nn.relu(grad_cam_map)

    # Rescale the Grad-CAM map to the input data size
    grad_cam_map = tf.image.resize(grad_cam_map, input_data.shape[1:3])

    return grad_cam_map.numpy()


# Function to create the saliency map
def create_saliency_map(model, input_data):
    # Convert input_data to tensor
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Watch the input_data to compute gradients
    input_data = tf.Variable(input_data, dtype=tf.float32, trainable=True)

    with tf.GradientTape() as tape:
        predictions = model(input_data)

    # Get the gradients of the predictions with respect to the input_data
    grads = tape.gradient(predictions, input_data)

    # Calculate the absolute saliency values
    saliency = tf.reduce_sum(tf.abs(grads), axis=-1)

    # Normalize the saliency values
    saliency /= tf.reduce_max(saliency)

    return saliency.numpy()


# Function to define the class loss
def class_loss(y_true, y_pred):
    class_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    tf.print("Class loss : ", class_loss)
    return class_loss


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


def main():
    # Load the model from the joblib file
    class_model = joblib.load(
        "C:/theCave/ISO-ID/train/state_of_the_art/combinational_hamm_9_isotopes_9_june/cnn_class.joblib"
    )

    class_model.compile(loss=class_loss)
    # Get model summary
    print("Model Summary:")
    class_model.summary()

    # Load the test data (single spectrum) from the CSV file
    filename = "C:/theCave/ISO-ID/python_script/testbook.csv"
    df = pd.read_csv(filename)
    test_data = df.iloc[:, 7][:1500].values  # Extracting the values as a NumPy array
    test_data = normalize(test_data)
    test_data = np.array(test_data) * 100000
    # Preprocess the data
    test_data = test_data.reshape(
        1, -1, 1
    )  # Add batch dimension and convert to (1, 1500, 1)
    test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)  # Convert to float32

    # Predict on the test data
    predictions = class_model.predict(test_data)
    print(predictions)

    # Create saliency map for the single spectrum
    saliency_map = create_saliency_map(class_model, test_data)

    # Visualize the saliency map as a 1D plot
    plt.plot(
        saliency_map[0]
    )  # Accessing the 1D array from the batch dimension (shape: (1, 1500, 1))
    plt.title("Saliency Map for Single Spectrum")
    plt.xlabel("Data Point Index")
    plt.ylabel("Saliency")
    plt.show()

    # Get the predicted class index (you need to replace this with the actual class index)

    class_index = np.argmax(predictions[0])

    # Create and visualize the Grad-CAM map
    grad_cam_map = create_grad_cam_map(class_model, test_data, class_index)
    plt.plot(grad_cam_map[0])
    plt.title("Grad-CAM Map for Single Spectrum")
    plt.xlabel("Data Point Index")
    plt.ylabel("Grad-CAM Score")
    plt.show()


if __name__ == "__main__":
    main()
