import tensorflow as tf


def class_loss(y_true, y_pred):
    y_true_binary = tf.where(
        tf.not_equal(y_true, 0), tf.constant(1, dtype=tf.float32), y_true
    )
    class_loss = tf.keras.losses.BinaryCrossentropy()(
        y_true_binary, y_pred
    )  # be carefull in what loss function are you using
    return 10 * class_loss


def conc_loss(y_true, y_pred):
    nonzero_labels = tf.where(tf.not_equal(y_true[:, :], 0))
    row_indices = nonzero_labels[:, 0]
    col_indices = nonzero_labels[:, 1]
    y_true_nonzero = tf.gather_nd(y_true[:, :], nonzero_labels)
    # Calculate the concentration loss for the predicted isotopes
    concentration_loss = tf.reduce_sum(
        tf.abs(
            tf.gather_nd(y_pred[:, :], nonzero_labels)
            - tf.gather_nd(y_true[:, :], nonzero_labels)
        )
        * y_true_nonzero
    ) / tf.reduce_sum(
        y_true[:, :]
    )  # Calculate weighted concentration loss

    # concentration_loss = tf.keras.losses.MeanSquaredError()(tf.gather_nd(y_pred[:, :], nonzero_labels), tf.gather_nd(y_true[:, :], nonzero_labels))
    # tf.print("Concen loss : ",concentration_loss)
    return 10 * concentration_loss
