import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from custom_loss import class_loss, conc_loss

# Load the trained models
class_model = load("C:/theCave/ISO-ID/train/trained_models/cnn_class.joblib")
conc_model = load("C:/theCave/ISO-ID/train/trained_models/cnn_conc.joblib")

# Load the validation dataset
validation_dataset = pd.read_csv("C:/theCave/ISO-ID/train/validation_select.csv")

# Preprocess the validation dataset for prediction
validation_features = np.array(
    [x.reshape(-1, 1) for x in validation_dataset.iloc[:, :-1].values]
)
validation_labels = np.array([x for x in validation_dataset.iloc[:, -1].values])

class_model.compile(loss=class_loss)
conc_model.compile(loss=conc_loss)


# Make predictions using the trained models
class_predictions = class_model.predict(validation_features)
conc_predictions = conc_model.predict(validation_features)

# Visualize the results for a random validation sample
random_index = np.random.randint(0, len(validation_features))
spectrum = validation_features[random_index].reshape(-1)
true_label = validation_labels[random_index]
predicted_label = class_predictions[random_index]
concentrations = conc_predictions[random_index]

# Plot the spectrum
plt.plot(spectrum)
plt.title("Validation Spectrum")
plt.xlabel("Bin Index")
plt.ylabel("Intensity")
plt.show()

# Print the true label and predicted label
print("True Label (Isotope Class):", true_label)
print("Predicted Label (Isotope Class):", predicted_label)

# Print the predicted concentrations
print("Predicted Concentrations:", concentrations)

# Objective value for the prediction result (you can use any appropriate metric)
# Let's use mean squared error for the concentration predictions
mse = np.mean((true_label - concentrations) ** 2)
print("Mean Squared Error for Concentrations:", mse)
