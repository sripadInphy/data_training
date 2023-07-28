import numpy as np
import pandas as pd
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

# Visualize the results for all validation samples
num_samples = len(validation_features)

# Create a list to store the results for each spectrum
results_list = []
threshold = 0.6
for i in range(num_samples):
    spectrum = validation_features[i].reshape(-1)
    true_label = np.array(
        [
            float(val)
            for val in validation_labels[i].replace("[", "").replace("]", "").split(",")
        ]
    )
    predicted_label = class_predictions[i]
    concentrations = np.round(
        conc_predictions[i], 2
    )  # Round predicted values to two decimal places

    # Create a new list with zeros
    high_concentrations = np.zeros_like(concentrations)

    # Set the values to concentrations where predicted_label > threshold
    high_concentrations[predicted_label > threshold] = concentrations[
        predicted_label > threshold
    ]

    # Calculate the mean absolute error for the concentrations
    mae_concentration = np.mean(np.abs(true_label - high_concentrations))

    # Add the results to the list
    results_list.append(
        [
            i + 1,
            true_label.tolist(),
            [round(val, 2) for val in predicted_label],
            [round(val, 2) for val in high_concentrations],
            mae_concentration,
        ]
    )

# Create a DataFrame with the results and save it to a CSV file
result_df = pd.DataFrame(
    results_list,
    columns=[
        "Spectrum Number",
        "True Values",
        "Class prediction",
        "Conc Predictions",
        "Mean Absolute Error",
    ],
)
result_df.to_csv("model_test_report.csv", index=False)

# Print the DataFrame with results
print(result_df)
