
import csv
import os

def log_features_to_csv(filename, features):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=features[0].keys())
        if not file_exists:
            writer.writeheader()  # Write the header only once
        for feature in features:
            writer.writerow(feature)