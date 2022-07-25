"""Feature engineers the customer churn dataset."""
import logging
import numpy as np
import pandas as pd
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("Starting preprocessing.")

    input_data_path = os.path.join("/opt/ml/processing/input", "train.csv")

    try:
        os.makedirs("/opt/ml/processing/train")
        os.makedirs("/opt/ml/processing/validation")
        os.makedirs("/opt/ml/processing/test")

    except:
        pass

    logger.info("Reading input data")

    # read csv
    model_data = pd.read_csv(input_data_path)


    # Split the data
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )

    train_data.to_csv("/opt/ml/processing/train/train.csv", index=False,encoding='utf-8')
    validation_data.to_csv(
        "/opt/ml/processing/validation/validation.csv", index=False,encoding='utf-8')
    test_data.to_csv(
        "/opt/ml/processing/test/test.csv", index=False,encoding='utf-8')