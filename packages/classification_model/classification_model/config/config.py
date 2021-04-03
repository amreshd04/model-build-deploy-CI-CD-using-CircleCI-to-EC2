import pathlib

import classification_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = "Response"

FEATURES = ["Gender","Age","Driving_License","Region_Code","Previously_Insured","Vehicle_Age","Vehicle_Damage","Annual_Premium","Policy_Sales_Channel","Vintage"]

DISCRETE_SET1_FEATURES = ['Age', 'Region_Code']
DISCRETE_SET2_FEATURES = ['Driving_License', 'Previously_Insured']
DISCRETE_SET3_FEATURES = ['Policy_Sales_Channel', 'Vintage']

CONTINUOUS_FEATURES = ["Annual_Premium"]

CATEGORICAL_FEATURES = ["Gender", "Vehicle_Age", "Vehicle_Damage"]

PIPELINE_NAME = "rf_model"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

ACCEPTABLE_MODEL_DIFFERENCE = 1


