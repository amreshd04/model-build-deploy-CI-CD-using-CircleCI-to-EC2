import pandas as pd

from classification_model.predict import make_prediction
from classification_model.processing.data_management import load_dataset

from api import config

def capture_predictions():
	save_file='test_data_predictions.csv'
	test_data = load_dataset(file_name='test.csv')

	multiple_test_input = test_data[99:200]

	predictions = make_prediction(input_data=multiple_test_input)

	predictions_df = pd.DataFrame(predictions)

	predictions_df.to_csv(f'{config.PACKAGE_ROOT}/{save_file}')

if __name__ == '__main__':
	capture_predictions()


