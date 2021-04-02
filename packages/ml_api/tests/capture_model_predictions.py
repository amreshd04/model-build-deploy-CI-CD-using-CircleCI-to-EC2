import pandas as pd

from classification_model.predict import make_prediction
from classification_model.processing.data_management import load_dataset
from classification_model.config import config

from api import config as api_config

def capture_predictions(*, save_file:str = 'test_data_predictions.csv')	:
	test_data = load_dataset(file_name='test.csv')

	test_data.drop('id', axis=1, inplace=True)
	test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES]=test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES].astype(str)

	multiple_test_json = test_data[99:200]

	predictions = make_prediction(input_data=multiple_test_json)

	predictions_df = pd.DataFrame(predictions)

	predictions_df.to_csv(
		f'{api_config.PACKAGE_ROOT.parent}/'
		f'classification_model/classification_model/datasets/{save_file}')

if __name__ == '__main__':
	capture_predictions()


