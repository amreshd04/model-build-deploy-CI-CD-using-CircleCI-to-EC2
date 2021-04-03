import math
import pytest

from classification_model.config import config as config
from classification_model.predict import make_prediction
from classification_model.processing.data_management import load_dataset
from api import config as api_config
import pandas as pd


@pytest.mark.skip
#@pytest.mark.differential
def test_model_prediction_differential(*, save_file='test_data_predictions.csv'):
	previous_model_df = pd.read_csv(f'{api_config.PACKAGE_ROOT}/{save_file}')

	previous_model_predictions = previous_model_df.predictions.values
	print('previous predictions:',previous_model_predictions)

	test_data = load_dataset(file_name=config.TESTING_DATA_FILE)

	test_data.drop('id', axis=1, inplace=True)
	test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES]=test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES].astype(str)

	multiple_test_input = test_data[0:200]

	current_result=make_prediction(input_data=multiple_test_input)
	current_model_predictions = current_result.get('predictions')
	print('current predictions:', current_model_predictions)

	assert len(previous_model_predictions) == len(current_model_predictions)


	for previous_value, current_value in zip(previous_model_predictions, current_model_predictions):
		previous_value = previous_value.item()
		current_value = current_value.item()

		assert math.isclose(previous_value, current_value, rel_tol=1)

