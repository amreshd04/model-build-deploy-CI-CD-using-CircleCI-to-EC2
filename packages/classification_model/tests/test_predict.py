import math
import numpy as np
from classification_model.predict import make_prediction
from classification_model.config import config
from classification_model.processing.data_management import load_dataset


def test_single_prediction():
	test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
	test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES]=test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES].astype(str)
	single_test_input = test_data[0:1]

	subject = make_prediction(input_data=single_test_input[config.FEATURES])

	assert subject is not None		
	assert isinstance(subject.get('predictions')[0], np.int64)
	assert math.ceil(subject.get('predictions')[0] == 0)


def test_multiple_predictions():
	# Given
	test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
	test_data.drop('id', axis=1, inplace=True)
	test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES]=test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES].astype(str)

	original_length = len(test_data)
	multiple_test_input = test_data

	# When
	subject = make_prediction(input_data=multiple_test_input)

	# Then
	assert subject is not None
	#print(multiple_test_input)
	#print(original_length)
	#print(subject)
	assert len(subject.get('predictions')) == 127037
	#assert len(subject.get('predictions')) != original_length




