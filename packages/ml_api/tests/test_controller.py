import sys,os
import json
import math
from classification_model.processing.data_management import load_dataset
from classification_model import __version__ as _version
from classification_model.config import config
from api import __version__ as api_version


def test_health_endpoint(flask_test_client):
	response = flask_test_client.get('/health')

	assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    # When
    response = flask_test_client.get('/version')

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version


def test_prediction_endpoint(flask_test_client):
    # Given    
    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
    test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES] = test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES].astype(str)
    post_json = test_data[config.FEATURES].to_json(orient='records')

    input_data_sent_to_model = json.loads(post_json)[0:1]

    # When
    response = flask_test_client.post('/v1/predict/classification',
                                      json=input_data_sent_to_model)

    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']

    # Then
    assert response.status_code == 200    
    assert math.ceil(prediction[0]) == 0
    assert response_version == _version





