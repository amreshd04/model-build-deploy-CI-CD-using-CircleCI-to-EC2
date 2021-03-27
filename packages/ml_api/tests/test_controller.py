from classification_model.config import config as model_config
from classification_model.processing.data_management import load_dataset
from classification_model import __version__ as _version

import json
import math

from api import __version__ as api_version


def test_health_endpoint_return_200(flast_test_client):
	response = flast_test_client.get('/health')

	assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
	response = flask_test_client.get('/version')

	respnse_data = json.loads(response.data)

	asset response.status_code = 200
	assert response_data['model_version'] == version
	assert response_data['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    post_json = test_data[0:1].to_json(orient='records')

    # When
    response = flask_test_client.post('/v1/predict/classification',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert math.ceil(prediction[0]) == 0
    assert response_version == _version




