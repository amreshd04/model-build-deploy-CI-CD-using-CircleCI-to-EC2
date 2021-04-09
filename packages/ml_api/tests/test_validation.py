import sys,os
import json
from classification_model.config import config
from classification_model.processing.data_management import load_dataset


def test_prediction_endpoint(flask_test_client):
    # Given    
    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
    test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES] = test_data[config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+config.DISCRETE_SET3_FEATURES].astype(str)
    post_json = test_data[config.FEATURES].to_json(orient='records')

    input_data_sent_to_model = json.loads(post_json)[0:500]
    # When
    response = flask_test_client.post('/v1/predict/classification',
                                      json=input_data_sent_to_model)

    # Then   
    assert response.status_code == 200
    assert len(input_data_sent_to_model) == len(json.loads(response.data)['predictions'])
    assert json.loads(response.data)['errors'] == None    
