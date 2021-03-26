import numpy as np
import pandas as pd

from classification_model.processing.data_management import load_pipeline
from classification_model.config import config
from classification_model.processing.validation import validate_inputs
from classification_model import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_rf_pipe =  load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:

	data = pd.DataFrame(input_data)
	validated_data = validate_inputs(input_data=data)

	prediction = _rf_pipe.predict(validated_data)

	output = prediction

	results = {"predictions":output, "version": _version}

	_logger.info(
			f"Making predictions with model version: {_version} "
			f"Inputs: {validated_data} "
			f"Predictions: {results}"
		)

	return results



