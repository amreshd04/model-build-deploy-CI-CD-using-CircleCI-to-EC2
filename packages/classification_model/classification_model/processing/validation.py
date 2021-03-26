from classification_model.config import config
import pandas as pd

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:

	validated_data = input_data.copy()


	if input_data[config.CONTINUOUS_FEATURES].isnull().any().any():
		validated_data = validated_data.dropna(axis=0, subset=config.CONTINUOUS_FEATURES)


	if input_data[config.CATEGORICAL_FEATURES].isnull().any().any():
		validated_data = validated_data.dropna(axis=0, subset=config.CATEGORICAL_FEATURES)

	if input_data[config.DISCRETE_SET1_FEATURES + config.DISCRETE_SET2_FEATURES +
	 			 config.DISCRETE_SET3_FEATURES].isnull().any().any():
	 			 validated_data = validated_data.dropna(axis=0, subset=config.DISCRETE_SET1_FEATURES + config.DISCRETE_SET2_FEATURES +
	 			 config.DISCRETE_SET3_FEATURES)

	return validated_data