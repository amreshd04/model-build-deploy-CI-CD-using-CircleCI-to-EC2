from marshmallow import Schema, fields
from marshmallow import ValidationError

import typing as t
import json


class InvalidInputError(Exception):
    """Invalid model input."""


class InsuranceDataRequestSchema(Schema):
	Gender = fields.str(allow_none=True)
	Age = fields.Integer()
	Driving_License = fields.Integer()
	Region_Code = fields.Float()
	Previously_Insured = fields.Integer()
	Vehicle_Age = fields.str()
	Vehicle_Damage = fields.str()
	Annual_Premium = fields.Float()
	Policy_Sales_Channel = fields.Float()
	Vintage = fields.Integer()



def _filter_error_rows(errors: dict, validated_input: t.List[dict]) -> t.List[dict]:

	indexes = errors.keys()

	for index in sorted(indexes, reverse=True):
		del validated_input[index]

	return validated_input

def validate_inputs(input_data):
    """Check prediction inputs against schema."""

    # set many=True to allow passing in a list
    schema = HouseDataRequestSchema(strict=True, many=True)

    errors = None
    try:
        schema.load(input_data)
    except ValidationError as exc:
        errors = exc.messages   

    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data

    return validated_input, errors




