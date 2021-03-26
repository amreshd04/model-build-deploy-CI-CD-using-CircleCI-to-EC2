from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from classification_model.processing import preprocessors as pp
from classification_model.processing import features
from classification_model.config import config

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder, CountFrequencyEncoder
from feature_engine.discretisation import EqualFrequencyDiscretiser

import logging

_logger = logging.getLogger(__name__)

rf_pipe = Pipeline(
[
    ('numeric_impute', MeanMedianImputer(imputation_method='median', variables=config.CONTINUOUS_FEATURES)),
    
    ('categorical_impute', CategoricalImputer(imputation_method='missing', 
                                              variables=config.CATEGORICAL_FEATURES+
                                              config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+
                                              config.DISCRETE_SET3_FEATURES)),
    
    ('rare_label_encode', RareLabelEncoder(tol=0.02, n_categories=10,
                                           variables=config.CATEGORICAL_FEATURES+
                                              config.DISCRETE_SET1_FEATURES+config.DISCRETE_SET2_FEATURES+
                                              config.DISCRETE_SET3_FEATURES,
                                            replace_with='Rare')),
    
    ('categorical_encode1', OrdinalEncoder(encoding_method='arbitrary', 
                                          variables=config.CATEGORICAL_FEATURES+config.DISCRETE_SET2_FEATURES)),
    
    ('categorical_encode2', OrdinalEncoder(encoding_method='ordered', 
                                          variables=config.DISCRETE_SET1_FEATURES)),
    
    ('categorical_encode3', CountFrequencyEncoder(encoding_method='count',
                                          variables=config.DISCRETE_SET3_FEATURES)),
    
    ('continuous_discretization', EqualFrequencyDiscretiser(q=20, variables=config.CONTINUOUS_FEATURES, return_object=True)),
    
    ('continuous_encoding', OrdinalEncoder(encoding_method='ordered', variables=config.CONTINUOUS_FEATURES)),
    
    ('scaling', StandardScaler()),
        
    ('clf', RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=10, random_state=0))    
])