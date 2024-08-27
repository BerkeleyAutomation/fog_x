
from typing import Any, Dict
import numpy as np
from fog_x.feature import FeatureType


def data_to_tf_schema(data: Dict[str, Any]) -> Dict[str, FeatureType]:
    """
    Convert data to a tf schema
    """
    schema = {}
    for k, v in data.items():
        if "/" in k: # make the subkey to be within dict 
            main_key, sub_key = k.split("/")
            if main_key not in schema:
                schema[main_key] = {}
            schema[main_key][sub_key] = FeatureType.from_data(v).to_tf_feature_type()
        else:
            schema[k] = FeatureType.from_data(v).to_tf_feature_type()
    return schema