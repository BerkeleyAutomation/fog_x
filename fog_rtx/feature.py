from typing import List, Optional, Tuple
from sqlalchemy import Integer, String, LargeBinary, Float
from fog_rtx.database.utils import type_py2sql, type_np2sql

SUPPORTED_DTYPES = [
    "null",
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "timestamp(s)",
    "timestamp(ms)",
    "timestamp(us)",
    "timestamp(ns)",
    "timestamp(s, tz)",
    "timestamp(ms, tz)",
    "timestamp(us, tz)",
    "timestamp(ns, tz)",
    "binary",
    "large_binary",
    "string",
    "large_string",
]


class FeatureType:
    """
    class for feature definition and conversions
    """

    def __init__(
        self,
        dtype: Optional[str] = None,
        shape: Optional[List[int]] = None,
        is_type_enforced=False,  # whether a type is enforced
    ) -> None:
        self.dtype = dtype
        self.shape = shape
        self.enforced = is_type_enforced

        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"

        if self.enforced and dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        self.np_dtype = None

    def __str__(self):
        return f"dtype={self.dtype}, shape={self.shape})"

    def __repr__(self):
        return self.__str__()

    def from_tf_feature_type(self, feature_type):
        """
        Convert from tf feature
        """
        self.shape = feature_type.shape
        self.np_dtype = feature_type.np_dtype
        self.dtype = str(self.np_dtype)
        return self


    def to_tf_feature_type(self): 
        """
        Convert to tf feature
        """
        pass

    def to_sql_type(self):
        """
        Convert to sql type
        """
        if self.shape is not None:
            return LargeBinary
        else: # single value 
            return type_np2sql(self.dtype)