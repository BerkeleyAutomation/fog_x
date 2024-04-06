import datetime
import decimal

import numpy as np
import pyarrow as pa
import sqlalchemy  # type: ignore
from polars import datatypes as pld

_type_py2sql_dict = {
    int: sqlalchemy.sql.sqltypes.BigInteger,
    str: sqlalchemy.sql.sqltypes.Unicode,
    float: sqlalchemy.sql.sqltypes.Float,
    decimal.Decimal: sqlalchemy.sql.sqltypes.Numeric,
    datetime.datetime: sqlalchemy.sql.sqltypes.DateTime,
    bytes: sqlalchemy.sql.sqltypes.LargeBinary,
    bool: sqlalchemy.sql.sqltypes.Boolean,
    datetime.date: sqlalchemy.sql.sqltypes.Date,
    datetime.time: sqlalchemy.sql.sqltypes.Time,
    datetime.timedelta: sqlalchemy.sql.sqltypes.Interval,
    list: sqlalchemy.sql.sqltypes.ARRAY,
    dict: sqlalchemy.sql.sqltypes.JSON,
}


def type_py2sql(pytype):
    """Return the closest sql type for a given python type"""
    if pytype in _type_py2sql_dict:
        return _type_py2sql_dict[pytype]
    else:
        raise NotImplementedError(
            "You may add custom `sqltype` to `"
            + str(pytype)
            + "` assignment in `_type_py2sql_dict`."
        )


def type_np2py(dtype=None, arr=None):
    """Return the closest python type for a given numpy dtype"""

    if (dtype is None and arr is None) or (
        dtype is not None and arr is not None
    ):
        raise ValueError(
            "Provide either keyword argument `dtype` or `arr`: a numpy dtype or a numpy array."
        )

    if dtype is None:
        dtype = arr.dtype

    # 1) Make a single-entry numpy array of the same dtype
    # 2) force the array into a python 'object' dtype
    # 3) the array entry should now be the closest python type
    single_entry = np.empty([1], dtype=dtype).astype(object)

    return type(single_entry[0])


def type_np2sql(dtype=None, arr=None):
    """Return the closest sql type for a given numpy dtype"""
    return type_py2sql(type_np2py(dtype=dtype, arr=arr))


def _datasets_dtype_to_arrow(datasets_dtype: str) -> pa.DataType:
    """
    _datasets_dtype_to_arrow takes a datasets string dtype and converts it to a pyarrow.DataType.
    In effect, `dt == datasets_dtype_to_arrow(_datasets_dtype_to_arrow(dt))`
    """

    if datasets_dtype == "null":
        return pa.null()
    elif datasets_dtype == "bool":
        return pa.bool_()
    elif datasets_dtype == "int8":
        return pa.int8()
    elif datasets_dtype == "int16":
        return pa.int16()
    elif datasets_dtype == "int32":
        return pa.int32()
    elif datasets_dtype == "int64":
        return pa.int64()
    elif datasets_dtype == "uint8":
        return pa.uint8()
    elif datasets_dtype == "uint16":
        return pa.uint16()
    elif datasets_dtype == "uint32":
        return pa.uint32()
    elif datasets_dtype == "uint64":
        return pa.uint64()
    elif datasets_dtype == "float16":
        return pa.float16()
    elif datasets_dtype == "float32":
        return pa.float32()
    elif datasets_dtype == "float64":
        return pa.float64()
    elif datasets_dtype == "binary":
        return pa.binary()
    elif datasets_dtype == "large_binary":
        return pa.large_binary()
    elif datasets_dtype == "string":
        return pa.string()
    elif datasets_dtype == "str":
        return pa.string()
    elif datasets_dtype == "large_string":
        return pa.large_string()
    elif datasets_dtype.startswith("timestamp"):
        # Extracting unit and timezone from the string
        unit, tz = None, None
        if "[" in datasets_dtype and "]" in datasets_dtype:
            unit = datasets_dtype[
                datasets_dtype.find("[") + 1 : datasets_dtype.find("]")
            ]
            tz_start = datasets_dtype.find("tz=")
            if tz_start != -1:
                tz = datasets_dtype[tz_start + 3 :]
        return pa.timestamp(unit, tz)
    else:
        raise ValueError(
            f"Datasets dtype {datasets_dtype} does not have a PyArrow dtype equivalent."
        )


def _datasets_dtype_to_pld(datasets_dtype: str) -> pld.DataType:
    """
    _datasets_dtype_to_pld takes a datasets string dtype and converts it to a polars.DataType.
    In effect, `dt == _datasets_dtype_to_pld(_pld_to_datasets_dtype(dt))` (assuming _pld_to_datasets_dtype is the reverse mapping).
    """

    if datasets_dtype == "null":
        return pld.Null
    elif datasets_dtype == "bool":
        return pld.Boolean
    elif datasets_dtype == "int8":
        return pld.Int8
    elif datasets_dtype == "int16":
        return pld.Int16
    elif datasets_dtype == "int32":
        return pld.Int32
    elif datasets_dtype == "int64":
        return pld.Int64
    elif datasets_dtype == "uint8":
        return pld.UInt8
    elif datasets_dtype == "uint16":
        return pld.UInt16
    elif datasets_dtype == "uint32":
        return pld.UInt32
    elif datasets_dtype == "uint64":
        return pld.UInt64
    elif datasets_dtype == "float16":
        # Note: Polars does not have a native float16 type. This will be mapped to float32.
        return pld.Float32
    elif datasets_dtype == "float32":
        return pld.Float32
    elif datasets_dtype == "float64":
        return pld.Float64
    elif datasets_dtype == "binary":
        return pld.Binary
    elif datasets_dtype == "large_binary":
        # Polars does not differentiate between binary and large_binary
        return pld.Binary
    elif (
        datasets_dtype == "string"
        or datasets_dtype == "str"
        or datasets_dtype == "large_string"
    ):
        # Polars treats all strings as Utf8
        return pld.Utf8
    elif datasets_dtype == "object":
        return pld.Object
    elif datasets_dtype.startswith("timestamp"):
        # Handling timestamps in Polars is a bit different. You might need additional handling based on your use case.
        return pld.Datetime
    else:
        raise ValueError(
            f"Datasets dtype {datasets_dtype} does not have a Polars dtype equivalent."
        )
