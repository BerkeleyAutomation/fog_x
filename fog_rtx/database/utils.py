import datetime
import decimal

import numpy as np
import sqlalchemy  # type: ignore

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
