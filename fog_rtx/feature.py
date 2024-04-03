from typing import List, Optional, Tuple

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
        dimension: Optional[List[int]] = None,
        is_type_enforced=False,  # whether a type is enforced
    ) -> None:
        self.dtype = dtype
        self.dimension = dimension
        self.enforced = is_type_enforced

        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"

        if self.enforced and dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def __str__(self):
        return f"dtype={self.dtype}, dimension={self.dimension})"

    def __repr__(self):
        return self.__str__()
