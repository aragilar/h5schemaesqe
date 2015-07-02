import numpy as np

import hdf5_wrapper

class SolnBase(hdf5_wrapper.HDF5Wrapper, filetype="Soln"):
    pass

class SolnV1(SolnBase, version="1.0"):
    a = {"b": int, "c": str}
    s = np.ndarray
