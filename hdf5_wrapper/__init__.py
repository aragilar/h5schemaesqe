"""
Wrapper around h5py and hdf5 to manage different versions of output files
scripts
"""

from collections import namedtuple
from collections.abc import Mapping
from types import SimpleNamespace

from numpy import array, ndarray
import h5py


### From python 3.4 enum ###
def _is_descriptor(obj):
    """Returns True if obj is a descriptor, False otherwise."""
    return (
            hasattr(obj, '__get__') or
            hasattr(obj, '__set__') or
            hasattr(obj, '__delete__'))


def _is_dunder(name):
    """Returns True if a __dunder__ name, False otherwise."""
    return (name[:2] == name[-2:] == '__' and
            name[2:3] != '_' and
            name[-3:-2] != '_' and
            len(name) > 4)


def _is_sunder(name):
    """Returns True if a _sunder_ name, False otherwise."""
    return (name[0] == name[-1] == '_' and
            name[1:2] != '_' and
            name[-2:-1] != '_' and
            len(name) > 2)


class _HDF5WrapperDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema = {}
    def __setitem__(self, key, value):
        if _is_sunder(key):
            raise ValueError('_names_ are reserved')
        elif _is_dunder(key):
            pass
        elif not _is_descriptor(value):
            self._schema[key] = value
        super().__setitem__(key, value)


def _bases_filetype(bases):
    return [
        base for base in bases
        if hasattr(base, "_filetype_") and base._filetype_ is not None
    ]

def _file_group_mapper(key, schema, named_tuple):
    key_schema = schema[key]
    def getter(self):
        converted_values = {}
        for name, objtype in key_schema.items():
            if objtype == ndarray:
                converted_values[name] = array(self._file_[key][name])
            else:
                converted_values[name] = objtype(self._file_[key].attrs[name])
        return named_tuple(**converted_values)

    def setter(self, mapping):
        if self._file_.get(key) is None:
            self._file_.create_group(key)
        for name, value in vars(mapping).items():
            objtype = key_schema[name]
            if objtype == ndarray:
                self._file_[key][name] = array(value)
            else:
                self._file_[key].attrs[name] = objtype(value)

    return getter, setter


def _file_single_mapper(key, objtype):
    def getter(self):
        if objtype == ndarray:
            return array(self._file_[key])
        else:
            return objtype(self._file_.attrs[key])
    def setter(self, value):
        if objtype == ndarray:
            self._file_[key] = array(value)
        else:
            self._file_.attrs[key] = objtype(value)
    return getter, setter


class HDF5WrapperMeta(type):
    """
    Base class for hdf5 wrappers
    """
    @classmethod
    def __prepare__(
        metacls, clsname, bases, filetype=None, version=None,
        **kwargs
    ):
        if filetype is not None:
            return {
                "_filetype_": filetype,
                "_fileversions_": {},
                "_extra_metadata_": {}
            }
        elif version is not None:
            bases = _bases_filetype(bases)
            if not bases:
                raise RuntimeError("Need to subclass from a base filetype")
            elif len(bases) > 1:
                raise RuntimeError("Too many mixed filetypes")
            base = bases[0]
            if version in base._fileversions_:
                raise RuntimeError("Version already exists")
            return _HDF5WrapperDict(
                _filetype_ = base._filetype_,
                named_tuples = SimpleNamespace(),
                _single_dataset_types_ = {},
                _version_ = version,
            )
        return {}

    def __new__(
        metacls, clsname, bases, attrs, filetype=None, version=None, **kwargs
    ):
        if hasattr(attrs, "_schema"):
            schema = {key: attrs._schema[key] for key in attrs._schema}
            for key in attrs._schema:
                del attrs[key]
            named_tuples = attrs["named_tuples"]
            single_dataset_types = attrs["_single_dataset_types_"]
            for key in schema:
                if isinstance(schema[key], Mapping):
                    setattr(
                        named_tuples, key, namedtuple(key, list(schema[key]))
                    )
                    getter, setter = _file_group_mapper(
                        key, schema, getattr(named_tuples,key))
                else:
                    objtype = schema[key]
                    single_dataset_types[key] = objtype
                    getter, setter = _file_single_mapper(key, objtype)
                attrs[key] = property(getter, setter)

        for key, val in kwargs.items():
            getter, setter = _file_single_mapper(key, type(val))
            attrs[key] = property(getter, setter)
            attrs["_extra_metadata_"][key] = val

        return super().__new__(metacls, clsname, bases, attrs)

    def __init__(cls, name, bases, attrs, **kwargs):
        cls_version = kwargs.get("version")
        if cls_version is not None:
            _bases_filetype(bases)[-1]._fileversions_[cls_version] = cls
        super().__init__(name, bases, attrs)


class HDF5Wrapper(metaclass=HDF5WrapperMeta):
    _file_ = None
    _filename_ = None
    _require_close_ = True
    _new_file_ = True
    def __init__(self, f, _new_file=None, **kwargs):
        if isinstance(f, h5py.File):
            self._file_ = f
            self._filename_ = f.filename
            self._require_close_ = False
            if _new_file is not None:
                self._new_file_ = _new_file
            else:
                self._new_file_ = self._is_new_file_(f)
        elif hasattr(f, "name"):
            self._filename_ = f.name
            if _new_file is not None:
                self._new_file_ = _new_file
        else:
            self._filename_ = f
            if _new_file is not None:
                self._new_file_ = _new_file
        self._kwargs_ = kwargs

    def __enter__(self):
        if self._file_ is None:
            self._file_ = h5py.File(self._filename_, **self._kwargs_)
            if self._new_file_:
                self._file_.attrs["version"] = self._version_
                self._file_.attrs["filetype"] = self._filetype_
                for key, val in self._extra_metadata_.items():
                    setattr(self, key, val)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._require_close_:
            return self._file_.__exit__(exc_type, exc_value, traceback)
        return False  # Don't handle any exceptions

    @classmethod
    def open(cls, filename, **kwargs):
        with h5py.File(filename) as f:
            if not cls._is_new_file_(f):
                if not cls._filetype_ == f.attrs["filetype"]:
                    raise RuntimeError(
                        "file is not of type {}".format(cls._filetype_)
                    )
                version = f.attrs["version"]
                if version not in cls._fileversions_:
                    raise RuntimeError(
                        "unknown file version {}".format(version)
                    )
                return cls._fileversions_[version](
                    filename, _new_file=False, **kwargs
                )
            return cls.newest()(filename, **kwargs)

    @classmethod
    def newest(cls):
        return cls._fileversions_[max(cls._fileversions_)]

    @classmethod
    def oldest(cls):
        return cls._fileversions_[min(cls._fileversions_)]


    @staticmethod
    def _is_new_file_(f):
        """
        Check if `f` is an empty/new hdf5 file
        """
        if not f.attrs.keys() and not f.keys():
            return True
        return False

    @property
    def version(self):
        """
        Solution file version
        """
        if self._file_ is None:
            with h5py.File(self._filename_, **self._kwargs_) as f:
                return f.attrs.get("version")
        return self._file_.attrs.get("version")

    @property
    def name(self):
        """
        Name of file on filesystem
        """
        return self._filename_

    @property
    def filetype(self):
        return self._filetype_

__all__ = ["HDF5Wrapper"]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
