"""
"""
from collections import namedtuple
from collections.abc import (
        MutableMapping, MutableSequence, Iterable, Sequence, Mapping
)
import numbers

from numpy import ndarray

NO_ITEM_IN_GROUP = "No item in group called {}"

def valid_index(index, length):
    index, length = int(index), int(length)
    if index >= length:
        return False
    if index < - length:
        return False
    return True


class HDF5GroupBase:
    def __init__(
        self, group_name, group_mapping, parent=None, hdf5_file=None,
        inherit_namedtuple=False
    ):
        if parent is None and hdf5_file is None:
            raise RuntimeError("Not bound to an hdf5 file")

        self._name = group_name
        self._parent = parent
        self._hdf5_file = hdf5_file or self._parent._hdf5_file
        self._group_mapping = group_mapping

        self._set_named_tuple(inherit_namedtuple)
        self._update_group_path()
        self._use_dataset = [ndarray]
        self._use_attrs = [int, str, float]

    def _update_group_path(self):
        if self._parent is None:
            if self._name != "root":
                raise RuntimeError("Top level group should be called root")
            self._group_path = "/"
        else:
            self._group_path = self._parent._group_path + "/" + self._name

    def _set_named_tuple(self, inherit_namedtuple):
        if inherit_namedtuple:
            self._named_tuple = self._parent._named_tuple
        else:
            try:
                self._named_tuple = namedtuple(
                    self._name, ' '.join(self._group_mapping.keys())
                )
            except AttributeError:
                self._named_tuple = None


    def _get_item_from_file(self, path, name, objtype):
        if isinstance(objtype, (Mapping, Sequence)):
            return self._get_item_group(path, name)
        elif objtype in self._use_dataset:
            return objtype(self._get_item_dataset(path, name))
        else:
            return objtype(self._get_item_attrs(path, name))


    def _set_item_in_file(self, path, name, obj, objtype):
        if isinstance(objtype, dict):
            self._set_group_map(path, name, obj)
        elif isinstance(objtype, list):
            self._set_multi_group(path, name, obj)
        elif objtype in self._use_dataset:
            self._set_item_dataset(path, name, obj)
        elif objtype in self._use_attrs:
            self._set_item_attrs(path, name, obj)
        else:
            raise TypeError("Type cannot be converted to hdf5 type")

    def _get_item_group(self, path, name):
        if hasattr(self, "_children"):
            return self._children[name]
        else:
            return self._instances[int(name)]

    def _get_item_dataset(self, path, name):
        return self._hdf5_file[path + "/" + name]

    def _get_item_attrs(self, path, name):
        return self._hdf5_file[path].attrs[name]

    def _set_item_dataset(self, path, name, obj):
        group = self._hdf5_file.require_group(path)
        group[name] = obj

    def _set_item_attrs(self, path, name, obj):
        group = self._hdf5_file.require_group(path)
        group.attrs[name] = obj

    def _set_group_map(self, path, name, obj):
        child = self[str(name)]
        if isinstance(obj, child._named_tuple):
            child.update(**vars(obj))
        else:
            raise TypeError("Not a valid definition of {}".format(name))

    def _set_multi_group(self, path, name, obj):
        child = self[str(name)]
        if isinstance(obj, Mapping):
            for key, val in obj.items():
                child[key] = val
        elif isinstance(obj, Iterable):
            for i, item in enumerate(obj):
                child[i] = item
        else:
            raise TypeError("Not a valid definition of {}".format(name))


class HDF5GroupMap(MutableMapping, HDF5GroupBase):
    def __init__(
        self, group_name, group_mapping, parent=None, hdf5_file=None,
        inherit_namedtuple=False
    ):
        super().__init__(
            group_name, group_mapping, parent=parent, hdf5_file=hdf5_file,
            inherit_namedtuple=inherit_namedtuple
        )
        self._children = {}

        for name, func in self._group_mapping.items():
            if isinstance(func, dict):
                self._children[name] = HDF5GroupMap(name, func, parent=self)
            elif isinstance(func, list):
                self._children[name] = HDF5MultiGroup(name, func[0], parent=self)

    def __getattr__(self, name):
        try:
            if name in self.__dict__:
                return self.__dict__[name]
            elif name in self._group_mapping:
                return self[name]
            raise AttributeError(NO_ITEM_IN_GROUP.format(name))
        except KeyError:
            raise AttributeError(NO_ITEM_IN_GROUP.format(name))

    def __setattr__(self, name, item):
        try:
            if "_group_mapping" not in self.__dict__:
                self.__dict__[name] = item
            elif name in self._group_mapping:
                self[name] = item
            else:
                self.__dict__[name] = item
        except KeyError:
            raise AttributeError(NO_ITEM_IN_GROUP.format(name))

    def _get_namedtuples_from_subgroups(self):
        subgroup_namedtuples = [
            nt
            for group in self._children.values()
            for nt in group._get_namedtuples_from_subgroups()
        ]
        if self._named_tuple is not None:
            subgroup_namedtuples.append(self._named_tuple)
        return subgroup_namedtuples

    def __getitem__(self, name):
        if name in self._group_mapping:
            return self._get_item_from_file(
                self._group_path, name, self._group_mapping[name]
            )
        raise KeyError(NO_ITEM_IN_GROUP.format(name))

    def __setitem__(self, name, item):
        if name in self._group_mapping:
            self._set_item_in_file(
                self._group_path, name, item, self._group_mapping[name]
            )
        else:
            raise KeyError(NO_ITEM_IN_GROUP.format(name))

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        return iter(self._group_mapping)

    def __delitem__(self, item):
        raise NotImplemented

class HDF5MultiGroup(MutableSequence, HDF5GroupBase):
    def __init__(
        self, group_name, group_mapping, parent=None, hdf5_file=None,
        inherit_namedtuple=False
    ):
        super().__init__(
            group_name, group_mapping, parent=parent, hdf5_file=hdf5_file,
            inherit_namedtuple=inherit_namedtuple
        )
        self._instances = []
        self._get_instances()

    def _get_instances(self):
        group = self._hdf5_file.get(self._group_path)
        if group is not None:
            for index in sorted(group):
                self._create_new_instance(index)

    def _create_new_instance(self, index):
        def new_instance(self, index):
            if isinstance(self._group_mapping, dict):
                self._instances.append(HDF5GroupMap(
                    str(index), self._group_mapping, parent=self,
                    inherit_namedtuple=True
                ))
            elif isinstance(self._group_mapping, list):
                self._instances.append(HDF5MultiGroup(
                    str(index), self._group_mapping[0], parent=self,
                    inherit_namedtuple=False
                ))
            else:
                raise RuntimeError("Invalid definition of file structure")

        if index >= len(self):
            new_instance(self, len(self))

        else:
            for i, instance in reverse(
                enumerate(self._instances[index:], index + 1)
            ):
                new_instance(self, i)
                self[str(i)] = instance
            new_instance(self, index)

    def __getitem__(self, index):
        if valid_index(index, len(self)):
            return self._get_item_from_file(
                self._group_path, index, self._group_mapping[index]
            )
        raise IndexError("Out of range")

    def __setitem__(self, index, item):
        if isinstance(index, numbers.Integral):
            self._set_item_in_file(
                self._group_path, index, item,
                self._group_mapping
            )
        elif isinstance(index, slice):
            for i, obj in zip(range(*slice.indices(self._instances)), item):
                self._set_item_in_file(
                    self._group_path, i, obj,
                    self._group_mapping
                )
        raise TypeError("Must be integer or slice")

    def __delitem__(self, item):
        raise NotImplemented

    def __len__(self):
        return len(self._instances)

    def insert(self, index, item):
        self._create_new_instance(index)
        self[index] = item

    def _get_namedtuples_from_subgroups(self):
        try:
            subgroup_namedtuples = \
                self._instances[0]._get_namedtuples_from_subgroups()
        except IndexError:
            if isinstance(self._group_mapping, dict):
                child = HDF5GroupMap(
                    "invalid", self._group_mapping, parent=self,
                    inherit_namedtuple=True
                )
            elif isinstance(self._group_mapping, list):
                child = HDF5MultiGroup(
                    "invalid", self._group_mapping[0], parent=self,
                    inherit_namedtuple=False
                )
            else:
                raise RuntimeError("Invalid definition of file structure")
            subgroup_namedtuples = child._get_namedtuples_from_subgroups()
        if self._named_tuple is not None:
            subgroup_namedtuples.append(self._named_tuple)
        return subgroup_namedtuples

class HDF5FileProxy:
    def __init__(self):
        self._file = None
    def __getitem__(self, name):
        return self._file[name]
    def __setitem__(self, name, value):
        self._file[name] = value

    def get(self, key, default=None):
        if self._file is None:
            return default
        return self._file.get(key, default)

    def require_group(self, *args, **kwargs):
        return self._file.require_group(*args, **kwargs)

    @property
    def fileobj(self):
        return self._file

    @fileobj.setter
    def fileobj(self, f):
        self._file = f


def class_list_to_map(class_list):
    class_map = {}
    for cls in class_list:
        name = cls.__name__
        if name in class_map:
            if cls._fields == class_map[name]._fields:
                continue
            raise RuntimeError("Duplicate names")
        class_map[name] = cls
    return class_map

def hdf5_schema_to_class(schema, version, filetype):
    class HDF5File:
        def __init__(self, **kwargs):
            self._file = HDF5FileProxy()
            self._root_group = HDF5GroupMap("root", schema, hdf5_file=self._file)
            self._named_tuples = class_list_to_map(
                self._root_group._get_namedtuples_from_subgroups()
            )

        @property
        def fileobj(self):
            return self._file.fileobj

        @fileobj.setter
        def fileobj(self, f):
            self._file.fileobj = f
            self._file.fileobj.attrs["version"] = version
            self._file.fileobj.attrs["filetype"] = filetype

        @property
        def root(self):
            return self._root_group

        @property
        def named_tuples(self):
            return self._named_tuples


        @property
        def version(self):
            """
            Solution file version
            """
            return self.fileobj.attrs.get("version")

        @property
        def filetype(self):
            """
            Solution file version
            """
            return self.fileobj.attrs.get("filetype")
    return HDF5File

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
