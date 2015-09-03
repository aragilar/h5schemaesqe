"""
Wrapper around h5py/hdf5 to provide a schema-like definition of groups and
datasets
"""

from collections import namedtuple
from collections.abc import (
    MutableMapping, Iterable, Mapping
)

from pathlib import PurePosixPath

NO_ITEM_IN_GROUP = "No item in group called {}"


def add_attrs(*names):
    def wrapper(cls):
        def getattr_func(self, name):
            for map_name in names:
                if name in self.__dict__[map_name]:
                    return self.__dict__[map_name][name]
            raise AttributeError("No such attribute {}".format(name))

        def setattr_func(self, name, item):
            if name in names:
                self.__dict__[name] = item
            used_map_names = [
                map_name for map_name in names
                if self.__dict__.get(map_name) is not None
            ]

            for map_name in used_map_names:
                if name in self.__dict__[map_name]:
                    self.__dict__[map_name][name] = item
                    break
            else:
                self.__dict__[name] = item

        cls.__getattr__ = getattr_func
        cls.__setattr__ = setattr_func

        return cls

    return wrapper


def get_wrapper(schema):
    """
    Get correct group class based on schema.
    """
    if isinstance(schema, HDF5Group):
        return GroupWrapper
    elif isinstance(schema, HDF5MultiGroup):
        return MultiGroupWrapper
    else:
        TypeError("Invalid schema type")


class BaseHDF5Object:
    """
    Base class for schema components
    """
    pass


class BaseHDF5Group(BaseHDF5Object, Mapping):
    """
    Base class for groups in schema
    """
    _children = {}

    def generate_namedtuples(self, name):
        """
        Generate namedtuples based on schema
        """
        raise NotImplementedError("Subclass must implement this")

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, name):
        return self._children[name]


class HDF5Group(BaseHDF5Group):
    """
    Schema representation of an HDF5 group
    """
    def __init__(self, **kwargs):
        self._children = kwargs

    def generate_namedtuples(self, name):
        """
        Generate namedtuples based on schema
        """
        for child_name, child in self._children.items():
            if isinstance(child, BaseHDF5Group):
                yield from child.generate_namedtuples(child_name)
        yield namedtuple(name, ' '.join(self._children))


class HDF5MultiGroup(BaseHDF5Group):
    """
    Schema representation of an HDF5 group which contains arbitrary many groups
    with the name structure
    """
    def __init__(self, namedtuple_name, group):
        self._namedtuple_name = namedtuple_name
        self._children[self._namedtuple_name] = group

    def generate_namedtuples(self, name):
        """
        Generate namedtuples based on schema
        """
        for child_name, child in self._children.items():
            if isinstance(child, BaseHDF5Group):
                yield from child.generate_namedtuples(child_name)


class HDF5Link(BaseHDF5Object):
    """
    Schema representation of a HDF5 link
    """
    pass


class NamedtupleNamespace:
    """
    Create a namespace containing the namedtuples from a schema
    """
    def __init__(self, schema):
        for nt in schema.generate_namedtuples("root"):
            self.__dict__[nt.__name__] = nt


class HDF5File:
    """
    Wrapper around h5py.File to use schemas
    """
    def __init__(self, schema, namedtuples, file):
        self._schema = schema
        self._namedtuples = namedtuples
        self._file = file

        self._namedtuples_dict = vars(self._namedtuples)

        self._root = get_wrapper(self._schema)(
            "root", self._schema, self._file, self._namedtuples_dict
        )

    @property
    def file(self):
        """
        The actual h5py file object
        """
        return self._file

    @property
    def namedtuples(self):
        """
        Namedtuples associated with this file's schema
        """
        return self._namedtuples

    @property
    def root(self):
        """
        The root group
        """
        return self._root

    @root.setter
    def root(self, item):
        """
        The root group
        """
        if isinstance(item, self._namedtuples_dict["root"]):
            self._root.update(**vars(item))
        else:
            raise TypeError("Not a valid object")


class HDF5Path(PurePosixPath):
    """
    Representation of an HDF5 path
    """
    def shared_path(self, other):
        """
        Return the shared component between two HDF5 paths
        """
        path = []
        for self_part, other_part in zip(self.parts, other.parts):
            if self_part != other_part:
                break
            path.append(self_part)
        return HDF5Path(path)

    def __str__(self):
        return "/" + "/".join(self.parts)


@add_attrs("_schema")
class BaseGroupWrapper(MutableMapping):
    """
    Base class for wrappers around HDF5 groups
    """
    def __init__(self, name, schema, file, namedtuples, parent=None):

        self._name = name
        self._parent = parent
        self._file = file
        self._schema = schema
        self._use_attrs = [int, str, float]
        self._children = {}
        self._namedtuples = namedtuples

        if self._parent is None:
            self._path = HDF5Path("/")
        else:
            self._path = HDF5Path(self._parent._path, self._name)

    def _get_item_from_file(self, path, name, objtype):
        """
        Get object from file
        """
        if isinstance(objtype, BaseHDF5Group):
            return self._get_group(path, name)
        elif isinstance(objtype, HDF5Link):
            return self._get_link(path, name)
        elif objtype in self._use_attrs:
            return objtype(self._get_attr(path, name))
        else:
            return objtype(self._get_dataset(path, name))

    def _set_item_in_file(self, path, name, obj, objtype):
        """
        Write object to file
        """
        if isinstance(objtype, HDF5Group):
            self._set_group(path, name, obj)
        elif isinstance(objtype, HDF5MultiGroup):
            self._set_multi_group(path, name, obj)
        elif isinstance(objtype, HDF5Link):
            self._set_link(path, name, obj)
        elif objtype in self._use_attrs:
            self._set_attr(path, name, obj)
        else:
            self._set_dataset(path, name, obj)

    def _get_group(self, path, name):
        """
        Get wrapper around group based on file contents
        """
        return self._children[name]

    def _get_dataset(self, path, name):
        """
        Get dataset converted correctly
        """
        return self._file[str(HDF5Path(path, name))]

    def _get_attr(self, path, name):
        """
        Get attr converted correctly
        """
        return self._file[str(path)].attrs[name]

    def _get_link(self, path, name):
        """
        Get wrapper around link
        """
        link_path = HDF5Path(path, name)
        actual_path = HDF5Path(self._file[link_path].name)
        common_path = link_path.shared_path(actual_path)
        ancestor = self._get_ancestor(common_path)
        return ancestor._get_descendant(actual_path)

    def _get_ancestor(self, path):
        """
        Find ancestor based on path
        """
        if self._path == path:
            return self
        elif self._parent is None:
            raise RuntimeError("No such ancestor")
        return self._parent._get_ancestor(path)

    def _get_descendant(self, path):
        """
        Find descendant based on path
        """
        if self._path == path:
            return self
        return self[path.parts[len(self._path.parts)]]._get_descendant(path)

    def _set_dataset(self, path, name, obj):
        """
        Set dataset
        """
        group = self._file.require_group(str(path))
        group[name] = obj

    def _set_attr(self, path, name, obj):
        """
        Set attribute
        """
        group = self._file.require_group(str(path))
        group.attrs[name] = obj

    def _set_group(self, path, name, obj):
        """
        Set group
        """
        child = self[str(name)]
        if isinstance(obj, child._namedtuple):
            child.update(**vars(obj))
        else:
            raise TypeError("Not a valid definition of {}".format(name))

    def _set_multi_group(self, path, name, obj):
        """
        Set multi group
        """
        child = self[str(name)]
        if isinstance(obj, Mapping):
            for key, val in obj.items():
                child[key] = val
        elif isinstance(obj, Iterable):
            for i, item in enumerate(obj):
                child[i] = item
        else:
            raise TypeError("Not a valid definition of {}".format(name))

    def _set_link(self, path, name, obj):
        """
        Set link
        """
        if isinstance(obj, BaseHDF5Group):
            self._file[str(HDF5Path(path, name))] = self._file[str(obj._path)]
        else:
            raise TypeError("Not a valid definition of {}".format(name))

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        return iter(self._schema)

    def __delitem__(self, item):
        raise NotImplementedError("Need to create this method")


class GroupWrapper(BaseGroupWrapper):
    """
    Wrapper around HDF5 groups matching a schema
    """
    def __init__(
        self, name, schema, file, namedtuples, parent=None,
        namedtuple_name=None
    ):
        super().__init__(name, schema, file, namedtuples, parent=parent)

        for name, subschema in self._schema.items():
            if isinstance(subschema, BaseHDF5Group):
                self._children[name] = get_wrapper(subschema)(
                    name, subschema, file, namedtuples, parent=self
                )

        self._namedtuple_name = namedtuple_name or self._name
        self._namedtuple = self._namedtuples[self._namedtuple_name]

    def __getitem__(self, name):
        if name in self._schema:
            return self._get_item_from_file(
                self._path, name, self._schema[name]
            )
        raise KeyError(NO_ITEM_IN_GROUP.format(name))

    def __setitem__(self, name, item):
        if name in self._schema:
            self._set_item_in_file(
                self._path, name, item, self._schema[name]
            )
        else:
            raise KeyError(NO_ITEM_IN_GROUP.format(name))


class MultiGroupWrapper(BaseGroupWrapper):
    """
    Wrapper around HDF5 groups which contain multiple groups matching a schema
    """
    def __init__(
        self, name, schema, file, namedtuples, parent=None,
        namedtuple_name=None
    ):
        super().__init__(name, schema, file, namedtuples, parent=parent)

        self._namedtuple_name = self._schema._namedtuple_name
        self._subschema = self._schema[self._namedtuple_name]
        self._subgroup_cls = get_wrapper(self._subschema)
        self._namedtuple = self._namedtuples[self._namedtuple_name]

        for name in self._file.get(str(self._path), []):
            self._children[name] = self._subgroup_cls(
                name, self._subschema, file, namedtuples, parent=self
            )

    def __getitem__(self, name):
        return self._children[name]

    def __setitem__(self, name, item):
        if isinstance(item, self._namedtuple):
            self._children[name] = self._subgroup_cls(
                name, self._subschema, self._file, self._namedtuples,
                parent=self, namedtuple_name=self._namedtuple_name
            )
            self._children[name].update(**vars(item))
        else:
            raise TypeError("Not a valid object")


__all__ = [
    "HDF5Link", "HDF5Group", "HDF5MultiGroup", "HDF5File",
    "NamedtupleNamespace"
]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
