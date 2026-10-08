import collections
import collections.abc

for _name in ("Iterable", "Container", "Mapping", "MutableMapping", "Sequence"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
