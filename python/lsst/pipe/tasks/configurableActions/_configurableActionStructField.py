# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

__all__ = ("ConfigurableActionStructField", "ConfigurableActionStruct")

from typing import Iterable, Mapping, Optional, TypeVar, Union, Type, Tuple, List, Any, Dict

from lsst.pex.config.config import Config, Field, FieldValidationError, _typeStr, _joinNamePath
from lsst.pex.config.comparison import compareConfigs, compareScalars, getComparisonName
from lsst.pex.config.callStack import StackFrame, getCallStack, getStackFrame

from . import ConfigurableAction


class ConfigurableActionStructUpdater:
    """This descriptor exists to abstract the logic of using a dictionary to
    update a ConfigurableActionStruct through attribute assignment. This is
    useful in the context of setting configuration through pipelines or on
    the command line.
    """
    def __set__(self, instance: ConfigurableActionStruct,
                value: Union[Mapping[str, ConfigurableAction], ConfigurableActionStruct]) -> None:
        if isinstance(value, Mapping):
            pass
        elif isinstance(value, ConfigurableActionStruct):
            # If the update target is a ConfigurableActionStruct, get the
            # internal dictionary
            value = value._attrs
        else:
            raise ValueError("Can only update a ConfigurableActionStruct with an instance of such, or a "
                             "mapping")
        for name, action in value.items():
            setattr(instance, name, action)

    def __get__(self, instance, objtype=None) -> None:
        # This descriptor does not support fetching any value
        return None


class ConfigurableActionStructRemover:
    """This descriptor exists to abstract the logic of removing an interable
    of action names from a ConfigurableActionStruct at one time using
    attribute assignment. This is useful in the context of setting
    configuration through pipelines or on the command line.

    Raises
    ------
    AttributeError
        Raised if an attribute specified for removal does not exist in the
        ConfigurableActionStruct
    """
    def __set__(self, instance: ConfigurableActionStruct,
                value: Union[str, Iterable[str]]) -> None:
        # strings are iterable, but not in the way that is intended. If a
        # single name is specified, turn it into a tuple before attempting
        # to remove the attribute
        if isinstance(value, str):
            value = (value, )
        for name in value:
            delattr(instance, name)

    def __get__(self, instance, objtype=None) -> None:
        # This descriptor does not support fetching any value
        return None


class ConfigurableActionStruct:
    """A ConfigurableActionStruct is the storage backend class that supports
    the ConfigurableActionStructField. This class should not be created
    directly.

    This class allows managing a collection of `ConfigurableActions` with a
    struct like interface, that is to say in an attribute like notation.

    Attributes can be dynamically added or removed as such:

    ConfigurableActionStructInstance.variable1 = a_configurable_action
    del ConfigurableActionStructInstance.variable1

    Each action is then available to be individually configured as a normal
    `lsst.pex.config.Config` object.

    ConfigurableActionStruct supports two special convenance attributes.

    The first is `update`. You may assign a dict of `ConfigurableActions` or
    a `ConfigurableActionStruct` to this attribute which will update the
    `ConfigurableActionStruct` on which the attribute is invoked such that it
    will be updated to contain the entries specified by the structure on the
    right hand side of the equals sign.

    The second convenience attribute is named remove. You may assign an
    iterable of strings which correspond to attribute names on the
    `ConfigurableActionStruct`. All of the corresponding attributes will then
    be removed. If any attribute does not exist, an `AttributeError` will be
    raised. Any attributes in the Iterable prior to the name which raises will
    have been removed from the `ConfigurableActionStruct`
    """
    # declare attributes that are set with __setattr__
    _config: Config
    _attrs: Dict[str, ConfigurableAction]
    _field: ConfigurableActionStructField
    _history: List[tuple]

    # create descriptors to handle special update and remove behavior
    update = ConfigurableActionStructUpdater()
    remove = ConfigurableActionStructRemover()

    def __init__(self, config: Config, field: ConfigurableActionStructField,
                 value: Mapping[str, ConfigurableAction], at: Any, label: str):
        object.__setattr__(self, '_config', config)
        object.__setattr__(self, '_attrs', {})
        object.__setattr__(self, '_field', field)
        object.__setattr__(self, '_history', [])

        self.history.append(("Struct initialized", at, label))

        if value is not None:
            for k, v in value.items():
                setattr(self, k, v)

    @property
    def history(self) -> List[tuple]:
        return self._history

    @property
    def fieldNames(self) -> Iterable[str]:
        return self._attrs.keys()

    def __setattr__(self, attr: str, value: Union[ConfigurableAction, Type[ConfigurableAction]],
                    at=None, label='setattr', setHistory=False) -> None:

        if hasattr(self._config, '_frozen') and self._config._frozen:
            msg = "Cannot modify a frozen Config. "\
                  f"Attempting to set item {attr} to value {value}"
            raise FieldValidationError(self._field, self._config, msg)

        if attr not in (self.__dict__.keys() | type(self).__dict__.keys()):
            name = _joinNamePath(self._config._name, self._field.name, attr)
            if at is None:
                at = getCallStack()
            if isinstance(value, ConfigurableAction):
                valueInst = type(value)(__name=name, __at=at, __label=label, **value._storage)
            else:
                valueInst = value(__name=name, __at=at, __label=label)
            self._attrs[attr] = valueInst
        else:
            super().__setattr__(attr, value)

    def __getattr__(self, attr):
        if attr in object.__getattribute__(self, '_attrs'):
            return self._attrs[attr]
        else:
            super().__getattribute__(attr)

    def __delattr__(self, name):
        if name in self._attrs:
            del self._attrs[name]
        else:
            super().__delattr__(name)

    def __iter__(self) -> Iterable[ConfigurableAction]:
        return iter(self._attrs.values())

    def items(self) -> Iterable[Tuple[str, ConfigurableAction]]:
        return iter(self._attrs.items())


T = TypeVar("T", bound="ConfigurableActionStructField")


class ConfigurableActionStructField(Field):
    r"""`ConfigurableActionStructField` is a `~lsst.pex.config.Field` subclass
    that allows `ConfigurableAction`\ s to be organized in a
    `~lsst.pex.config.Config` class in a manor similar to how a
    `~lsst.pipe.base.Struct` works.

    This class implements a `ConfigurableActionStruct` as an intermediary
    object to organize the `ConfigurableActions`. See it's documentation for
    futher information.
    """
    # specify StructClass to make this more generic for potential future
    # inheritance
    StructClass = ConfigurableActionStruct

    # Explicitly annotate these on the class, they are present in the base
    # class through injection, so type systems have trouble seeing them.
    name: str
    default: Optional[Mapping[str, ConfigurableAction]]

    def __init__(self, doc: str, default: Optional[Mapping[str, ConfigurableAction]] = None,
                 optional: bool = False,
                 deprecated=None):
        source = getStackFrame()
        self._setup(doc=doc, dtype=self.__class__, default=default, check=None,
                    optional=optional, source=source, deprecated=deprecated)

    def __set__(self, instance: Config,
                value: Union[None, Mapping[str, ConfigurableAction], ConfigurableActionStruct],
                at: Iterable[StackFrame] = None, label: str = 'assigment'):
        if instance._frozen:
            msg = "Cannot modify a frozen Config. "\
                  "Attempting to set field to value %s" % value
            raise FieldValidationError(self, instance, msg)

        if at is None:
            at = getCallStack()

        if value is None or value == self.default:
            value = self.StructClass(instance, self, value, at=at, label=label)
        else:
            history = instance._history.setdefault(self.name, [])
            history.append((value, at, label))

        if not isinstance(value, ConfigurableActionStruct):
            raise FieldValidationError(self, instance,
                                       "Can only assign things that are subclasses of Configurable Action")
        instance._storage[self.name] = value

    def __get__(self: T, instance: Config, owner: None = None, at: Iterable[StackFrame] = None,
                label: str = "default"
                ) -> Union[None, T, ConfigurableActionStruct]:
        if instance is None or not isinstance(instance, Config):
            return self
        else:
            field: Optional[ConfigurableActionStruct] = instance._storage[self.name]
            return field

    def rename(self, instance: Config):
        actionStruct: ConfigurableActionStruct = self.__get__(instance)
        if actionStruct is not None:
            for k, v in actionStruct.items():
                fullname = _joinNamePath(instance._name, self.name, k)
                v._rename(fullname)

    def validate(self, instance):
        value = self.__get__(instance)
        if value is not None:
            for item in value:
                item.validate()

    def toDict(self, instance):
        actionStruct = self.__get__(instance)
        if actionStruct is None:
            return None

        dict_ = {k: v.toDict() for k, v in actionStruct.items()}

        return dict_

    def save(self, outfile, instance):
        actionStruct = self.__get__(instance)
        fullname = _joinNamePath(instance._name, self.name)
        if actionStruct is None:
            outfile.write(u"{}={!r}\n".format(fullname, actionStruct))
            return

        outfile.write(u"{}={!r}\n".format(fullname, {}))
        for v in actionStruct:
            outfile.write(u"{}={}()\n".format(v._name, _typeStr(v)))
            v._save(outfile)

    def freeze(self, instance):
        actionStruct = self.__get__(instance)
        if actionStruct is not None:
            for v in actionStruct:
                v.freeze()

    def _compare(self, instance1, instance2, shortcut, rtol, atol, output):
        """Compare two fields for equality.

        Parameters
        ----------
        instance1 : `lsst.pex.config.Config`
            Left-hand side config instance to compare.
        instance2 : `lsst.pex.config.Config`
            Right-hand side config instance to compare.
        shortcut : `bool`
            If `True`, this function returns as soon as an inequality if found.
        rtol : `float`
            Relative tolerance for floating point comparisons.
        atol : `float`
            Absolute tolerance for floating point comparisons.
        output : callable
            A callable that takes a string, used (possibly repeatedly) to
            report inequalities.

        Returns
        -------
        isEqual : bool
            `True` if the fields are equal, `False` otherwise.

        Notes
        -----
        Floating point comparisons are performed by `numpy.allclose`.
        """
        d1: ConfigurableActionStruct = getattr(instance1, self.name)
        d2: ConfigurableActionStruct = getattr(instance2, self.name)
        name = getComparisonName(
            _joinNamePath(instance1._name, self.name),
            _joinNamePath(instance2._name, self.name)
        )
        if not compareScalars(f"keys for {name}", set(d1.fieldNames), set(d2.fieldNames), output=output):
            return False
        equal = True
        for k, v1 in d1.items():
            v2 = getattr(d2, k)
            result = compareConfigs(f"{name}.{k}", v1, v2, shortcut=shortcut,
                                    rtol=rtol, atol=atol, output=output)
            if not result and shortcut:
                return False
            equal = equal and result
        return equal
