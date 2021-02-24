from __future__ import annotations
import inspect

__all__ = ["ConfigurableActionsField", "ConfigurableAction"]

import ast

from dataclasses import dataclass
from typing import Iterable, Mapping, Union, Type, Any

from lsst.pex.config.config import Config, Field, FieldValidationError, _typeStr, _joinNamePath
from lsst.pex.config.comparison import compareConfigs, compareScalars, getComparisonName
from lsst.pex.config.callStack import getCallStack, getStackFrame


class ConfigurableAction(Config):
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError("This method should be overloaded in subclasses")


@dataclass
class ProxyConfigurableAction:
    """This exists as a workaround for behavior in configOverrides.py:144 in
    pipe_base. If this gets moved outside of this package, this should likely
    be removed and the logic there changed. This arises as this is the first
    non Config class option that we allow assignment for in Pipelines.
    """
    proxy: ConfigurableAction

    @property
    def dtype(self):
        return str


class ConfigurableActionStruct:
    def __init__(self, config, field, value, at, label):
        object.__setattr__(self, '_config', config)
        object.__setattr__(self, '_attrs', {})
        object.__setattr__(self, '_field', field)
        object.__setattr__(self, '_history', [])

        self.history.append(("Struct initialized", at, label))

        if value is not None:
            for k, v in value.items():
                setattr(self, k, v)

    @property
    def _fields(self):
        # This exists as a workaround, see the note in the ProxyConfig class
        return {name: ProxyConfigurableAction(value) for name, value in self._attrs.items()}

    @property
    def history(self):
        return self._history

    @property
    def fieldNames(self) -> Iterable[str]:
        return self._attrs.keys()

    @staticmethod
    def _get_variables_in_scope():
        vars = {}
        stack = inspect.stack()
        for s in stack:
            vars.update(s.frame.f_globals)
        return vars

    def __setattr__(self, attr: str, value: Union[str, Type[ConfigurableAction]],
                    at=None, label='setattr', setHistory=False) -> None:

        if hasattr(self._config, '_frozen') and self._config._frozen:
            msg = "Cannot modify a frozen Config. "\
                  f"Attempting to set item {attr} to value {value}"
            raise FieldValidationError(self._field, self._config, msg)

        if isinstance(value, str):
            vars = self._get_variables_in_scope()
            if value in vars and issubclass(vars[value], ConfigurableAction):
                value = vars[value]
            else:
                msg = (f"The string {vars[value]} does not correspond to a ConfigurableAction subclass, "
                       "or does not exist")
                raise FieldValidationError(self._field, self._config, msg)

        if attr not in self.__dict__ and issubclass(value, ConfigurableAction):
            name = _joinNamePath(self._config._name, self._field.name, attr)
            if at is None:
                at = getCallStack()
            self._attrs[attr] = value(__name=name, __at=at, __label=label)
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

    def __iter__(self):
        yield from self._attrs.items()


class ConfigurableActionsField(Field):

    StructClass = ConfigurableActionStruct

    def __init__(self, doc, default=None, optional=False, deprecated=None):
        source = getStackFrame()
        self._setup(doc=doc, dtype=self.__class__, default=default, check=None,
                    optional=optional, source=source, deprecated=deprecated)

    def __set__(self, instance, value: Union[None, str, dict, ConfigurableActionStruct],
                at=None, label='assigment'):
        if instance._frozen:
            msg = "Cannot modify a frozen Config. "\
                  "Attempting to set field to value %s" % value
            raise FieldValidationError(self, instance, msg)

        if at is None:
            at = getCallStack()

        if value is None or value is self.default:
            value = self.StructClass(instance, self, value, at=at, label=label)
        else:
            history = instance._history.setdefault(self.name, [])
            history.append((value, at, label))

        if isinstance(value, str):
            if not isinstance(value := ast.literal_eval(value), Mapping):
                raise FieldValidationError(self, instance,
                                           "Only strings that are of the form python dicts are allowed")
        if isinstance(value, Mapping):
            existing = self.__get__(instance)
            for k, v in value.items():
                setattr(existing, k, v)
            return

        if not isinstance(value, ConfigurableActionStruct):
            raise FieldValidationError(self, instance,
                                       "Can only assign things that are subclasses of Configurable Action")
        instance._storage[self.name] = value

    def __get__(self, instance, owner=None, at=None, label="default"
                ) -> Union[StructClass, ConfigurableActionStruct]:
        if instance is None or not isinstance(instance, Config):
            return self
        else:
            return instance._storage[self.name]

    def rename(self, instance):
        actionStruct: ConfigurableActionStruct = self.__get__(instance)
        if actionStruct is not None:
            for k, v in actionStruct:
                fullname = _joinNamePath(instance._name, self.name, k)
                v._rename(fullname)

    def validate(self, instance):
        value = self.__get__(instance)
        if value is not None:
            for k, item in value:
                item.validate()

    def toDict(self, instance):
        actionStruct = self.__get__(instance)
        if actionStruct is None:
            return None

        dict_ = {}
        for k, v in actionStruct:
            dict_[k] = v.toDict()

        return dict_

    def save(self, outfile, instance):
        actionStruct = self.__get__(instance)
        fullname = _joinNamePath(instance._name, self.name)
        if actionStruct is None:
            outfile.write(u"{}={!r}\n".format(fullname, actionStruct))
            return

        outfile.write(u"{}={!r}\n".format(fullname, {}))
        for _, v in actionStruct:
            outfile.write(u"{}={}()\n".format(v._name, _typeStr(v)))
            v._save(outfile)

    def freeze(self, instance):
        actionStruct = self.__get__(instance)
        if actionStruct is not None:
            for _, v in actionStruct:
                v.freeze()

    def _compare(self, instance1, instance2, shortcut, rtol, atol, output):
        """Compare two fields for equality.

        Used by `lsst.pex.ConfigStructField.compare`.

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
        d1 = getattr(instance1, self.name)
        d2 = getattr(instance2, self.name)
        name = getComparisonName(
            _joinNamePath(instance1._name, self.name),
            _joinNamePath(instance2._name, self.name)
        )
        if not compareScalars(f"keys for {name}", set(d1.fieldNames), set(d2.fieldNames), output=output):
            return False
        equal = True
        for k, v1 in d1:
            v2 = getattr(d2, k)
            result = compareConfigs(f"{name}.{k}", v1, v2, shortcut=shortcut,
                                    rtol=rtol, atol=atol, output=output)
            if not result and shortcut:
                return False
            equal = equal and result
        return equal
