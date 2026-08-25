"""Generic component registry for plug-in architecture."""

from __future__ import annotations

from typing import Any

_REGISTRIES: dict[str, dict[str, type]] = {}


def register(registry_name: str, name: str):
    """Class decorator that registers a component under *registry_name*/*name*.

    Usage::

        @register("loss", "my_loss")
        class MyLoss(nn.Module):
            ...
    """

    def decorator(cls: type) -> type:
        if registry_name not in _REGISTRIES:
            _REGISTRIES[registry_name] = {}
        if name in _REGISTRIES[registry_name]:
            raise ValueError(
                f"Duplicate registration: {registry_name}/{name} "
                f"(existing: {_REGISTRIES[registry_name][name]}, new: {cls})"
            )
        _REGISTRIES[registry_name][name] = cls
        return cls

    return decorator


def create(registry_name: str, name: str, **kwargs: Any):
    """Instantiate a registered component by *registry_name* and *name*."""
    if registry_name not in _REGISTRIES or name not in _REGISTRIES[registry_name]:
        available = list(_REGISTRIES.get(registry_name, {}).keys())
        raise KeyError(
            f"'{name}' not found in registry '{registry_name}'. "
            f"Available: {available}"
        )
    return _REGISTRIES[registry_name][name](**kwargs)


def list_registered(registry_name: str) -> list[str]:
    """Return names registered under *registry_name*."""
    return list(_REGISTRIES.get(registry_name, {}).keys())
