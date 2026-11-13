# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layer plugin host: named model sites that an enabled plugin may serve.

A site is one span of a layer (``attn.out.wo_a..wo_b``) or one op ATOM calls
directly (``attn.index.topk``). The model calls it as
``plugin_host.call(site, native, *args, **kwargs)``: with no owner, or when
every owner's ``select`` declines, that is ``native(*args, **kwargs)`` and
nothing else. Ops AITER routes itself stay on ``aiter.plugin``; the same enable
list drives both (``configure``).

Rules:
- Default deny. Installing a plugin makes it available; only ``--plugin-enable``
  (``Config.plugin_enable``) enables it. ``ATOM_PLUGIN_HOST=0`` only disables.
- A site has ordered owners, meant for disjoint cells (shapes): the enable
  list's order is the order their ``select`` is asked in, and the first that
  accepts serves. ``ATOM_PLUGIN_HOST_STRICT=1`` asks every owner and raises
  when two accept one call. ``register`` checks the hook version, the site and
  its declared return names, and refuses with a reason.
- ``select`` is a pure metadata predicate: no host sync, no launch. Once
  ``impl`` runs, its errors propagate; there is no native retry.
- Traced code (``torch.compiler.is_compiling()``) always takes the native
  path, so a compiled graph never bakes a plugin in; the decision is recorded
  once per trace as ``("native", "traced")``. A site inside a compiled region
  is built with ``span_op``: inline when nothing serves it, else one opaque op
  whose body decides on every execution.
- Decisions, not launches, are counted: a CUDA-graph replay runs no Python.
"""

from __future__ import annotations

import functools
import logging
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType
from typing import Any

import torch

logger = logging.getLogger("atom")

HOOK_VERSION = 1
ATOM_HOOK_VERSION = HOOK_VERSION  # the spelling plugin templates probe
ENTRY_POINT_GROUP = "atom.layer_plugins"
AITER_SOURCE = "atom --plugin-enable"


@dataclass(frozen=True)
class Site:
    """What a site hands its owner and what it must give back.

    ``returns`` names the native result: a tuple of that length, a single
    value when it has one name, nothing checked when empty (in-place ops).
    """

    returns: tuple[str, ...]
    kind: str  # "span" (producer..consumer) or "op"
    target: str


# Hook version 1. A span site returns its consumer's input pair: the consumer
# (wo_b, w2, main_proj) stays ATOM's and reads ``x_scale=None`` as bf16 input.
SITES = MappingProxyType(
    {
        "attn.out.wo_a..wo_b": Site(
            ("y", "y_scale"), "span", "atom/models/deepseek_v41/attention.py"
        ),
        "moe.shared.act..down": Site(
            ("act", "act_scale"), "span", "atom/models/deepseek_v4/__init__.py"
        ),
        "draft.ctx.concat_quant..main_proj": Site(
            ("aux", "aux_scale"), "span", "atom/models/deepseek_v41/dspark.py"
        ),
        "attn.qk_norm": Site(
            ("q_fp8", "q_scale", "k_out"), "op", "atom/model_ops/layernorm.py"
        ),
        "attn.index.topk": Site((), "op", "atom/model_ops/indexer/score.py"),
        "attn.attend": Site(("out",), "op", "atom/model_ops/v4_kernels/paged_2src.py"),
        "moe.topk": Site((), "op", "atom/model_ops/moe.py"),
        "op.quantize_fp8_group32": Site(
            ("x_fp8", "x_scale"), "op", "atom/model_ops/blockscale.py"
        ),
    }
)


class RegistrationRefused(ValueError):
    """A registration that can never be valid; ``reason`` is its short code."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class PluginContractError(RuntimeError):
    """An owner's result does not have its site's declared shape."""


class PluginConflict(RuntimeError):
    """Strict mode: more than one owner accepted the same call."""


@dataclass(frozen=True)
class Registration:
    site: str
    owner: str
    accepted: bool
    reason: str


@dataclass(frozen=True)
class Decision:
    site: str
    route: str  # "plugin" or "native"
    reason: str  # selected, declined, select-error, traced
    owner: str | None = None  # the owner that served; native: None
    version: int | None = None
    declined: int = 0  # owners asked before it (all of them, when native)


@dataclass(frozen=True)
class _Route:
    owner: str
    impl: Callable[..., Any]
    select: Callable[..., Any] | None
    version: int
    arity: int


_lock = threading.RLock()
_enabled: tuple[str, ...] = ()
_kill_switch = False
_strict = False
_loading: str | None = None  # the entry point whose register() is running
# site -> its owners' routes in enable-list order. Replaced, never mutated, so
# a call reads one consistent snapshot.
_routes: MappingProxyType = MappingProxyType({})
_refusals: list[Registration] = []
_counts: Counter = Counter()
_last: ContextVar[Decision | None] = ContextVar("atom_plugin_decision", default=None)


def _refuse(site, owner, reason, detail):
    _refusals.append(Registration(site, owner, False, reason))
    logger.warning("plugin_host: %s refused for %s: %s", owner, site, detail)
    raise RegistrationRefused(reason, detail)


def register(
    site: str,
    impl: Callable[..., Any],
    *,
    owner: str,
    version: int,
    returns: Iterable[str],
    select: Callable[..., Any] | None = None,
) -> Registration:
    """Add ``owner`` to ``site``'s owners, in its enable-list place. Refusals
    that are bugs raise; an owner that is simply not enabled (or the kill
    switch) gets ``accepted=False``."""
    global _routes
    with _lock:
        spec = SITES.get(site)
        if spec is None:
            _refuse(site, owner, "unknown-site", f"hook v{HOOK_VERSION} has no {site}")
        if version != HOOK_VERSION:
            _refuse(site, owner, "version-mismatch", f"v{version} != v{HOOK_VERSION}")
        if tuple(returns) != spec.returns:
            _refuse(
                site, owner, "returns-mismatch", f"{tuple(returns)} != {spec.returns}"
            )
        if not callable(impl) or (select is not None and not callable(select)):
            _refuse(site, owner, "not-callable", "impl and select must be callable")
        if _loading is not None and owner != _loading:
            _refuse(site, owner, "owner-mismatch", f"loading entry point {_loading}")
        current = _routes.get(site, ())
        if any(route.owner == owner for route in current):
            _refuse(site, owner, "duplicate", f"{owner} already serves {site}")
        if _kill_switch or owner not in _enabled:
            reason = "kill-switch" if _kill_switch else "not-enabled"
            _refusals.append(Registration(site, owner, False, reason))
            return _refusals[-1]
        route = _Route(owner, impl, select, version, len(spec.returns))
        owners = sorted((*current, route), key=lambda r: _enabled.index(r.owner))
        _routes = MappingProxyType({**_routes, site: tuple(owners)})
        logger.info(
            "plugin_host: %s owners, in order: %s (hook v%d)",
            site,
            ", ".join(r.owner for r in owners),
            version,
        )
        return Registration(site, owner, True, "registered")


def unregister(site: str, *, owner: str) -> None:
    global _routes
    with _lock:
        rest = tuple(r for r in _routes.get(site, ()) if r.owner != owner)
        routes = {k: v for k, v in _routes.items() if k != site}
        _routes = MappingProxyType({**routes, site: rest} if rest else routes)


def _record(decision: Decision) -> None:
    _last.set(decision)
    _counts[(decision.site, decision.route, decision.reason, decision.owner)] += 1


def _record_traced(ctx) -> None:
    # Runs once, while Dynamo traces `call`, reading `call`'s own locals.
    site, owner = (ctx.get_local(n).as_python_constant() for n in ("site", "owner"))
    _record(Decision(site, "native", "traced", owner))


try:
    from torch._dynamo.comptime import comptime as _comptime
except Exception:  # noqa: BLE001 -- recording a traced decision is best effort
    _comptime = None


def _accepts(route: _Route, args, kwargs) -> bool | None:
    """The owner's answer; None when its select raised (a decline)."""
    if route.select is None:
        return True
    try:
        return bool(route.select(*args, **kwargs))
    except Exception as error:  # noqa: BLE001 -- a broken select is a decline
        logger.warning("plugin_host: %s select failed: %s", route.owner, error)
        return None


def _decide(site, routes, args, kwargs, strict) -> tuple[Decision, _Route | None]:
    """Ask the owners in order; the first that accepts serves. Strict asks
    them all and refuses a call two of them accept."""
    chosen, declined, errors = None, 0, 0
    for route in routes:
        answer = _accepts(route, args, kwargs)
        if not answer:
            errors += answer is None
            declined += chosen is None
            continue
        if chosen is not None:
            raise PluginConflict(
                f"{site}: {chosen.owner} and {route.owner} both accept this call"
            )
        chosen = route
        if not strict:
            break
    if chosen is None:
        reason = "select-error" if errors else "declined"
        return Decision(site, "native", reason, declined=declined), None
    decision = Decision(
        site, "plugin", "selected", chosen.owner, chosen.version, declined
    )
    return decision, chosen


def call(site: str, native: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``native(*args, **kwargs)``, unless one of the site's owners serves it."""
    routes = _routes.get(site)
    if routes is None:
        return native(*args, **kwargs)
    if torch.compiler.is_compiling():
        owner = ",".join(r.owner for r in routes)  # noqa: F841 -- _record_traced
        if _comptime is not None:
            _comptime(_record_traced)
        return native(*args, **kwargs)
    decision, route = _decide(site, routes, args, kwargs, _strict)
    _record(decision)
    if route is None:
        return native(*args, **kwargs)
    result = route.impl(*args, **kwargs)
    if route.arity > 1 and not (
        isinstance(result, tuple) and len(result) == route.arity
    ):
        raise PluginContractError(
            f"{route.owner} returned {type(result).__name__} for {site}, "
            f"expected {SITES[site].returns}"
        )
    return result


def site(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a whole function as the native of ``name``."""
    if name not in SITES:
        raise KeyError(f"hook v{HOOK_VERSION} has no site {name}")

    def decorate(native):
        @functools.wraps(native)
        def wrapper(*args, **kwargs):
            return call(name, native, *args, **kwargs)

        return wrapper

    return decorate


# Spans a compiled graph contains: name -> (run, fake). See `span_op`.
_spans: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {}
SPAN_OP = "atom_plugin::span"


def _span(x: torch.Tensor, extra: torch.Tensor | None, layer: str) -> torch.Tensor:
    # Through sys.modules: the op outlives a re-import of this module.
    return sys.modules[__name__]._spans[layer][0](x, extra)


def _span_fake(x: torch.Tensor, extra: torch.Tensor | None, layer: str) -> torch.Tensor:
    return sys.modules[__name__]._spans[layer][1](x, extra)


def _define_span_op() -> None:
    namespace, name = SPAN_OP.split("::")
    if hasattr(getattr(torch.ops, namespace), name):
        return
    library = torch.library.Library(namespace, "FRAGMENT")
    schema = torch.library.infer_schema(_span, mutates_args=())
    library.define(name + schema)
    library.impl(name, _span, "CompositeExplicitAutograd")
    library._register_fake(name, _span_fake)
    _define_span_op.library = library  # the registration lives as long as it


def span_op(
    site: str,
    layer: str,
    run: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    fake: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """At model build: how a traced module runs the span ``run(x, extra)``.

    Nothing serves ``site``: ``run`` itself, inline, so the compiled graph is
    the one ATOM has without a plugin. Any owner serves it: one opaque op
    (``atom_plugin::span``, shapes from ``fake``) whose body runs ``run``
    eagerly, where its ``call`` asks the owners on every execution -- Dynamo
    and a CUDA graph see the same op whichever owner serves.
    Decided once, after ``configure``, from the routes registered then.
    """
    if site not in _routes:
        return run
    if layer in _spans:
        raise ValueError(f"span layer name {layer!r} is not unique")
    _define_span_op()
    _spans[layer] = (run, fake)
    op = torch.ops.atom_plugin.span.default
    return lambda x, extra=None: op(x, extra, layer)


def explain(site: str, *args: Any, **kwargs: Any) -> Decision:
    """What ``call`` would decide for these arguments, without running
    anything; ``owners(site)`` is the order it asks them in."""
    routes = _routes.get(site)
    if routes is None:
        reason = "kill-switch" if _kill_switch else "no-owner"
        return Decision(site, "native", reason)
    return _decide(site, routes, args, kwargs, _strict)[0]


def owners(site: str) -> tuple[str, ...]:
    """The site's owners, in the order their ``select`` is asked."""
    return tuple(route.owner for route in _routes.get(site, ()))


def last_decision() -> Decision | None:
    return _last.get()


def decision_counts() -> Counter:
    """Decisions so far, keyed ``(site, route, reason, owner)``."""
    return Counter(_counts)


def status() -> dict[str, Any]:
    return {
        "hook_version": HOOK_VERSION,
        "kill_switch": _kill_switch,
        "strict": _strict,
        "enabled": _enabled,
        "sites": {name: owners(name) for name in _routes},
        "span_ops": tuple(_spans),
        "refusals": tuple(_refusals),
    }


def _entry_points() -> dict[str, metadata.EntryPoint]:
    return {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def configure(
    enable: Iterable[str] | str | None,
    *,
    kill_switch: bool | None = None,
    strict: bool | None = None,
):
    """Enable exactly ``enable`` (replacing any earlier call) in this process.

    Loads each enabled ``atom.layer_plugins`` entry point, in list order -- a
    callable ``register(host)`` that calls ``host.register`` for its sites --
    and hands the same list to ``aiter.plugin.configure`` when that interface
    exists. A plugin that fails to load is logged and skipped. ``kill_switch``
    and ``strict`` default to ATOM_PLUGIN_HOST=0 / ATOM_PLUGIN_HOST_STRICT=1.
    """
    global _enabled, _kill_switch, _strict, _routes, _loading
    if isinstance(enable, str):
        enable = enable.split(",")
    names = tuple(dict.fromkeys(n.strip() for n in enable or () if n.strip()))
    if kill_switch is None or strict is None:
        from atom.utils import envs

        kill_switch = not envs.ATOM_PLUGIN_HOST if kill_switch is None else kill_switch
        strict = envs.ATOM_PLUGIN_HOST_STRICT if strict is None else strict
    with _lock:
        _enabled, _kill_switch, _strict = names, kill_switch, strict
        _routes = MappingProxyType({})
        _refusals.clear()
        _spans.clear()  # built against the previous configuration
        if names and not kill_switch:
            available = _entry_points()
            for name in names:
                if name not in available:
                    continue  # an AITER-only provider, or not installed
                _loading = name
                try:
                    available[name].load()(sys.modules[__name__])
                except Exception as error:  # noqa: BLE001 -- isolate plugins
                    _refusals.append(Registration("*", name, False, "load-error"))
                    logger.warning("plugin_host: %s failed to load: %s", name, error)
                finally:
                    _loading = None
    # The one kill switch turns the AITER half off too.
    _configure_aiter(() if kill_switch else names)
    if names:
        logger.info("plugin_host: %s", status())


def _configure_aiter(names: tuple[str, ...]) -> bool:
    """The AITER half of the one enable list; False when AITER has no
    ``aiter.plugin`` (an older build), which leaves AITER routing untouched."""
    try:
        from aiter import plugin as aiter_plugin
    except Exception:  # noqa: BLE001 -- absent or broken: nothing to drive
        return False
    configure_aiter = getattr(aiter_plugin, "configure", None)
    if not callable(configure_aiter):
        return False
    configure_aiter(source=AITER_SOURCE, enable=list(names))
    return True
