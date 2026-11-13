# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The layer plugin host (`atom/plugin_host.py`) and its V4.1 call sites.

CPU only. The host itself imports nothing but torch; the call-site tests load
the model modules against the AITER stand-ins and drive one span through its
native and its plugin route.
"""

import ast
import functools
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs

from atom import plugin_host as host

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WO = "attn.out.wo_a..wo_b"
TOPK = "attn.index.topk"
CONFIGURE_AITER = host._configure_aiter  # before the fixture stands it in


@pytest.fixture(autouse=True)
def fresh_host(monkeypatch):
    """Default deny, no entry points, AITER left alone, empty record."""
    calls = []
    monkeypatch.setattr(host, "_entry_points", dict)
    monkeypatch.setattr(host, "_configure_aiter", lambda names: calls.append(names))
    host.configure((), kill_switch=False, strict=False)
    host._counts.clear()
    host._last.set(None)
    yield calls
    host.configure((), kill_switch=False, strict=False)
    host._counts.clear()


def _pair(*args, **kwargs):
    return "y", "scale"


def _enable(*names, strict=False):
    host.configure(names, kill_switch=False, strict=strict)


# -- registration --------------------------------------------------------


def test_register_accepts_an_enabled_owner():
    _enable("bmm")
    done = host.register(WO, _pair, owner="bmm", version=1, returns=("y", "y_scale"))
    assert done == host.Registration(WO, "bmm", True, "registered")
    assert host.status()["sites"] == {WO: ("bmm",)}


@pytest.mark.parametrize(
    "site, kwargs, reason",
    [
        ("attn.out.nope", {}, "unknown-site"),
        (WO, {"version": 2}, "version-mismatch"),
        (WO, {"returns": ("y",)}, "returns-mismatch"),
        (WO, {"returns": ("y_scale", "y")}, "returns-mismatch"),
        (WO, {"select": "yes"}, "not-callable"),
    ],
)
def test_register_refuses_with_a_reason(site, kwargs, reason):
    _enable("bmm")
    args = {"owner": "bmm", "version": 1, "returns": ("y", "y_scale"), **kwargs}
    with pytest.raises(host.RegistrationRefused) as refused:
        host.register(site, _pair, **args)
    assert refused.value.reason == reason
    assert host.status()["refusals"][-1].reason == reason
    assert host.status()["sites"] == {}


def test_owners_are_ordered_by_the_enable_list_not_by_registration():
    _enable("bmm8k", "bmm_prefill", "bmm_quant")
    for owner in ("bmm_quant", "bmm8k", "bmm_prefill"):
        host.register(WO, _pair, owner=owner, version=1, returns=("y", "y_scale"))
    assert host.owners(WO) == ("bmm8k", "bmm_prefill", "bmm_quant")
    assert host.status()["sites"] == {WO: ("bmm8k", "bmm_prefill", "bmm_quant")}


def test_an_owner_registers_a_site_once():
    _enable("bmm8k")
    host.register(WO, _pair, owner="bmm8k", version=1, returns=("y", "y_scale"))
    with pytest.raises(host.RegistrationRefused) as refused:
        host.register(WO, _pair, owner="bmm8k", version=1, returns=("y", "y_scale"))
    assert refused.value.reason == "duplicate"
    assert host.owners(WO) == ("bmm8k",)


def test_default_deny_registers_nothing():
    done = host.register(WO, _pair, owner="bmm", version=1, returns=("y", "y_scale"))
    assert (done.accepted, done.reason) == (False, "not-enabled")
    assert host.call(WO, lambda: "native") == "native"
    assert host.decision_counts() == {}


def test_kill_switch_wins_over_the_enable_list(monkeypatch):
    monkeypatch.setenv("ATOM_PLUGIN_HOST", "0")
    host.configure(["bmm"])
    done = host.register(WO, _pair, owner="bmm", version=1, returns=("y", "y_scale"))
    assert (done.accepted, done.reason) == (False, "kill-switch")
    assert host.status()["kill_switch"] is True


def test_no_environment_variable_enables_a_plugin(monkeypatch):
    monkeypatch.setenv("ATOM_PLUGIN_HOST", "1")
    host.configure(None)
    done = host.register(WO, _pair, owner="bmm", version=1, returns=("y", "y_scale"))
    assert not done.accepted


# -- entry points and the AITER half --------------------------------------


class _EntryPoint:
    def __init__(self, register):
        self._register = register

    def load(self):
        return self._register


def test_configure_loads_only_enabled_entry_points(monkeypatch, fresh_host):
    loaded = []

    def bmm(h):
        loaded.append("bmm")
        h.register(WO, _pair, owner="bmm", version=1, returns=("y", "y_scale"))

    def other(h):
        loaded.append("other")

    eps = {"bmm": _EntryPoint(bmm), "other": _EntryPoint(other)}
    monkeypatch.setattr(host, "_entry_points", lambda: eps)
    host.configure("bmm, dense_scalepack", kill_switch=False)
    assert loaded == ["bmm"]
    assert host.status()["sites"] == {WO: ("bmm",)}
    # The whole list goes to AITER too, in order: an AITER-only provider
    # (dense_scalepack) is enabled from the same place.
    assert fresh_host[-1] == ("bmm", "dense_scalepack")


def test_an_entry_point_registers_only_as_itself(monkeypatch):
    def impostor(h):
        h.register(WO, _pair, owner="bmm", version=1, returns=("y", "y_scale"))

    monkeypatch.setattr(host, "_entry_points", lambda: {"x": _EntryPoint(impostor)})
    host.configure(["x", "bmm"], kill_switch=False)
    reasons = [r.reason for r in host.status()["refusals"]]
    assert "owner-mismatch" in reasons and host.status()["sites"] == {}


def test_a_broken_plugin_is_isolated(monkeypatch):
    def broken(h):
        raise ImportError("no kernel")

    monkeypatch.setattr(host, "_entry_points", lambda: {"x": _EntryPoint(broken)})
    host.configure(["x"], kill_switch=False)
    assert host.status()["refusals"][-1] == host.Registration(
        "*", "x", False, "load-error"
    )
    assert host.call(WO, lambda: "native") == "native"


def test_kill_switch_hands_aiter_an_empty_list(fresh_host):
    host.configure(["bmm"], kill_switch=True)
    assert fresh_host[-1] == ()


def test_aiter_configure_is_driven_when_present(monkeypatch):
    seen = {}
    plugin = types.ModuleType("aiter.plugin")
    plugin.configure = lambda *, source, enable: seen.update(src=source, on=enable)
    aiter = types.ModuleType("aiter")
    aiter.plugin = plugin
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.plugin", plugin)
    assert CONFIGURE_AITER(("silu8k", "dense_scalepack")) is True
    assert seen == {"src": host.AITER_SOURCE, "on": ["silu8k", "dense_scalepack"]}


def test_aiter_without_a_plugin_interface_degrades(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
    monkeypatch.setitem(sys.modules, "aiter.plugin", None)
    assert CONFIGURE_AITER(("silu8k",)) is False


# -- call: fallback and decisions ----------------------------------------


def _serve_wo(select=None, impl=_pair):
    _enable("bmm")
    host.register(
        WO, impl, owner="bmm", version=1, returns=("y", "y_scale"), select=select
    )


def test_no_owner_is_the_native_call_and_records_nothing():
    assert host.call(WO, lambda a, *, b: (a, b), 1, b=2) == (1, 2)
    assert host.last_decision() is None and host.decision_counts() == {}


def test_selected_owner_serves_and_is_recorded():
    seen = []
    _serve_wo(
        select=lambda *a, **k: True, impl=lambda *a, **k: seen.append(a) or _pair()
    )
    assert host.call(WO, lambda *a, **k: "native", 1, 2, dtype="bf16") == ("y", "scale")
    assert seen == [(1, 2)]
    assert host.last_decision() == host.Decision(WO, "plugin", "selected", "bmm", 1)
    assert host.decision_counts() == {(WO, "plugin", "selected", "bmm"): 1}


def test_declining_owner_falls_back_to_native():
    _serve_wo(select=lambda x: x > 8)
    assert host.call(WO, lambda x: ("native", x), 4) == ("native", 4)
    assert host.call(WO, lambda x: ("native", x), 16) == ("y", "scale")
    assert host.decision_counts() == {
        (WO, "native", "declined", None): 1,
        (WO, "plugin", "selected", "bmm"): 1,
    }


def _bmm_family(strict=False):
    """bmm8k (M 8192/11750), bmm_prefill (16384), bmm_quant (6): disjoint cells."""
    cells = {"bmm8k": (8192, 11750), "bmm_prefill": (16384,), "bmm_quant": (6,)}
    _enable(*cells, strict=strict)
    for owner, rows in cells.items():
        host.register(
            WO,
            lambda m, owner=owner: (owner, m),
            owner=owner,
            version=1,
            returns=("y", "y_scale"),
            select=lambda m, rows=rows: m in rows,
        )


@pytest.mark.parametrize("strict", [False, True])
def test_owners_on_disjoint_cells_coexist(strict):
    _bmm_family(strict)

    def native(m):
        return "native", m

    assert host.call(WO, native, 11750) == ("bmm8k", 11750)
    assert host.last_decision() == host.Decision(WO, "plugin", "selected", "bmm8k", 1)
    assert host.call(WO, native, 16384) == ("bmm_prefill", 16384)
    assert host.last_decision().declined == 1
    assert host.call(WO, native, 6) == ("bmm_quant", 6)
    assert host.last_decision().declined == 2
    assert host.call(WO, native, 64) == ("native", 64)
    assert host.last_decision() == host.Decision(WO, "native", "declined", declined=3)
    assert host.decision_counts() == {
        (WO, "plugin", "selected", "bmm8k"): 1,
        (WO, "plugin", "selected", "bmm_prefill"): 1,
        (WO, "plugin", "selected", "bmm_quant"): 1,
        (WO, "native", "declined", None): 1,
    }


def test_the_first_accepting_owner_serves_in_enable_order():
    _enable("silu8k", "silu_quant")
    for owner in ("silu_quant", "silu8k"):
        host.register(
            WO,
            lambda owner=owner: (owner, 0),
            owner=owner,
            version=1,
            returns=("y", "y_scale"),
        )
    assert host.call(WO, lambda: "native") == ("silu8k", 0)
    assert host.last_decision().declined == 0
    assert host.explain(WO).owner == "silu8k"


def test_strict_mode_refuses_a_call_two_owners_accept():
    _enable("silu8k", "silu_quant", strict=True)
    ran = []
    for owner, rows in (("silu8k", (8192,)), ("silu_quant", (8192, 16384))):
        host.register(
            WO,
            lambda m, owner=owner: ran.append(owner) or (owner, m),
            owner=owner,
            version=1,
            returns=("y", "y_scale"),
            select=lambda m, rows=rows: m in rows,
        )
    with pytest.raises(host.PluginConflict, match="silu8k and silu_quant"):
        host.call(WO, lambda m: "native", 8192)
    with pytest.raises(host.PluginConflict):
        host.explain(WO, 8192)
    assert ran == []  # nothing ran
    assert host.status()["strict"] is True
    assert host.call(WO, lambda m: "native", 16384) == ("silu_quant", 16384)


def test_strict_mode_comes_from_the_environment_and_enables_nothing(monkeypatch):
    monkeypatch.setenv("ATOM_PLUGIN_HOST_STRICT", "1")
    host.configure(None, kill_switch=False)
    assert host.status()["strict"] is True and host.status()["enabled"] == ()
    monkeypatch.setenv("ATOM_PLUGIN_HOST_STRICT", "0")
    host.configure(["bmm"], kill_switch=False)
    assert host.status()["strict"] is False


def test_a_raising_select_is_a_decline():
    _serve_wo(select=lambda: 1 / 0)
    assert host.call(WO, lambda: "native") == "native"
    assert host.last_decision().reason == "select-error"


def test_a_wrong_result_shape_raises_and_never_retries_native():
    _serve_wo(impl=lambda: "only-y")
    native = []
    with pytest.raises(host.PluginContractError):
        host.call(WO, lambda: native.append(1))
    assert native == []


def test_in_place_sites_are_not_shape_checked():
    _enable("indexer_top512")
    host.register(
        TOPK, lambda *a, **k: None, owner="indexer_top512", version=1, returns=()
    )
    assert host.call(TOPK, lambda: "native") is None


def test_explain_runs_nothing():
    ran = []
    _serve_wo(select=lambda x: x > 8, impl=lambda x: ran.append(x))
    assert host.explain(WO, 4).route == "native"
    assert host.explain(WO, 16) == host.Decision(WO, "plugin", "selected", "bmm", 1)
    assert host.explain("moe.topk").reason == "no-owner"
    assert ran == [] and host.decision_counts() == {}


def test_site_decorator_routes_the_whole_function():
    @host.site(WO)
    def native(x, *, dequantize=False):
        return ("native", x, dequantize)

    assert native(3, dequantize=True) == ("native", 3, True)
    assert native.__name__ == "native"
    _serve_wo()
    assert native(3) == ("y", "scale")
    with pytest.raises(KeyError):
        host.site("not.a.site")


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(), reason="Dynamo unsupported here"
)
def test_traced_calls_stay_native_without_a_graph_break():
    torch._dynamo.reset()
    _serve_wo(impl=lambda x: (x * 100, None))

    @torch.compile(backend="eager", fullgraph=True)
    def forward(x):
        y, _ = host.call(WO, lambda t: (t + 1, None), x)
        return y

    out = forward(torch.ones(4))
    assert torch.equal(out, torch.full((4,), 2.0))
    assert host.decision_counts() == {(WO, "native", "traced", "bmm"): 1}
    torch._dynamo.reset()


# -- call sites ----------------------------------------------------------


def _sites_in(path):
    """Site names a module passes to `plugin_host.call` / `plugin_host.site`."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "plugin_host"
            and node.func.attr in ("call", "site")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            names.add(node.args[0].value)
    return names


def test_every_site_has_exactly_its_declared_call_site():
    found = {}
    for path in (REPO_ROOT / "atom").rglob("*.py"):
        for name in _sites_in(path):
            found.setdefault(name, []).append(path.relative_to(REPO_ROOT).as_posix())
    assert found == {name: [site.target] for name, site in host.SITES.items()}


@pytest.mark.parametrize(
    "argv, want",
    [([], ()), (["--plugin-enable", " bmm8k,,silu8k "], ("bmm8k", "silu8k"))],
)
def test_plugin_enable_reaches_the_config(argv, want):
    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(parser.parse_args(argv))
    assert args._get_engine_kwargs()["plugin_enable"] == want
    assert Config.__dataclass_fields__["plugin_enable"].default == ()


def test_the_enable_list_keys_the_compile_cache_only_when_set():
    from atom.config import Config

    base = {
        "quant_config": None,
        "compilation_config": None,
        "parallel_config": None,
        "tensor_parallel_size": 4,
        "prefill_context_parallel_size": 1,
        "sequence_parallel_size": 1,
        "dcp_config": None,
        "enable_dp_attention": False,
        "index_cache_dtype": None,
        "index_cache_layout": None,
        "hf_config": SimpleNamespace(),
        "max_model_len": 4096,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 16384,
        "kv_cache_dtype": "fp8",
        "speculative_config": None,
    }

    def key(**kw):
        return Config.compute_hash(SimpleNamespace(**base, **kw))

    assert key() == key(plugin_enable=())
    assert key(plugin_enable=("silu8k",)) != key()
    assert key(plugin_enable=("silu8k",)) != key(plugin_enable=("dense_scalepack",))


def test_model_runner_configures_the_host_before_building():
    source = (REPO_ROOT / "atom/model_engine/model_runner.py").read_text(
        encoding="utf-8"
    )
    configure = source.index("plugin_host.configure(config.plugin_enable)")
    assert configure < source.index("self._build_and_load_model(model_class)")


def test_proposer_hands_the_taps_to_a_model_that_takes_them():
    source = (REPO_ROOT / "atom/spec_decode/dspark_proposer.py").read_text(
        encoding="utf-8"
    )
    assert 'getattr(self.model, "context_takes_taps", False)' in source


def _attention_module():
    with atom_under_stubs() as load:
        return load("atom.models.deepseek_v41.attention")


def _project_out(module, fused, groups=2, rank=64, k=128, rows=3, heads=4):
    """Run `Attention._project_out` on a stand-in layer; return wo_b's args."""
    x_fp8 = torch.randn(rows, groups, k)
    x_scale = torch.randn(rows, groups, k // 32)
    module.inverse_rope_group_quant = lambda *a, **kw: (x_fp8, x_scale)
    module.batched_gemm_a8w8_mxscale_bpreshuffle = lambda x, w, xs, ws, dtype: (
        torch.full((rows, groups, rank), 1.0)
    )
    module.batched_gemm_a8w8_mxscale_bpreshuffle_fused_quant = lambda x, w, xs, ws: (
        torch.full((rows, groups, rank), 2.0),
        torch.full((rows, groups, rank // 32), 3.0),
    )
    wo_b = []
    layer = SimpleNamespace(
        heads=heads,
        head_dim=groups * k // heads,
        groups=groups,
        o_rank=rank,
        _fuse_woa_quant=fused,
        wo_a=SimpleNamespace(
            weight=torch.zeros(groups * rank, k),
            weight_scale=torch.zeros(groups * rank // 32, k // 32),
        ),
        wo_b=lambda y, x_scale=None: wo_b.append((y, x_scale)) or "out",
    )
    layer._wo_a = functools.partial(module.Attention._wo_a, layer)
    output = torch.zeros(1, rows, heads, groups * k // heads)
    rope = SimpleNamespace(cos_cache=None, sin_cache=None)
    assert (
        module.Attention._project_out(layer, output, rope, torch.zeros(1, rows))
        == "out"
    )
    return wo_b[0], (x_fp8, x_scale)


@pytest.mark.parametrize("fused", [True, False])
def test_wo_a_site_native_feeds_wo_b_as_before(fused):
    module = _attention_module()
    (y, y_scale), _ = _project_out(module, fused)
    assert y.shape == (1, 3, 2 * 64)
    if fused:
        assert torch.equal(y_scale, torch.full((1, 3, 4), 3.0))
    else:
        assert y_scale is None and torch.equal(y, torch.ones(1, 3, 128))


def test_wo_a_site_plugin_pair_reaches_wo_b():
    module = _attention_module()
    got = []

    def plugin(x_fp8, x_scale, w, ws, *, dtype):
        got.append((x_fp8, x_scale, w.shape, ws.shape))
        return torch.full((3, 2, 64), 7.0), torch.full((3, 2, 2), 8.0)

    _serve_wo(impl=plugin)
    (y, y_scale), (x_fp8, x_scale) = _project_out(module, fused=False)
    assert got[0][0] is x_fp8 and got[0][1] is x_scale
    assert got[0][2:] == ((2, 64, 128), (2, 2, 4))
    assert torch.equal(y, torch.full((1, 3, 128), 7.0))
    assert torch.equal(y_scale, torch.full((1, 3, 4), 8.0))
    assert host.last_decision().owner == "bmm"


SHARED = "moe.shared.act..down"
DIM = 8


def _shared_expert(name, fused=False):
    """`Expert` from deepseek_v4, built (`span_op`) on a stand-in layer after
    `configure`, as ModelRunner orders it: (layer, forward, w2's calls)."""
    with atom_under_stubs() as load:
        module = load("atom.models.deepseek_v4")
    module.aiter_silu_and_mul = lambda out, combined, limit: out.copy_(
        combined[:, : out.shape[1]].clamp(max=limit)
    )
    w2 = []

    class W2:
        native_a8_group_rows = 32
        output_size = DIM

        def __call__(self, x, x_scale=None):
            w2.append((x, x_scale))
            return x.new_ones(x.shape[0], DIM, dtype=torch.bfloat16)

    layer = SimpleNamespace(
        gate_up_proj=lambda x: torch.cat([x, x], dim=-1),
        use_fused_clamp_act_mul=fused,
        swiglu_limit=10.0,
        w2=W2(),
    )
    for method in ("_act", "_act_down", "_act_down_fake"):
        setattr(layer, method, functools.partial(getattr(module.Expert, method), layer))
    layer.act_down = host.span_op(SHARED, name, layer._act_down, layer._act_down_fake)
    return layer, functools.partial(module.Expert.forward, layer), w2


def _serve_shared(seen, select=None):
    def silu_quant(combined, weights, *, swiglu_limit, group_rows):
        seen.append((combined.shape, weights, swiglu_limit, group_rows))
        rows = combined.shape[0]
        return torch.zeros(rows, 3), torch.ones(rows, 1)

    _enable("silu_quant")
    host.register(
        SHARED,
        silu_quant,
        owner="silu_quant",
        version=1,
        returns=("act", "act_scale"),
        select=select or (lambda combined, weights, **kw: kw["group_rows"] == 32),
    )


def test_shared_site_without_a_plugin_stays_inline():
    layer, forward, w2 = _shared_expert("model.layers.0.ffn.shared_experts")
    assert layer.act_down == layer._act_down  # no opaque op in the graph
    x = torch.arange(6.0).reshape(2, 3) * 5
    assert forward(x).shape == (2, DIM)
    act, scale = w2[0]
    assert scale is None and torch.equal(act, x.clamp(max=10.0))
    assert host.decision_counts() == {}


def test_shared_site_with_a_plugin_is_one_opaque_op():
    seen = []
    _serve_shared(seen)
    layer, forward, w2 = _shared_expert("model.layers.1.ffn.shared_experts")
    assert layer.act_down != layer._act_down
    assert forward(torch.ones(2, 3)).shape == (2, DIM)
    assert seen == [((2, 6), None, 10.0, 32)]
    assert torch.equal(w2[0][1], torch.ones(2, 1))
    assert host.last_decision().route == "plugin"


def test_shared_span_op_serves_every_owner_by_cell():
    served = []
    _enable("silu8k", "silu_quant")
    for owner, rows in (("silu8k", 4), ("silu_quant", 2)):
        host.register(
            SHARED,
            lambda c, w, owner=owner, **kw: served.append(owner)
            or (torch.zeros(c.shape[0], 3), torch.ones(c.shape[0], 1)),
            owner=owner,
            version=1,
            returns=("act", "act_scale"),
            select=lambda c, w, rows=rows, **kw: c.shape[0] == rows,
        )
    layer, forward, w2 = _shared_expert("model.layers.5.ffn.shared_experts")
    assert layer.act_down != layer._act_down
    for rows in (4, 2, 3):
        forward(torch.ones(rows, 3))
    assert served == ["silu8k", "silu_quant"]
    assert w2[2][1] is None  # three rows: both declined, native
    assert host.last_decision().declined == 2


def test_shared_span_op_fake_gives_w2_output_shape():
    _serve_shared([])
    layer, _, w2 = _shared_expert("model.layers.2.ffn.shared_experts")
    out = layer.act_down(torch.empty(5, 6, device="meta"), None)
    assert (out.shape, out.dtype, out.device.type) == ((5, DIM), torch.bfloat16, "meta")
    assert w2 == [] and host.decision_counts() == {}


def test_span_layer_names_are_unique():
    _serve_shared([])
    _shared_expert("model.layers.3.ffn.shared_experts")
    with pytest.raises(ValueError):
        _shared_expert("model.layers.3.ffn.shared_experts")


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(), reason="Dynamo unsupported here"
)
def test_compiled_shared_site_decides_at_run_time():
    """Traced with fullgraph: the plugin still serves, and its select still
    runs on every execution, not once at trace time."""
    torch._dynamo.reset()
    seen = []
    _serve_shared(seen, select=lambda combined, weights, **kw: combined.shape[0] > 2)
    _, forward, _ = _shared_expert("model.layers.4.ffn.shared_experts")
    compiled = torch.compile(forward, backend="eager", fullgraph=True, dynamic=True)
    compiled(torch.ones(4, 3))
    compiled(torch.ones(2, 3))
    compiled(torch.ones(8, 3))
    assert [s[0][0] for s in seen] == [4, 8]
    assert host.decision_counts() == {
        (SHARED, "plugin", "selected", "silu_quant"): 2,
        (SHARED, "native", "declined", None): 1,
    }
    torch._dynamo.reset()


def _draft_member(name):
    """One `DeepseekV41DSpark` member, compiled alone: the module itself needs
    FlyDSL, which the CPU gate does not install."""
    tree = ast.parse(
        (REPO_ROOT / "atom/models/deepseek_v41/dspark.py").read_text(encoding="utf-8")
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41DSpark"
    )
    for node in cls.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", 0) == name:
            return ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.decorator_list = []  # @staticmethod
            namespace = {"torch": torch}
            code = compile(ast.Module([node], []), "dspark.py", "exec")
            exec(code, namespace)  # noqa: S102 -- ATOM's own source, one function
            return namespace[name]
    raise LookupError(name)


def test_draft_concat_native_is_the_old_cat_and_slice():
    concat = functools.partial(_draft_member("_concat_taps"), group_rows=32)
    taps = [torch.randn(8, 4) for _ in range(3)]
    rows = torch.tensor([1, 3, 5, 7])
    old = torch.cat(taps, dim=-1)
    assert torch.equal(concat(taps, None, 6)[0], old[:6])
    assert torch.equal(concat(taps, rows, 3)[0], old[rows][:3])
    assert torch.equal(concat(old, rows, 3)[0], old[rows][:3])
    assert concat(taps, None, 6)[1] is None
    assert _draft_member("context_takes_taps") is True


def test_draft_site_feeds_main_proj_its_pair():
    source = (REPO_ROOT / "atom/models/deepseek_v41/dspark.py").read_text(
        encoding="utf-8"
    )
    assert source.count("main_proj(aux_concat, x_scale=aux_scale)") == 2
    assert "self.project_context(ctx, ctx_scale)" in source
    assert "self.project_context_quant(ctx, ctx_scale)" in source
