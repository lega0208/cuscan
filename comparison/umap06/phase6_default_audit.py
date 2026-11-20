"""Phase 6 static default-pinning audit.

Mechanically inventories every callsite that builds a UMAP configuration and may reach a
fit or initializer sink (``UMAP::fit``, ``umap_core::initialize``,
``umap::standard_umap::initialize``, the Python ``cuscan.UMAP`` estimator and the native
``fit_umap``) across the consumer roots, and records whether ``optimizer`` and ``init`` are
set explicitly, set by a later mutation, inherited from a named helper, left at the library
default, or unresolvable. Sites that leave either field to the library default (or cannot be
resolved) and are not allowlisted are failures, so the audit exits non-zero until the
pinning rule of ``umap_06_phase6_plan.md`` is applied.

This is a static scan. It cannot prove the effective options that reach a sink; the Phase 6
runtime trace is the independent check. Known limitations are listed in the JSON output.

Usage::

    uv run --no-sync python comparison/umap06/phase6_default_audit.py \
        [--root .] [--out-dir artifacts/umap06_p6/audit] [--no-write] [--quiet]

Exit codes: 0 clean, 1 failing sites (unpinned, unresolved or unreached-unallowlisted),
2 usage or internal error.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA = "umap06.phase6.default_audit.v1"
ROOTS = (
    "crates",
    "python",
    "comparison",
    "scripts",
    "analyze",
    ".github/workflows",
    "README.md",
    "docs",
)
SKIP_DIRS = {"target", "__pycache__", ".venv", "node_modules", ".git", ".pytest_cache"}
FIELDS = ("optimizer", "init")
OLD_DEFAULTS = {
    "rust": {"optimizer": "Optimizer::Compatibility", "init": "Init::Spectral"},
    "python": {"optimizer": '"compatibility"', "init": '"spectral"'},
}

# Field states that count as "pinned" (the field does not follow the library default).
COVERED_STATES = {"explicit", "mutated", "helper"}

LIMITATIONS = [
    "Rust parsing is lexical (comment/string masking plus brace matching), not a full parser.",
    "Rust helper and sink resolution is per file plus a sibling support/mod.rs or common/mod.rs; "
    "cross-crate helper flow is not followed.",
    "Rust sink reachability is a per-file call graph over UMAP::new, .fit(, .fit_transform(, "
    "initialize( and fit_umap( tokens; configs passed into other files' functions are reachable "
    "only if the enclosing function itself reaches a sink token.",
    "A Rust struct-update base that is a variable is resolved only to an earlier `let` binding of "
    "a recorded site in the same function; other bases (fields, parameters) are `dynamic`.",
    "Python `**mapping` arguments are resolved through dict literals, dict(...) calls, .update, "
    "subscript and setdefault assignments in the enclosing function and module; filtered dict "
    "comprehensions over a mapping are treated as that mapping (keys may be over-approximated).",
    "Python sink reachability: a site reaches a sink when its scope calls .fit/.fit_transform/"
    "fit_umap, or it is a helper definition or helper call.",
    "Markdown and YAML are scanned lexically: fenced rust/python blocks use the language scanner, "
    "everything else uses a balanced-parenthesis keyword check.",
    "Calls on the `umap` module (umap-learn) are upstream and excluded.",
]


@dataclasses.dataclass(frozen=True)
class AllowEntry:
    path: str
    function: str | None  # enclosing function name; None matches any
    match: str | None  # substring that must appear in the site's source text; None matches any
    category: str
    reason: str


# Plan-named exemptions (umap_06_phase6_plan.md "Pinning rule"). Extend only with a reason.
ALLOWLIST: tuple[AllowEntry, ...] = (
    *(
        AllowEntry(path, function, None, "default_subject", reason)
        for path, functions, reason in (
            (
                "crates/umap/tests/test_standard_estimator.rs",
                (
                    "default_fit_matches_explicit_adam_auto_and_recursive",
                    "default_edge_cases",
                ),
                "Exercises library-default fit and transform.",
            ),
            (
                "crates/umap/tests/test_standard_quality.rs",
                ("run_cluster_quality",),
                "Intentionally tests library defaults as well as explicit compatibility.",
            ),
            (
                "crates/umap/tests/test_standard_serialization.rs",
                ("default_fit_roundtrips_auto_adam",),
                "Verifies persistence of library defaults.",
            ),
            (
                "python/tests/test_serialization.py",
                ("test_v5_library_defaults_round_trip",),
                "Verifies persistence of library defaults.",
            ),
            (
                "python/tests/test_standard_umap.py",
                (
                    "test_defaults_are_auto_and_adam",
                    "test_default_fit_matches_explicit_adam_auto",
                    "test_native_default_arguments",
                ),
                "Verifies wrapper and native defaults.",
            ),
            (
                "crates/umap/benches/support/mod.rs",
                ("estimator", "benchmarks"),
                "Benchmarks the library-default axis against explicit compatibility.",
            ),
        )
        for function in functions
    ),
    AllowEntry(
        "crates/umap-core/tests/test_standard_init.rs",
        "default_config_requires_fit_driver",
        None,
        "default_subject",
        "The host initializer must reject recursive initialization under library defaults.",
    ),
    AllowEntry(
        "crates/umap/tests/test_standard_umap.rs",
        "standard_config_defaults_match_the_reference_controls",
        None,
        "default_subject",
        "Asserts the requested initializer and other library defaults directly.",
    ),
    AllowEntry(
        "crates/umap/tests/test_standard_serialization.rs",
        "phase4_settings_roundtrip_and_persisted_validation",
        "UMAPConfig::default()",
        "non_sink",
        "Reads only the default Phase 4 exclusion and adaptation controls.",
    ),
    *(
        AllowEntry(path, function, None, "non_sink", "Only validates configuration fields.")
        for path, functions in (
            (
                "crates/umap-core/src/config.rs",
                (
                    "validates_modern_controls_even_for_compatibility",
                    "validates_phase4_negative_sampling_controls",
                    "validates_memory_limit",
                ),
            ),
            (
                "crates/umap/tests/test_standard_umap.rs",
                (
                    "config_rejects_invalid_control_values",
                    "reference_permits_zero_epochs_and_optimizer_controls",
                    "optimizer_controls_reject_negative_or_nonfinite_floats",
                ),
            ),
        )
        for function in functions
    ),
    *(
        AllowEntry(
            "python/tests/test_standard_umap.py",
            function,
            None,
            "non_sink",
            "Invalid parameters are rejected before initialization or optimizer execution.",
        )
        for function in (
            "test_optimizer_settings_reject_invalid_values",
            "test_native_fit_rejects_optimizer_settings_before_cuda_creation",
            "test_phase4_settings_reject_invalid_values_in_wrapper_and_native",
            "test_invalid_parameters_raise_value_error",
        )
    ),
    AllowEntry(
        "crates/umap-core/src/config.rs",
        "defaults_and_init_resolution",
        None,
        "default_subject",
        "Asserts the library defaults themselves; rewritten to the new defaults.",
    ),
    AllowEntry(
        "python/tests/test_standard_umap.py",
        "test_phase4_settings_defaults",
        None,
        "default_subject",
        "Asserts the library defaults themselves; rewritten to add init/optimizer assertions.",
    ),
    AllowEntry(
        "crates/umap/tests/test_standard_estimator.rs",
        "invalid_config_and_input_are_reported_as_errors",
        "n_components: 0",
        "non_sink",
        "n_components: 0 is rejected by validation before any init/optimizer-dependent path.",
    ),
    AllowEntry(
        "crates/umap/src/standard_umap/estimator.rs",
        "profile_recursive_driver",
        None,
        "non_sink",
        "Reads only n_components, the curve and recursive_init.",
    ),
    AllowEntry(
        "crates/cuscan-python/src/lib.rs",
        None,
        None,
        "sets_every_field",
        "PyO3 literals set every field from the Python arguments.",
    ),
    AllowEntry(
        ".github/workflows/_wheels.yml",
        None,
        None,
        "intentional_default_consumer",
        "Installed-wheel smoke test must follow the shipped default.",
    ),
    AllowEntry(
        "comparison/save_embeddings.py",
        None,
        None,
        "intentional_default_consumer",
        "Example embedding export follows the shipped default.",
    ),
    AllowEntry(
        "README.md",
        None,
        None,
        "intentional_default_consumer",
        "Usage example follows the shipped default.",
    ),
    AllowEntry(
        "docs/development.md",
        None,
        None,
        "intentional_default_consumer",
        "Usage example follows the shipped default.",
    ),
)


@dataclasses.dataclass
class Site:
    path: str
    line: int
    language: str
    kind: str  # literal | default_call | helper_call | sink_call | text_call
    function: str | None
    is_test: bool
    text: str
    fields: dict[str, str]  # field -> state
    sources: dict[str, str]  # field -> detail (helper name, base expression, ...)
    helper_definition: str | None = None  # helper this site defines, if any
    helper_ref: str | None = None  # "path::helper" a helper_call site inherits from
    reaches_sink: bool = True
    status: str = ""
    allow: dict[str, str] | None = None
    missing_pins: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------- Rust


def mask_rust(source: str) -> str:
    """Replace comment and string/char literal contents with spaces, preserving offsets."""
    out = list(source)
    i, n = 0, len(source)

    def blank(start: int, end: int) -> None:
        for k in range(start, end):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = source[i]
        if source.startswith("//", i):
            end = source.find("\n", i)
            end = n if end < 0 else end
            blank(i, end)
            i = end
        elif source.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if source.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif source.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif (
            c == "r"
            and re.match(r'r#*"', source[i:])
            and (i == 0 or not (source[i - 1].isalnum() or source[i - 1] == "_"))
        ):
            m = re.match(r'r(#*)"', source[i:])
            assert m
            closing = '"' + m.group(1)
            end = source.find(closing, i + m.end())
            end = n if end < 0 else end + len(closing)
            blank(i + 1, end)
            i = end
        elif c == '"':
            j = i + 1
            while j < n and source[j] != '"':
                j += 2 if source[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
        elif c == "'":
            m = re.match(r"'(\\(x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'])'", source[i:])
            if m:
                blank(i + 1, i + m.end() - 1)
                i += m.end()
            else:
                i += 1  # lifetime
        else:
            i += 1
    return "".join(out)


def match_close(masked: str, open_index: int) -> int:
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack = [pairs[masked[open_index]]]
    i = open_index + 1
    while i < len(masked) and stack:
        c = masked[i]
        if c in pairs:
            stack.append(pairs[c])
        elif c in ")}]":
            if c == stack[-1]:
                stack.pop()
        i += 1
    return i - 1


def split_top_level(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


@dataclasses.dataclass
class RustFn:
    name: str
    start: int  # offset of `fn`
    body_open: int
    body_close: int
    signature: str
    is_test: bool


def rust_functions(masked: str) -> list[RustFn]:
    fns = []
    for m in re.finditer(r"\bfn\s+(\w+)", masked):
        j, depth = m.end(), 0
        while j < len(masked):
            c = masked[j]
            if c in "(<[":
                depth += 1
            elif c in ")>]":
                depth -= 1 if not (c == ">" and masked[j - 1] == "-") else 0
            elif c == ";" and depth <= 0:
                j = -1
                break
            elif c == "{" and depth <= 0:
                break
            j += 1
        if j < 0 or j >= len(masked):
            continue
        close = match_close(masked, j)
        prefix_start = max(masked.rfind("}", 0, m.start()), masked.rfind(";", 0, m.start()), 0)
        attrs = masked[prefix_start : m.start()]
        is_test = bool(re.search(r"#\[\s*(tokio::)?test\b|#\[\s*rstest\b", attrs))
        fns.append(RustFn(m.group(1), m.start(), j, close, masked[m.end() : j], is_test))
    return fns


def innermost_fn(fns: list[RustFn], offset: int) -> RustFn | None:
    best = None
    for f in fns:
        if f.body_open <= offset <= f.body_close and (best is None or f.body_open > best.body_open):
            best = f
    return best


RUST_SINK_RE = re.compile(
    r"\bUMAP::new\s*\(|\.fit\s*\(|\.fit_transform\s*\(|\binitialize\s*\(|\bfit_umap\s*\("
)
RUST_CONFIG_LITERAL_RE = re.compile(r"\b(?:[\w:]+::)?UMAPConfig\s*\{")
RUST_DEFAULT_CALL_RE = re.compile(r"\b(?:[\w:]+::)?UMAPConfig::default\s*\(\s*\)")


@dataclasses.dataclass
class RustFile:
    path: str
    source: str
    masked: str
    fns: list[RustFn]
    line_starts: list[int]

    def line_of(self, offset: int) -> int:
        lo, hi = 0, len(self.line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1


def load_rust(path: str, source: str) -> RustFile:
    masked = mask_rust(source)
    starts = [0] + [m.end() for m in re.finditer("\n", source)]
    return RustFile(path, source, masked, rust_functions(masked), starts)


@dataclasses.dataclass
class RustRawSite:
    file: RustFile
    start: int
    end: int
    kind: str
    explicit: set[str]
    base: str | None  # struct update base, or "default" for UMAPConfig::default()
    binding: str | None
    helper_name: str | None = None


def binding_before(rf: RustFile, start: int) -> str | None:
    stmt_start = max(rf.masked.rfind(c, 0, start) for c in ";{}") + 1
    m = re.search(r"\blet\s+(?:mut\s+)?(\w+)\s*(?::[^=]+)?=\s*$", rf.masked[stmt_start:start])
    return m.group(1) if m else None


def rust_raw_sites(rf: RustFile) -> list[RustRawSite]:
    sites = []
    masked = rf.masked
    for m in RUST_CONFIG_LITERAL_RE.finditer(masked):
        before = masked[max(0, m.start() - 40) : m.start()]
        if re.search(r"(->|\bstruct|\bfor|\bimpl(<[^>]*>)?)\s*$", before):
            continue
        open_brace = m.end() - 1
        close = match_close(masked, open_brace)
        explicit, base = set(), None
        for part in split_top_level(masked[open_brace + 1 : close]):
            if part.startswith(".."):
                base = rf.source[
                    masked.index(part, open_brace) + 2 : masked.index(part, open_brace) + len(part)
                ].strip()
                base = re.sub(r"\s+", "", base)
                continue
            fm = re.match(r"(\w+)\s*(:|$)", part)
            if fm and fm.group(1) in FIELDS:
                explicit.add(fm.group(1))
        sites.append(
            RustRawSite(
                rf, m.start(), close + 1, "literal", explicit, base, binding_before(rf, m.start())
            )
        )
    for m in RUST_DEFAULT_CALL_RE.finditer(masked):
        if re.search(r"\.\.\s*$", masked[max(0, m.start() - 8) : m.start()]):
            continue
        sites.append(
            RustRawSite(
                rf,
                m.start(),
                m.end(),
                "default_call",
                set(),
                "default",
                binding_before(rf, m.start()),
            )
        )
    return sites


def rust_mutations(rf: RustFile, binding: str, after: int, fn: RustFn | None) -> set[str]:
    if fn is None:
        return set()
    body = rf.masked[after : fn.body_close]
    return {f for f in FIELDS if re.search(rf"\b{re.escape(binding)}\.{f}\s*=(?!=)", body)}


def audit_rust(files: list[RustFile]) -> list[Site]:
    by_dir: dict[str, list[RustFile]] = {}
    for rf in files:
        by_dir.setdefault(str(Path(rf.path).parent), []).append(rf)

    raw: dict[str, list[RustRawSite]] = {rf.path: rust_raw_sites(rf) for rf in files}

    # Helpers: non-test functions containing a config site whose signature yields a config
    # or an estimator built from it.
    helpers: dict[str, dict[str, tuple[RustFile, RustFn, list[RustRawSite]]]] = {}
    for rf in files:
        local: dict[str, tuple[RustFile, RustFn, list[RustRawSite]]] = {}
        for fn in rf.fns:
            if fn.is_test or not re.search(
                r"->[^{]*\b(UMAPConfig|UMAP|FittedUMAP)\b", fn.signature
            ):
                continue
            inner = [
                s
                for s in raw[rf.path]
                if fn.body_open <= s.start <= fn.body_close and innermost_fn(rf.fns, s.start) is fn
            ]
            if inner:
                local[fn.name] = (rf, fn, inner)
        helpers[rf.path] = local

    def visible_helpers(rf: RustFile) -> dict[str, tuple[RustFile, RustFn, list[RustRawSite]]]:
        out = dict(helpers[rf.path])
        parent = Path(rf.path).parent
        for support in ("support/mod.rs", "common/mod.rs"):
            spath = str(parent / support)
            if spath in helpers and spath != rf.path:
                for k, v in helpers[spath].items():
                    out.setdefault(k, v)
        return out

    # Sink reachability per file (local call graph).
    reach: dict[str, set[str]] = {}
    for rf in files:
        names = {f.name for f in rf.fns}
        reaching = {
            f.name for f in rf.fns if RUST_SINK_RE.search(rf.masked[f.body_open : f.body_close])
        }
        changed = True
        while changed:
            changed = False
            for f in rf.fns:
                if f.name in reaching:
                    continue
                body = rf.masked[f.body_open : f.body_close]
                if any(
                    re.search(rf"(?<![\w.]){re.escape(r)}\s*\(", body) for r in reaching & names
                ):
                    reaching.add(f.name)
                    changed = True
        reach[rf.path] = reaching

    memo: dict[tuple[str, int], tuple[dict[str, str], dict[str, str]]] = {}

    def resolve(site: RustRawSite, stack: tuple = ()) -> tuple[dict[str, str], dict[str, str]]:
        key = (site.file.path, site.start)
        if key in memo:
            return memo[key]
        if key in stack:
            return {f: "dynamic" for f in FIELDS}, {f: "cycle" for f in FIELDS}
        rf = site.file
        fn = innermost_fn(rf.fns, site.start)
        states, sources = {}, {}
        base_states: dict[str, str] = {}
        base_sources: dict[str, str] = {}
        base = site.base
        if base is not None:
            base = re.sub(r"\.clone\(\)$", "", base)
        if base is None and site.kind == "literal":
            base_states = {f: "unset_missing" for f in FIELDS}
        elif base in (None, "default") or re.fullmatch(
            r"(?:[\w:]+::)?(UMAPConfig|Default)::default\(\)", base or ""
        ):
            base_states = {f: "default" for f in FIELDS}
            base_sources = {f: "UMAPConfig::default()" for f in FIELDS}
        else:
            hm = re.fullmatch(r"(\w+)\((.*)\)", base)
            vis = visible_helpers(rf)
            if hm and hm.group(1) in vis:
                h_states, _ = helper_states(hm.group(1), vis[hm.group(1)], (*stack, key))
                for f in FIELDS:
                    base_states[f] = "helper" if h_states[f] in COVERED_STATES else h_states[f]
                    base_sources[f] = f"helper:{hm.group(1)}"
            elif re.fullmatch(r"\w+", base) and fn is not None:
                prior = [
                    s
                    for s in raw[rf.path]
                    if s.binding == base
                    and s.start < site.start
                    and innermost_fn(rf.fns, s.start) is fn
                ]
                if prior:
                    p_states, p_sources = resolve(prior[-1], (*stack, key))
                    base_states, base_sources = (
                        dict(p_states),
                        {f: f"binding:{base}<-{p_sources[f]}" for f in FIELDS},
                    )
                else:
                    base_states = {f: "dynamic" for f in FIELDS}
                    base_sources = {f: f"base:{base}" for f in FIELDS}
            else:
                base_states = {f: "dynamic" for f in FIELDS}
                base_sources = {f: f"base:{base}" for f in FIELDS}
        mutated = rust_mutations(rf, site.binding, site.end, fn) if site.binding else set()
        for f in FIELDS:
            if f in site.explicit:
                states[f], sources[f] = "explicit", "literal"
            elif f in mutated:
                states[f], sources[f] = "mutated", f"{site.binding}.{f} ="
            else:
                states[f], sources[f] = base_states[f], base_sources.get(f, "")
        memo[key] = (states, sources)
        return states, sources

    helper_memo: dict[tuple[str, str], tuple[dict[str, str], dict[str, str]]] = {}

    def helper_states(name: str, entry, stack: tuple = ()) -> tuple[dict[str, str], dict[str, str]]:
        hrf, _hfn, inner = entry
        key = (hrf.path, name)
        if key not in helper_memo:
            agg = {f: "explicit" for f in FIELDS}
            rank = {
                "explicit": 0,
                "mutated": 0,
                "helper": 0,
                "default": 1,
                "unset_missing": 1,
                "dynamic": 2,
            }
            for s in inner:
                st, _ = resolve(s, stack)
                for f in FIELDS:
                    if rank.get(st[f], 2) > rank.get(agg[f], 2):
                        agg[f] = st[f]
            helper_memo[key] = (agg, {f: f"helper:{name}" for f in FIELDS})
        return helper_memo[key]

    sites: list[Site] = []
    helper_def_offsets: dict[tuple[str, int], str] = {}
    for rf in files:
        for name, (hrf, _hfn, inner) in helpers[rf.path].items():
            for s in inner:
                helper_def_offsets[(hrf.path, s.start)] = name

    helper_called_from_reaching: dict[tuple[str, str], bool] = {}
    helper_calls: list[tuple[RustFile, int, int, str, tuple]] = []
    for rf in files:
        vis = visible_helpers(rf)
        for name, entry in vis.items():
            for m in re.finditer(rf"(?<![\w.:]){re.escape(name)}\s*\(", rf.masked):
                if re.search(r"\bfn\s+$", rf.masked[max(0, m.start() - 10) : m.start()]):
                    continue
                if re.search(
                    r"#!?\[\s*(cfg_attr\s*\()?\s*$", rf.masked[max(0, m.start() - 16) : m.start()]
                ):
                    continue  # attribute such as #[cfg(test)]
                if re.search(r"\.\.\s*$", rf.masked[max(0, m.start() - 8) : m.start()]):
                    continue  # struct-update base, recorded on the literal
                close = match_close(rf.masked, m.end() - 1)
                helper_calls.append((rf, m.start(), close + 1, name, entry))
                caller = innermost_fn(rf.fns, m.start())
                hkey = (entry[0].path, name)
                reached = caller is not None and (
                    caller.name in reach[rf.path] or caller.name in helpers[rf.path]
                )
                helper_called_from_reaching[hkey] = (
                    helper_called_from_reaching.get(hkey, False) or reached
                )

    def make_site(
        rf: RustFile, start: int, end: int, kind: str, states, sources, helper_def: str | None
    ) -> Site:
        fn = innermost_fn(rf.fns, start)
        text = rf.source[start:end]
        if fn is None:
            reaches = False
        elif helper_def is not None:
            reaches = helper_called_from_reaching.get((rf.path, helper_def), False)
        else:
            reaches = fn.name in reach[rf.path]
        return Site(
            path=rf.path,
            line=rf.line_of(start),
            language="rust",
            kind=kind,
            function=fn.name if fn else None,
            is_test=bool(fn and fn.is_test),
            text=re.sub(r"\s+", " ", text)[:400],
            fields=dict(states),
            sources=dict(sources),
            helper_definition=helper_def,
            reaches_sink=reaches,
        )

    for rf in files:
        for s in raw[rf.path]:
            states, sources = resolve(s)
            if (
                "config.rs" in rf.path
                and (innermost_fn(rf.fns, s.start) or RustFn("", 0, 0, 0, "", False)).name
                == "default"
                and rf.path.endswith("umap-core/src/config.rs")
            ):
                continue  # the Default impl is the definition of the defaults
            sites.append(
                make_site(
                    rf,
                    s.start,
                    s.end,
                    s.kind,
                    states,
                    sources,
                    helper_def_offsets.get((rf.path, s.start)),
                )
            )
    for rf, start, end, name, entry in helper_calls:
        h_states, _ = helper_states(name, entry)
        binding = binding_before(rf, start)
        fn = innermost_fn(rf.fns, start)
        mutated = rust_mutations(rf, binding, end, fn) if binding else set()
        states, sources = {}, {}
        for f in FIELDS:
            if f in mutated:
                states[f], sources[f] = "mutated", f"{binding}.{f} ="
            else:
                states[f] = "helper" if h_states[f] in COVERED_STATES else h_states[f]
                sources[f] = f"helper:{name}"
        site = make_site(rf, start, end, "helper_call", states, sources, None)
        site.helper_ref = f"{entry[0].path}::{name}"
        sites.append(site)
    return sites


# ---------------------------------------------------------------------------- Python


def _attr_chain(node: ast.AST) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return []


class PyScope:
    def __init__(self, node: ast.AST, parent: PyScope | None) -> None:
        self.node = node
        self.parent = parent


def py_sink_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return (cuscan UMAP names, cuscan module aliases)."""
    umap_names, modules = set(), set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "cuscan"
        ):
            for a in node.names:
                if a.name == "UMAP":
                    umap_names.add(a.asname or a.name)
                if a.name in ("_cuscan", "standard_umap"):
                    modules.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "cuscan":
                    modules.add(a.asname or a.name.split(".")[0])
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            chain = _attr_chain(node.value)
            if chain and chain[0] in {"cuscan"} | modules and chain[-1] == "UMAP":
                umap_names.add(node.targets[0].id)
    return umap_names, modules


def py_is_sink(call: ast.Call, umap_names: set[str], modules: set[str]) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        if f.id in umap_names:
            return "UMAP"
        if f.id == "fit_umap" and modules:
            return "fit_umap"
        return None
    chain = _attr_chain(f)
    if not chain or chain[0] not in modules | {"cuscan", "_cuscan"}:
        return None
    if chain[0] == "_cuscan" and "_cuscan" not in modules:
        return None
    if chain[-1] in ("UMAP", "fit_umap"):
        return chain[-1]
    return None


@dataclasses.dataclass
class KeySet:
    keys: dict[str, str]  # key -> detail
    dynamic: list[str]  # unresolved pieces
    from_caller: bool = False  # merges the enclosing function's **kwargs


class PyAnalyzer:
    def __init__(self, path: str, source: str, line_offset: int = 0) -> None:
        self.path = path
        self.source = source
        self.line_offset = line_offset
        self.tree = ast.parse(source)
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.umap_names, self.modules = py_sink_names(self.tree)

    def scope_of(self, node: ast.AST) -> ast.AST:
        cur = self.parents.get(node)
        while cur is not None and not isinstance(
            cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            cur = self.parents.get(cur)
        return cur or self.tree

    def scope_chain(self, node: ast.AST) -> list[ast.AST]:
        out, cur = [], self.scope_of(node)
        while True:
            out.append(cur)
            if isinstance(cur, ast.Module):
                return out
            cur = self.scope_of(cur)

    def outer_function(self, node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for s in self.scope_chain(node):
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return s
        return None

    def own_nodes(self, scope: ast.AST) -> Iterator[ast.AST]:
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            n = stack.pop()
            yield n
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                stack.extend(ast.iter_child_nodes(n))

    def kwarg_param(self, scope: ast.AST) -> str | None:
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and scope.args.kwarg:
            return scope.args.kwarg.arg
        return None

    def resolve_mapping(self, expr: ast.AST, at: ast.AST, depth: int = 0) -> KeySet:
        if depth > 8:
            return KeySet({}, ["depth"])
        if isinstance(expr, ast.Dict):
            ks = KeySet({}, [])
            for k, v in zip(expr.keys, expr.values, strict=True):
                if k is None:
                    sub = self.resolve_mapping(v, at, depth + 1)
                    ks.keys.update(sub.keys)
                    ks.dynamic += sub.dynamic
                    ks.from_caller |= sub.from_caller
                elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                    ks.keys[k.value] = "dict"
                else:
                    ks.dynamic.append(ast.unparse(k))
            return ks
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "dict"
        ):
            ks = KeySet({}, [])
            for a in expr.args:
                sub = self.resolve_mapping(a, at, depth + 1)
                ks.keys.update(sub.keys)
                ks.dynamic += sub.dynamic
                ks.from_caller |= sub.from_caller
            for kw in expr.keywords:
                if kw.arg is None:
                    sub = self.resolve_mapping(kw.value, at, depth + 1)
                    ks.keys.update(sub.keys)
                    ks.dynamic += sub.dynamic
                    ks.from_caller |= sub.from_caller
                else:
                    ks.keys[kw.arg] = "dict()"
            return ks
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            ks = KeySet({}, [])
            for side in (expr.left, expr.right):
                sub = self.resolve_mapping(side, at, depth + 1)
                ks.keys.update(sub.keys)
                ks.dynamic += sub.dynamic
                ks.from_caller |= sub.from_caller
            return ks
        if isinstance(expr, ast.DictComp) and isinstance(expr.generators[0].iter, ast.Call):
            it = expr.generators[0].iter
            if isinstance(it.func, ast.Attribute) and it.func.attr == "items":
                return self.resolve_mapping(it.func.value, at, depth + 1)
        if isinstance(expr, ast.Name):
            for scope in self.scope_chain(at):
                if self.kwarg_param(scope) == expr.id:
                    return KeySet({}, [], from_caller=True)
                ks = self._resolve_name_in(expr.id, scope, at, depth)
                if ks is not None:
                    return ks
        return KeySet({}, [ast.unparse(expr)])

    def _resolve_name_in(self, name: str, scope: ast.AST, at: ast.AST, depth: int) -> KeySet | None:
        found = False
        ks = KeySet({}, [])
        for n in sorted(
            self.own_nodes(scope),
            key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
        ):
            value = None
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets
            ):
                value = n.value
            elif (
                isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == name
                and n.value
            ):
                value = n.value
            if value is not None:
                found = True
                ks = self.resolve_mapping(value, n, depth + 1)
                continue
            if not found:
                continue
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and _attr_chain(n.func.value) == [name]
            ):
                if n.func.attr == "update":
                    for a in n.args:
                        sub = self.resolve_mapping(a, n, depth + 1)
                        ks.keys.update(sub.keys)
                        ks.dynamic += sub.dynamic
                        ks.from_caller |= sub.from_caller
                    for kw in n.keywords:
                        if kw.arg:
                            ks.keys[kw.arg] = "update"
                elif n.func.attr == "setdefault" and n.args and isinstance(n.args[0], ast.Constant):
                    ks.keys[str(n.args[0].value)] = "setdefault"
                elif n.func.attr == "pop" and n.args and isinstance(n.args[0], ast.Constant):
                    ks.keys.pop(str(n.args[0].value), None)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Subscript) and _attr_chain(t.value) == [name]:
                        if isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                            ks.keys[t.slice.value] = "subscript"
                        else:
                            ks.dynamic.append(ast.unparse(t))
        return ks if found else None

    def call_keys(self, call: ast.Call) -> KeySet:
        ks = KeySet({}, [])
        for kw in call.keywords:
            if kw.arg is not None:
                ks.keys[kw.arg] = "keyword"
            else:
                sub = self.resolve_mapping(kw.value, call)
                for k in sub.keys:
                    ks.keys.setdefault(k, f"**{ast.unparse(kw.value)}")
                ks.dynamic += sub.dynamic
                ks.from_caller |= sub.from_caller
        return ks

    def scope_mutations(self, call: ast.Call) -> set[str]:
        """Fields set after construction via set_params(...) or attribute assignment."""
        parent = self.parents.get(call)
        target = None
        if (
            isinstance(parent, ast.Assign)
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
        ):
            target = parent.targets[0].id
        if target is None:
            return set()
        out = set()
        for n in self.own_nodes(self.scope_of(call)):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "set_params"
                and _attr_chain(n.func.value) == [target]
            ):
                out |= {kw.arg for kw in n.keywords if kw.arg in FIELDS}
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if (
                        isinstance(t, ast.Attribute)
                        and t.attr in FIELDS
                        and _attr_chain(t.value) == [target]
                    ):
                        out.add(t.attr)
        return out

    def scope_fits(self, node: ast.AST) -> bool:
        for n in self.own_nodes(self.scope_of(node)):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("fit", "fit_transform")
            ):
                return True
            if (
                isinstance(n, ast.Call)
                and py_is_sink(n, self.umap_names, self.modules) == "fit_umap"
            ):
                return True
        return False


def audit_python_source(
    path: str, source: str, line_offset: int = 0, embedded: bool = False
) -> list[Site]:
    an = PyAnalyzer(path, source, line_offset)
    sinks = [
        n
        for n in ast.walk(an.tree)
        if isinstance(n, ast.Call) and py_is_sink(n, an.umap_names, an.modules)
    ]
    sites: list[Site] = []

    # Helpers: non-test functions with a **kwargs parameter that reach a sink using it.
    helper_defs: dict[str, tuple[ast.AST, ast.Call, KeySet]] = {}
    for call in sinks:
        fn = an.outer_function(call)
        if fn is None or fn.name.startswith("test_"):
            continue
        ks = an.call_keys(call)
        if ks.from_caller:
            helper_defs[fn.name] = (fn, call, ks)

    helper_call_nodes = [
        n
        for n in ast.walk(an.tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in helper_defs
    ]
    helper_reached = {name: False for name in helper_defs}
    for n in helper_call_nodes:
        caller = an.outer_function(n)
        assert isinstance(n.func, ast.Name)
        helper_reached[n.func.id] |= (
            an.scope_fits(n)
            or (caller is not None and caller.name in helper_defs)
            or isinstance(an.parents.get(n), ast.Attribute)
        )

    def fn_name(node: ast.AST) -> tuple[str | None, bool]:
        fn = an.outer_function(node)
        name = fn.name if fn else None
        return name, bool(name and name.startswith("test_"))

    for call in sinks:
        sink = py_is_sink(call, an.umap_names, an.modules)
        ks = an.call_keys(call)
        mutated = an.scope_mutations(call)
        name, is_test = fn_name(call)
        helper_def = name if name in helper_defs and helper_defs[name][1] is call else None
        states, sources = {}, {}
        for f in FIELDS:
            if f in ks.keys:
                states[f], sources[f] = "explicit", ks.keys[f]
            elif f in mutated:
                states[f], sources[f] = "mutated", "set_params/attribute"
            elif ks.dynamic:
                states[f], sources[f] = "dynamic", ",".join(ks.dynamic)
            elif ks.from_caller:
                states[f], sources[f] = "caller", "**kwargs from caller"
            else:
                states[f], sources[f] = "default", f"{sink} default"
        if helper_def is not None:
            reaches = helper_reached[helper_def]
        else:
            parent = an.parents.get(call)
            reaches = (
                sink == "fit_umap"
                or an.scope_fits(call)
                or (isinstance(parent, ast.Attribute) and parent.attr in ("fit", "fit_transform"))
            )
        sites.append(
            Site(
                path=path,
                line=call.lineno + line_offset,
                language="python-embedded" if embedded else "python",
                kind="sink_call",
                function=name,
                is_test=is_test,
                text=ast.unparse(call)[:400],
                fields=states,
                sources=sources,
                helper_definition=helper_def,
                reaches_sink=reaches,
            )
        )

    for call in helper_call_nodes:
        assert isinstance(call.func, ast.Name)
        hname = call.func.id
        _, _hcall, hks = helper_defs[hname]
        ks = an.call_keys(call)
        mutated = an.scope_mutations(call)
        name, is_test = fn_name(call)
        states, sources = {}, {}
        for f in FIELDS:
            if f in ks.keys:
                states[f], sources[f] = "explicit", ks.keys[f]
            elif f in mutated:
                states[f], sources[f] = "mutated", "set_params/attribute"
            elif ks.dynamic:
                states[f], sources[f] = "dynamic", ",".join(ks.dynamic)
            elif ks.from_caller and f not in hks.keys:
                states[f], sources[f] = "caller", "**kwargs from caller"
            elif f in hks.keys:
                states[f], sources[f] = "helper", f"helper:{hname}"
            elif hks.dynamic:
                states[f], sources[f] = "dynamic", f"helper:{hname}:" + ",".join(hks.dynamic)
            else:
                states[f], sources[f] = "default", f"helper:{hname} leaves default"
        parent = an.parents.get(call)
        reaches = an.scope_fits(call) or isinstance(parent, ast.Attribute) or (name in helper_defs)
        sites.append(
            Site(
                path=path,
                line=call.lineno + line_offset,
                language="python-embedded" if embedded else "python",
                kind="helper_call",
                function=name,
                is_test=is_test,
                text=ast.unparse(call)[:400],
                fields=states,
                sources=sources,
                reaches_sink=reaches,
                helper_ref=f"{path}::{hname}",
            )
        )

    # Embedded Python programs in string constants (e.g. subprocess worker scripts).
    for n in ast.walk(an.tree):
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and re.search(r"\b(UMAP|fit_umap)\(", n.value)
            and "\n" in n.value
        ):
            try:
                sites += audit_python_source(
                    path, n.value, line_offset + n.lineno - 1, embedded=True
                )
            except SyntaxError:
                sites += audit_text(
                    path, n.value, line_offset + n.lineno - 1, "python-embedded-text"
                )
    return sites


# ---------------------------------------------------------------------------- Text


TEXT_CALL_RE = re.compile(
    r"(?<![\w.])(?:cuscan\.)?(UMAP|fit_umap)\s*\(|(?<![\w])_cuscan\.fit_umap\s*\("
)


def audit_text(path: str, text: str, line_offset: int, language: str) -> list[Site]:
    sites = []
    for m in TEXT_CALL_RE.finditer(text):
        open_paren = text.index("(", m.start())
        depth, j = 0, open_paren
        while j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        args = text[open_paren : j + 1]
        line = text.count("\n", 0, m.start()) + 1 + line_offset
        states, sources = {}, {}
        for f in FIELDS:
            if re.search(rf"\b{f}\s*=", args):
                states[f], sources[f] = "explicit", "keyword"
            elif "**" in args:
                states[f], sources[f] = "dynamic", "** in text call"
            else:
                states[f], sources[f] = "default", "text call"
        sites.append(
            Site(
                path,
                line,
                language,
                "text_call",
                None,
                False,
                text[m.start() : j + 1][:400],
                states,
                sources,
            )
        )
    return sites


def audit_markdown(path: str, text: str) -> list[Site]:
    sites: list[Site] = []
    fence = re.compile(r"^```([\w+-]*)[^\n]*\n(.*?)^```", re.M | re.S)
    covered: list[tuple[int, int]] = []
    for m in fence.finditer(text):
        lang, body = m.group(1).lower(), m.group(2)
        offset = text.count("\n", 0, m.start(2))
        covered.append((m.start(), m.end()))
        if lang == "rust" and ("UMAPConfig" in body):
            rf = load_rust(path, body)
            for s in audit_rust([rf]):
                s.line += offset
                s.language = "markdown-rust"
                sites.append(s)
        elif lang in ("python", "py", "pycon"):
            try:
                sites += [
                    dataclasses.replace(s, language="markdown-python")
                    for s in audit_python_source(path, body, offset)
                ]
            except SyntaxError:
                sites += audit_text(path, body, offset, "markdown-text")
        else:
            sites += audit_text(path, body, offset, "markdown-text")
    rest = list(text)
    for a, b in covered:
        for k in range(a, b):
            if rest[k] != "\n":
                rest[k] = " "
    sites += audit_text(path, "".join(rest), 0, "markdown-text")
    return sites


# ---------------------------------------------------------------------------- Driver


def iter_files(root: Path) -> Iterator[Path]:
    for r in ROOTS:
        p = root / r
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not (set(f.relative_to(root).parts) & SKIP_DIRS):
                    yield f


def classify(site: Site) -> None:
    for entry in ALLOWLIST:
        if entry.path != site.path:
            continue
        if entry.function is not None and entry.function != site.function:
            continue
        if entry.match is not None and entry.match not in site.text:
            continue
        site.allow = {"category": entry.category, "reason": entry.reason}
        break
    uncovered = [f for f in FIELDS if site.fields[f] not in COVERED_STATES]
    unresolved = [f for f in FIELDS if site.fields[f] in ("dynamic", "caller")]
    if site.helper_definition is not None and all(
        site.fields[f] in COVERED_STATES | {"caller"} for f in FIELDS
    ):
        uncovered = [f for f in FIELDS if site.fields[f] not in COVERED_STATES]
    site.missing_pins = [f for f in FIELDS if site.fields[f] in ("default", "unset_missing")]
    if not uncovered:
        site.status = "pinned"
    elif site.allow is not None:
        site.status = "allowlisted"
    elif (
        site.helper_definition is not None
        and not site.missing_pins
        and set(uncovered) <= {f for f in FIELDS if site.fields[f] == "caller"}
    ):
        # Python helper passing the field through from its callers; the callers are audited.
        site.status = "helper_passthrough"
    elif unresolved and not site.missing_pins:
        site.status = "unresolved"
    elif not site.reaches_sink:
        site.status = "unreached_unallowlisted"
    else:
        site.status = "unpinned"


FAILING = {"unpinned", "unresolved", "unreached_unallowlisted"}


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(root: Path) -> dict[str, Any]:
    rust_files: list[RustFile] = []
    sites: list[Site] = []
    errors: list[dict[str, str]] = []
    digest = hashlib.sha256()
    scanned = 0
    for f in iter_files(root):
        rel = f.relative_to(root).as_posix()
        if f.suffix not in (".rs", ".py", ".md", ".yml", ".yaml"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            errors.append({"path": rel, "error": repr(exc)})
            continue
        if not re.search(r"UMAPConfig|\bUMAP\s*\(|fit_umap", text):
            continue
        scanned += 1
        digest.update(rel.encode() + b"\0" + hashlib.sha256(text.encode()).digest())
        if f.suffix == ".rs":
            rust_files.append(load_rust(rel, text))
        elif f.suffix == ".py":
            try:
                sites += audit_python_source(rel, text)
            except SyntaxError as exc:
                errors.append({"path": rel, "error": f"SyntaxError: {exc}"})
                sites += audit_text(rel, text, 0, "python-text")
        elif f.suffix == ".md":
            sites += audit_markdown(rel, text)
        else:
            sites += audit_text(rel, text, 0, "yaml-text")
    sites += audit_rust(rust_files)
    sites.sort(key=lambda s: (s.path, s.line, s.kind))
    for s in sites:
        classify(s)

    by_status: dict[str, int] = {}
    by_file: dict[str, dict[str, int]] = {}
    for s in sites:
        by_status[s.status] = by_status.get(s.status, 0) + 1
        by_file.setdefault(s.path, {}).setdefault(s.status, 0)
        by_file[s.path][s.status] += 1
    helpers = sorted(
        {(s.path, s.helper_definition) for s in sites if s.helper_definition is not None}
    )
    helper_report = []
    for path, name in helpers:
        defs = [s for s in sites if s.path == path and s.helper_definition == name]
        callers = [
            {"path": s.path, "line": s.line, "function": s.function, "status": s.status}
            for s in sites
            if s.helper_ref == f"{path}::{name}"
        ]
        helper_report.append(
            {
                "path": path,
                "helper": name,
                "fields": {f: sorted({d.fields[f] for d in defs}) for f in FIELDS},
                "reaches_sink": any(d.reaches_sink for d in defs),
                "callers": callers,
            }
        )
    failing = [s for s in sites if s.status in FAILING]
    return {
        "schema": SCHEMA,
        "head": git_head(root),
        "roots": list(ROOTS),
        "files_with_candidates": scanned,
        "source_digest": digest.hexdigest(),
        "old_defaults": OLD_DEFAULTS,
        "clean": not failing and not errors,
        "summary": {"sites": len(sites), "failing": len(failing), "by_status": by_status},
        "by_file": by_file,
        "helpers": helper_report,
        "allowlist": [dataclasses.asdict(a) for a in ALLOWLIST],
        "unused_allowlist": [
            dataclasses.asdict(a)
            for a in ALLOWLIST
            if not any(
                s.allow and s.path == a.path and (a.function is None or s.function == a.function)
                for s in sites
            )
        ],
        "errors": errors,
        "limitations": LIMITATIONS,
        "sites": [dataclasses.asdict(s) for s in sites],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 6 default audit",
        "",
        f"- head: `{report['head']}`",
        f"- clean: **{report['clean']}**",
        f"- sites: {report['summary']['sites']}, failing: {report['summary']['failing']}",
        f"- by status: {json.dumps(report['summary']['by_status'], sort_keys=True)}",
        "",
        "| status | file:line | function | kind | optimizer | init | missing | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["sites"]:
        note = (
            s["allow"]["category"]
            if s["allow"]
            else (f"defines {s['helper_definition']}" if s["helper_definition"] else "")
        )
        if not s["reaches_sink"]:
            note = (note + " no-sink").strip()
        opt = f"{s['fields']['optimizer']} ({s['sources']['optimizer']})"
        ini = f"{s['fields']['init']} ({s['sources']['init']})"
        lines.append(
            f"| {s['status']} | {s['path']}:{s['line']} | {s['function'] or ''} | "
            f"{s['kind']} | {opt} | {ini} | {','.join(s['missing_pins'])} | {note} |"
        )
    if report["errors"]:
        lines += ["", "## Errors", *[f"- {e['path']}: {e['error']}" for e in report["errors"]]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: <root>/artifacts/umap06_p6/audit"
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report = run(root)
    if not args.no_write:
        out = (args.out_dir or root / "artifacts/umap06_p6/audit").resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out / "audit.md").write_text(render_markdown(report), encoding="utf-8")
    if not args.quiet:
        for s in report["sites"]:
            if s["status"] in FAILING:
                print(
                    f"{s['status']}: {s['path']}:{s['line']} {s['function'] or ''} "
                    f"missing={','.join(s['missing_pins']) or '-'} "
                    f"optimizer={s['fields']['optimizer']} init={s['fields']['init']}"
                )
        print(json.dumps({"clean": report["clean"], **report["summary"]}, sort_keys=True))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
