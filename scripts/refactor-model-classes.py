#!/usr/bin/env python3
"""
Convert the three large switch(arch) statements in src/llama-model.cpp into a
per-architecture class hierarchy rooted at llm_arch_model_i.

For every LLM_ARCH_* the script emits a `llama_model_<stem>` class with:

    struct llama_model_qwen3 : public llm_arch_model_i {
        llama_model_qwen3(const struct llama_model_params & params) : llm_arch_model_i(params) {};
        void load_hparams(llama_model_loader & ml) override;
        void load_tensors(llama_model_loader & ml) override;

        struct graph : public llm_graph_context {
            graph(const llama_model & model, const llm_graph_params & params);
        };

        std::unique_ptr<llm_graph_context> build_graph_context(const llm_graph_params & params) const override;
    };

The script is designed to be run from the repository root. It uses tree-sitter
to parse C++ so that templates, nested switches, raw strings and comments are
handled correctly.

Usage:
    python scripts/refactor-model-classes.py [--dry-run] [--root .]

Dependencies:
    pip install tree_sitter tree_sitter_cpp
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("refactor-model-classes")


# ---------------------------------------------------------------------------
# tree-sitter setup
# ---------------------------------------------------------------------------

def _load_parser():
    try:
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser
    except ImportError as e:
        sys.exit(
            f"missing dependency: {e.name}.\n"
            "install with:  pip install tree_sitter tree_sitter_cpp"
        )
    return Parser(Language(tscpp.language()))


PARSER = None

def parser():
    global PARSER
    if PARSER is None:
        PARSER = _load_parser()
    return PARSER


def parse(src_bytes: bytes):
    return parser().parse(src_bytes)


def text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


def find_descendants(node, type_: str):
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == type_:
            yield n
        stack.extend(reversed(n.children))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Arch:
    enum: str             # e.g. "LLM_ARCH_QWEN3"
    name: str             # e.g. "jina-bert-v2" (from LLM_ARCH_NAMES)
    file_stem: str        # e.g. "jina-bert-v2"  (used for filesystem, keeps hyphens)
    ident: str            # e.g. "jina_bert_v2"  (used for C++ identifiers)
    class_name: str       # e.g. "llama_model_jina_bert_v2"
    hparams_body: Optional[str] = None
    tensors_body: Optional[str] = None
    graph_body:   Optional[str] = None


@dataclass
class SwitchCase:
    labels: List[str]           # e.g. ["LLM_ARCH_LLAMA", "LLM_ARCH_LLAMA_EMBED"]
    body:   str                 # raw source text of the body (between the last `:` and the `break;`)


# ---------------------------------------------------------------------------
# Parse LLM_ARCH_NAMES (enum -> string) from src/llama-arch.cpp
# ---------------------------------------------------------------------------

NAME_MAP_RE = re.compile(
    r'\{\s*(LLM_ARCH_[A-Z0-9_]+)\s*,\s*"([^"]*)"\s*\}',
    re.MULTILINE,
)


def parse_arch_names(arch_cpp: Path) -> Dict[str, str]:
    txt = arch_cpp.read_text()
    # Only accept the block after "LLM_ARCH_NAMES ="
    start = txt.find("LLM_ARCH_NAMES")
    if start < 0:
        sys.exit(f"could not find LLM_ARCH_NAMES in {arch_cpp}")
    end = txt.find("};", start)
    if end < 0:
        sys.exit(f"could not find end of LLM_ARCH_NAMES in {arch_cpp}")
    block = txt[start:end]
    return dict(NAME_MAP_RE.findall(block))


def file_stem_of(name: str) -> str:
    # filesystem stem: keep hyphens as-is (matches existing src/models/*.cpp naming)
    return re.sub(r"[^a-z0-9_-]", "-", name.lower())


def ident_of(name: str) -> str:
    # C++ identifier suffix: hyphens become underscores
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


# ---------------------------------------------------------------------------
# Extract the three switch statements from src/llama-model.cpp
# ---------------------------------------------------------------------------

TARGET_METHODS = {"load_hparams", "load_tensors", "build_graph"}


def _unwrap_declarator(decl):
    """Peel pointer_declarator / reference_declarator wrappers to reach the function_declarator."""
    while decl is not None and decl.type in ("pointer_declarator", "reference_declarator"):
        inner = decl.child_by_field_name("declarator")
        if inner is None:
            return None
        decl = inner
    return decl


def find_method(root, src: bytes, class_name: str, method_name: str):
    """Find llama_model::<method_name>. Returns the function_definition node."""
    for fn in find_descendants(root, "function_definition"):
        declarator = _unwrap_declarator(fn.child_by_field_name("declarator"))
        if declarator is None or declarator.type != "function_declarator":
            continue
        qname = declarator.child_by_field_name("declarator")
        if qname is None or qname.type != "qualified_identifier":
            continue
        scope = qname.child_by_field_name("scope")
        name  = qname.child_by_field_name("name")
        if scope is None or name is None:
            continue
        if text(scope, src) == class_name and text(name, src) == method_name:
            return fn
    return None


def find_arch_switch(fn_node, src: bytes):
    """Find the `switch (arch) { ... }` inside a function body. Returns the switch_statement node."""
    for sw in find_descendants(fn_node, "switch_statement"):
        cond = sw.child_by_field_name("condition")
        if cond is None:
            continue
        # condition_clause -> children like '(' identifier ')'
        ids = [c for c in cond.children if c.type == "identifier"]
        if ids and text(ids[0], src) == "arch":
            return sw
    return None


def extract_cases(switch_node, src: bytes) -> List[SwitchCase]:
    """Walk case_statement children. Groups fallthrough labels with the next bodied case."""
    body = None
    for c in switch_node.children:
        if c.type == "compound_statement":
            body = c
            break
    if body is None:
        return []

    cases = []
    pending_labels: List[str] = []
    for child in body.children:
        if child.type != "case_statement":
            continue

        # `case LLM_ARCH_X:` labels include a `value` field on case_statement
        value = child.child_by_field_name("value")
        if value is None:
            # `default:` case — skip for our purposes
            continue

        label = text(value, src)

        # Fallthrough: case_statement with only {case, value, ":"} and no following stmts
        # Statement children come after the `:` token.
        colon_idx = None
        for i, ch in enumerate(child.children):
            if ch.type == ":":
                colon_idx = i
                break
        stmts = child.children[colon_idx + 1:] if colon_idx is not None else []

        if not stmts:
            pending_labels.append(label)
            continue

        labels = pending_labels + [label]
        pending_labels = []

        # Drop the trailing `break;` — inside the new non-switch method it's a
        # syntax error. Keep everything before it.
        body_stmts = stmts
        if body_stmts and body_stmts[-1].type == "break_statement":
            body_stmts = body_stmts[:-1]
        if not body_stmts:
            cases.append(SwitchCase(labels=labels, body=""))
            continue

        start = body_stmts[0].start_byte
        end   = body_stmts[-1].end_byte
        body_text = src[start:end].decode("utf8", errors="replace")
        cases.append(SwitchCase(labels=labels, body=body_text))

    return cases


def extract_switches(model_cpp: Path) -> Tuple[Dict[str, List[SwitchCase]], bytes, object]:
    src = model_cpp.read_bytes()
    tree = parse(src)
    out: Dict[str, List[SwitchCase]] = {}
    for method in TARGET_METHODS:
        fn = find_method(tree.root_node, src, "llama_model", method)
        if fn is None:
            sys.exit(f"could not find llama_model::{method} in {model_cpp}")
        sw = find_arch_switch(fn, src)
        if sw is None:
            sys.exit(f"could not find `switch (arch)` in llama_model::{method}")
        out[method] = extract_cases(sw, src)
        logger.info("parsed %s: %d case groups", method, len(out[method]))
    return out, src, tree


# ---------------------------------------------------------------------------
# Build the per-arch map from parsed switches
# ---------------------------------------------------------------------------

def build_archs(names: Dict[str, str], switches: Dict[str, List[SwitchCase]]) -> List[Arch]:
    archs: Dict[str, Arch] = {}
    for enum, name in names.items():
        if enum in ("LLM_ARCH_UNKNOWN", "LLM_ARCH_CLIP"):
            continue
        fs = file_stem_of(name)
        ident = ident_of(name)
        archs[enum] = Arch(enum=enum, name=name, file_stem=fs, ident=ident,
                           class_name=f"llama_model_{ident}")

    def assign(method: str, attr: str):
        for case in switches[method]:
            for label in case.labels:
                if label not in archs:
                    logger.warning("%s references unknown arch %s; skipping", method, label)
                    continue
                setattr(archs[label], attr, case.body)

    assign("load_hparams", "hparams_body")
    assign("load_tensors", "tensors_body")
    assign("build_graph",  "graph_body")

    return list(archs.values())


# ---------------------------------------------------------------------------
# Emit: per-arch header
# ---------------------------------------------------------------------------

HEADER_TEMPLATE = '''\
#pragma once

#include "../llama-arch-model.h"
#include "../llama-graph.h"

struct {class_name} : public llm_arch_model_i {{
    {class_name}(const struct llama_model_params & params) : llm_arch_model_i(params) {{}};

    void load_hparams(llama_model_loader & ml) override;
    void load_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {{
        graph(const llama_model & model, const llm_graph_params & params);
    }};

    std::unique_ptr<llm_graph_context> build_graph_context(const llm_graph_params & params) const override;
}};
'''


def emit_header(arch: Arch) -> str:
    return HEADER_TEMPLATE.format(class_name=arch.class_name)


# ---------------------------------------------------------------------------
# Emit: per-arch .cpp
#
# Strategy: leave the existing graph-builder constructor body alone, but rename
# its qualified name from `llm_build_<stem>::llm_build_<stem>` to
# `<class>::graph::graph`. Then append three new methods with bodies copied
# out of the switches.
# ---------------------------------------------------------------------------

def rename_graph_ctor(src: bytes, old_class: str, new_class: str) -> bytes:
    """Rewrite the existing graph-builder ctor's qualified name.

    Operates on the full source bytes. Uses tree-sitter to find the relevant
    function_definition nodes so we don't accidentally rewrite string literals.
    """
    tree = parse(src)
    edits: List[Tuple[int, int, bytes]] = []  # (start, end, replacement)

    for fn in find_descendants(tree.root_node, "function_definition"):
        declarator = _unwrap_declarator(fn.child_by_field_name("declarator"))
        if declarator is None or declarator.type != "function_declarator":
            continue
        qname = declarator.child_by_field_name("declarator")
        if qname is None or qname.type != "qualified_identifier":
            continue
        scope = qname.child_by_field_name("scope")
        name  = qname.child_by_field_name("name")
        if scope is None or name is None:
            continue
        scope_text = text(scope, src)
        name_text  = text(name, src)
        if scope_text == old_class and name_text == old_class:
            edits.append((qname.start_byte, qname.end_byte,
                          f"{new_class}::graph::graph".encode()))

    if not edits:
        return src

    # apply edits in reverse so offsets stay valid
    out = bytearray(src)
    for start, end, repl in sorted(edits, reverse=True):
        out[start:end] = repl
    return bytes(out)


def rewrite_make_unique(src: bytes, builder_to_class: Dict[str, str]) -> bytes:
    """Replace `std::make_unique<llm_build_X>(...)` (including templates) with the new graph class.

    For shared builders, builder_to_class maps `llm_build_bert` -> `llama_model_bert::graph`.
    """
    tree = parse(src)
    edits: List[Tuple[int, int, bytes]] = []

    for call in find_descendants(tree.root_node, "call_expression"):
        fn = call.child_by_field_name("function")
        if fn is None or fn.type != "template_function":
            continue
        name_node = fn.child_by_field_name("name")
        args_node = fn.child_by_field_name("arguments")
        if name_node is None or args_node is None:
            continue
        if text(name_node, src) != "std::make_unique":
            continue

        # args_node is a template_argument_list
        ta = [c for c in args_node.children if c.type == "type_descriptor"]
        if not ta:
            continue
        tt = ta[0]
        # The type_descriptor's child is usually a qualified_identifier or template_type
        inner = tt.children[0]
        inner_text = text(inner, src)

        # Strip any template params to find the base name
        base = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", inner_text)
        if not base:
            continue
        base_name = base.group(1)
        if base_name not in builder_to_class:
            continue

        # Replace the base name in-place, preserving any template args after it
        abs_start = inner.start_byte
        abs_end   = inner.start_byte + len(base_name)
        edits.append((abs_start, abs_end, builder_to_class[base_name].encode()))

    if not edits:
        return src
    out = bytearray(src)
    for start, end, repl in sorted(edits, reverse=True):
        out[start:end] = repl
    return bytes(out)


def append_arch_methods(src: bytes, arch: Arch) -> bytes:
    parts = [src]
    parts.append(b"\n\n")
    parts.append(f"void {arch.class_name}::load_hparams(llama_model_loader & ml) {{\n".encode())
    parts.append(_indent(arch.hparams_body or "/* no hparams */\n").encode())
    parts.append(b"\n}\n\n")

    parts.append(f"void {arch.class_name}::load_tensors(llama_model_loader & ml) {{\n".encode())
    parts.append(_indent(arch.tensors_body or "/* no tensors */\n").encode())
    parts.append(b"\n}\n\n")

    parts.append(
        f"std::unique_ptr<llm_graph_context> {arch.class_name}::build_graph_context(const llm_graph_params & params) const {{\n"
        "    std::unique_ptr<llm_graph_context> llm;\n".encode()
    )
    parts.append(_indent(arch.graph_body or "/* no graph */\n").encode())
    parts.append(b"\n    return llm;\n}\n")

    return b"".join(parts)


def _indent(body: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in body.splitlines())


# ---------------------------------------------------------------------------
# Emit: interface + factory
# ---------------------------------------------------------------------------

INTERFACE_HEADER = '''\
#pragma once

#include "llama.h"
#include "llama-arch.h"

#include <memory>

struct llama_model_loader;
struct llm_graph_context;
struct llm_graph_params;

struct llm_arch_model_i {
    const llama_model_params & params;

    explicit llm_arch_model_i(const llama_model_params & p) : params(p) {}
    virtual ~llm_arch_model_i() = default;

    virtual void load_hparams(llama_model_loader & ml) = 0;
    virtual void load_tensors(llama_model_loader & ml) = 0;
    virtual std::unique_ptr<llm_graph_context> build_graph_context(const llm_graph_params & params) const = 0;
};

std::unique_ptr<llm_arch_model_i> llm_arch_model_make(llm_arch arch, const llama_model_params & params);
'''


def emit_factory(archs: List[Arch]) -> str:
    lines = [
        '#include "llama-arch-model.h"',
        '',
    ]
    for a in archs:
        lines.append(f'#include "models/{a.file_stem}.h"')
    lines += [
        '',
        'std::unique_ptr<llm_arch_model_i> llm_arch_model_make(llm_arch arch, const llama_model_params & params) {',
        '    switch (arch) {',
    ]
    for a in archs:
        lines.append(f'        case {a.enum}: return std::make_unique<{a.class_name}>(params);')
    lines += [
        '        default: return nullptr;',
        '    }',
        '}',
        '',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rewrite src/llama-model.cpp: replace each switch with a factory delegation
# ---------------------------------------------------------------------------

DELEGATE_SNIPPETS = {
    "load_hparams": "    arch_impl = llm_arch_model_make(arch, params);\n    arch_impl->load_hparams(ml);",
    "load_tensors": "    arch_impl->load_tensors(ml);",
    "build_graph":  "    std::unique_ptr<llm_graph_context> llm = arch_impl->build_graph_context(params);",
}


def splice_out_switches(model_cpp: Path, dry_run: bool):
    src = model_cpp.read_bytes()
    tree = parse(src)

    # Collect byte ranges to replace: (start, end, replacement_bytes), in reverse order
    edits: List[Tuple[int, int, bytes]] = []

    for method in TARGET_METHODS:
        fn = find_method(tree.root_node, src, "llama_model", method)
        if fn is None:
            continue
        sw = find_arch_switch(fn, src)
        if sw is None:
            continue
        edits.append((sw.start_byte, sw.end_byte, DELEGATE_SNIPPETS[method].encode()))

    if not edits:
        logger.warning("no switches to splice out of %s", model_cpp)
        return

    out = bytearray(src)
    for start, end, repl in sorted(edits, reverse=True):
        out[start:end] = repl

    if dry_run:
        logger.info("[dry-run] would rewrite %s (%d bytes -> %d)", model_cpp, len(src), len(out))
    else:
        model_cpp.write_bytes(bytes(out))
        logger.info("rewrote %s", model_cpp)


# ---------------------------------------------------------------------------
# Minor header surgery
# ---------------------------------------------------------------------------

def patch_model_h(path: Path, dry_run: bool):
    """Add `std::unique_ptr<llm_arch_model_i> arch_impl;` to struct llama_model
    and drop the two stale TODO comments."""
    src = path.read_text()
    new = src
    new = new.replace(
        "    // TODO: move this to new llm_arch_model_i interface\n"
        "    llama_memory_i * create_memory(const llama_memory_params & params, const llama_cparams & cparams) const;\n",
        "    llama_memory_i * create_memory(const llama_memory_params & params, const llama_cparams & cparams) const;\n",
    )
    new = new.replace(
        "    // TODO: move this to new llm_arch_model_i interface\n"
        "    ggml_cgraph * build_graph(const llm_graph_params & params) const;\n",
        "    ggml_cgraph * build_graph(const llm_graph_params & params) const;\n",
    )
    # Add arch_impl member near the end of the public section.
    marker = "private:\n    llama_model_params params;"
    addition = (
        "    // per-arch implementation; owns load_hparams/load_tensors/build_graph logic\n"
        "    std::unique_ptr<llm_arch_model_i> arch_impl;\n\n"
        "private:\n    llama_model_params params;"
    )
    if marker in new and "arch_impl" not in new:
        new = new.replace(marker, addition)

    # Make sure the forward decl is visible.
    include_line = '#include "llama-arch-model.h"'
    if include_line not in new:
        new = new.replace(
            '#include "llama-arch.h"',
            '#include "llama-arch.h"\n' + include_line,
            1,
        )

    if new == src:
        return
    if dry_run:
        logger.info("[dry-run] would patch %s", path)
    else:
        path.write_text(new)
        logger.info("patched %s", path)


def patch_models_h(path: Path, archs: List[Arch], dry_run: bool):
    """Strip forward declarations for llm_build_<stem> that are now nested
    in llama_model_<stem>::graph. Keeps *_base helper structs."""
    src = path.read_text()
    lines = src.splitlines(keepends=True)
    stems = {a.ident for a in archs}

    out_lines: List[str] = []
    i = 0
    struct_re = re.compile(r"^\s*(template\s*<[^>]*>\s*)?struct\s+llm_build_([A-Za-z0-9_]+)\s*:")
    while i < len(lines):
        m = struct_re.match(lines[i])
        # Keep *_base helpers even if they match a stem
        if m and m.group(2) in stems and not m.group(2).endswith("_base"):
            # skip forward declaration up to matching '};'
            depth = 0
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
                if depth <= 0:
                    break
            # drop a trailing blank
            if i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        out_lines.append(lines[i])
        i += 1

    new = "".join(out_lines)
    if new == src:
        return
    if dry_run:
        logger.info("[dry-run] would prune %s", path)
    else:
        path.write_text(new)
        logger.info("pruned %s", path)


def patch_cmakelists(path: Path, archs: List[Arch], dry_run: bool):
    src = path.read_text()
    if "llama-arch-model.cpp" in src:
        return
    # Insert after llama-model.cpp
    inject = "            llama-arch-model.cpp\n"
    for a in archs:
        inject += f"            models/{a.file_stem}.h\n"
    new = src.replace(
        "llama-model.cpp\n",
        "llama-model.cpp\n" + inject,
        1,
    )
    if new == src:
        logger.warning("could not patch CMakeLists.txt (marker not found)")
        return
    if dry_run:
        logger.info("[dry-run] would patch %s", path)
    else:
        path.write_text(new)
        logger.info("patched %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."), help="repo root")
    ap.add_argument("--dry-run", action="store_true", help="don't write files, just log")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    root      = args.root.resolve()
    src       = root / "src"
    models    = src / "models"
    arch_h    = src / "llama-arch.h"
    arch_cpp  = src / "llama-arch.cpp"
    model_h   = src / "llama-model.h"
    model_cpp = src / "llama-model.cpp"
    models_h  = models / "models.h"
    cmake     = src / "CMakeLists.txt"

    for p in (arch_h, arch_cpp, model_h, model_cpp, models_h, cmake):
        if not p.is_file():
            sys.exit(f"missing expected file: {p}")

    names = parse_arch_names(arch_cpp)
    logger.info("found %d architectures in LLM_ARCH_NAMES", len(names))

    switches, _, _ = extract_switches(model_cpp)
    archs = build_archs(names, switches)
    archs.sort(key=lambda a: a.file_stem)

    # Warn about archs missing any body
    for a in archs:
        missing = [m for m, b in (
            ("load_hparams", a.hparams_body),
            ("load_tensors", a.tensors_body),
            ("build_graph",  a.graph_body),
        ) if b is None]
        if missing:
            logger.warning("%s missing bodies: %s", a.enum, ",".join(missing))

    # Skip archs with no switch bodies at all — nothing to emit.
    complete = [a for a in archs
                if a.hparams_body is not None
                and a.tensors_body is not None
                and a.graph_body is not None]
    skipped = [a.enum for a in archs if a not in complete]
    if skipped:
        logger.warning("skipping %d archs with missing bodies: %s",
                       len(skipped), ", ".join(skipped))

    builder_to_class = {
        f"llm_build_{a.ident}": f"{a.class_name}::graph"
        for a in complete
    }

    # 1. emit per-arch .h
    for a in complete:
        header_path = models / f"{a.file_stem}.h"
        header = emit_header(a)
        if args.dry_run:
            logger.info("[dry-run] would write %s", header_path)
        else:
            header_path.write_text(header)
            logger.info("wrote %s", header_path)

    # 2. rewrite each existing model cpp: rename ctor, add three new methods.
    #    Some archs share a .cpp with another arch (e.g. all BERT variants use
    #    models/bert.cpp) — if no dedicated file exists, we emit a stub that
    #    only contains the new class methods.
    for a in complete:
        cpp_path = models / f"{a.file_stem}.cpp"
        shared = not cpp_path.is_file()
        if shared:
            logger.info("no dedicated %s; will create stub", cpp_path)
            original = b""
        else:
            original = cpp_path.read_bytes()

        new = original
        include = f'#include "{a.file_stem}.h"\n'.encode()
        if include not in new:
            new = include + new

        new = rename_graph_ctor(new, f"llm_build_{a.ident}", a.class_name)
        new = rewrite_make_unique(new, builder_to_class)
        new = append_arch_methods(new, a)

        if args.dry_run:
            logger.info("[dry-run] would rewrite %s", cpp_path)
        else:
            cpp_path.write_bytes(new)
            logger.info("rewrote %s", cpp_path)

    # 3. interface + factory
    iface_path = src / "llama-arch-model.h"
    factory_path = src / "llama-arch-model.cpp"
    if args.dry_run:
        logger.info("[dry-run] would write %s", iface_path)
        logger.info("[dry-run] would write %s", factory_path)
    else:
        iface_path.write_text(INTERFACE_HEADER)
        factory_path.write_text(emit_factory(archs))
        logger.info("wrote %s and %s", iface_path, factory_path)

    # 4. rewrite the three switches in llama-model.cpp
    splice_out_switches(model_cpp, args.dry_run)

    # 5. header surgery
    patch_model_h(model_h, args.dry_run)
    patch_models_h(models_h, archs, args.dry_run)
    patch_cmakelists(cmake, archs, args.dry_run)

    logger.info("done. %d architectures processed.", len(archs))


if __name__ == "__main__":
    main()
