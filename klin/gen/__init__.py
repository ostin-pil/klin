"""Generation adapters, and the check that runs before the GPU does.

Driving a sampler is not what klin is for, and a shell script does it in eighty
lines. What a script cannot do is answer the question that matters before the
work happens rather than months after it.

**The licence posture of a graph is knowable before it runs.** A workflow names
its checkpoint and its LoRA stack, `klin.index` already resolves those names to
ledger records by filesystem identity, and `klin.policy` already applies the
project's rules to records. So the whole chain exists: klin can say "this graph
uses a non-commercial model, so nothing it produces can ship" before a single
step is sampled, and `--check` refuses to queue at all.

That inverts how this normally goes. The alternative is what this project
actually did: generate 1,384 plates over several weeks, then discover that 66 of
them came from FLUX.1-dev and cannot be used. The models were the same either
way; only the moment of finding out changed.

**Raw output does not enter the ledger.** The consuming project's rule is that
no model weights and no raw generation output ever reach the repository, and
only a conformed asset and its ledger line do. A ledger record per generated
image would put thousands of rows describing throwaway plates into a file that
is meant to describe shipped assets. Generated images go to the index, which is
derived and disposable and lives outside every checkout; a record appears when
an image becomes an asset, and that is a different verb.

Adapters are discovered the way `klin.fetch` discovers vendors: a module here
declaring `NAME` and `configure` becomes a subcommand, so a second generator is
a new file and no edit anywhere else.
"""

import importlib
import io
import json
import os
import pkgutil

from .. import index, ledger, manifest, policy
from ..fetch import Context


class GenError(Exception):
    pass


def adapters():
    """Every generation adapter in this package, by name."""
    found = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module("%s.%s" % (__name__, info.name))
        if getattr(module, "NAME", None) and hasattr(module, "configure"):
            found[module.NAME] = module
    return found


def configure(parser):
    """Wire every discovered adapter in as a subcommand."""
    kinds = parser.add_subparsers(dest="generator")
    for name in sorted(adapters()):
        module = adapters()[name]
        sub = kinds.add_parser(name, help=getattr(module, "HELP", None))
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="resolve and check, but queue nothing",
        )
        sub.add_argument(
            "--check",
            action="store_true",
            help="refuse to run when the output could not ship",
        )
        sub.add_argument(
            "--no-index",
            action="store_true",
            help="leave the outputs out of the index",
        )
        module.configure(sub)
        sub.set_defaults(func=_run_adapter, module=module)
    return parser


def _run_adapter(args, stream):
    data = {}
    path = args.manifest or os.path.join(args.repo, manifest.DEFAULT_MANIFEST)
    if os.path.isfile(path):
        data = manifest.load(path)
    return args.module.run(args, Context(args, data, stream))


def load_graph(path):
    """An API-format workflow, which is what `/prompt` accepts.

    The other export the ComfyUI UI offers is the editor's own format, with
    `nodes` and `links` arrays. It looks close enough to try and fails in a way
    that reads like a klin bug, so it is named here.
    """
    if not os.path.isfile(path):
        raise GenError("no workflow at %s" % path)
    try:
        graph = json.loads(io.open(path, encoding="utf-8").read())
    except ValueError as exc:
        raise GenError("%s does not parse: %s" % (path, exc))
    if not isinstance(graph, dict):
        raise GenError("%s is not a JSON object" % path)
    if "nodes" in graph and "links" in graph:
        raise GenError(
            "%s is the editor's workflow format, not the API format. In "
            "ComfyUI use Workflow > Export (API) rather than Export." % path
        )
    if not graph:
        raise GenError("%s is empty" % path)
    return graph


def weights_in(graph):
    """Every weight name a graph loads, as `(name, role)`.

    Read from the whole graph rather than from the chain behind one sampler,
    because this runs before anything is generated and the question is what the
    run may touch, not what one output turned out to use.
    """
    from ..index import comfy

    found = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        kind = node.get("class_type")
        inputs = node.get("inputs") or {}
        if kind in comfy.MODEL_LOADERS:
            key, role = comfy.MODEL_LOADERS[kind]
            if inputs.get(key):
                found.append((inputs[key], role))
        elif kind in comfy.LORA_LOADERS:
            key, strength_key = comfy.LORA_LOADERS[kind]
            # A slot at strength zero contributes nothing, and reporting it as
            # a licence risk would train the reader to ignore this output.
            if inputs.get(key) and inputs.get(strength_key) not in (0, 0.0):
                found.append((inputs[key], "lora"))
    return found


def posture(ctx, used_weights):
    """What the project's policy says about the models a run would use.

    Takes `(name, role)` pairs rather than a graph. Only one generator in the
    world describes its work as a node graph, and asking the rest to invent one
    to reach this would be the tail wagging the dog; what every generator can
    say is which weights it is about to load.

    Returns `(rows, findings, ok)`. `ok` is false when a rule fails or when a
    weight cannot be traced, because an untraceable model is exactly the case a
    ship gate exists to catch and treating it as clean would invert that.
    """
    data = ctx.manifest
    try:
        records = ledger.load(ctx.ledger_path())
    except manifest.ManifestError:
        records = []
    models = index.model_map(index.models_dir(data), records)
    by_id = dict((r["id"], r) for r in records)

    rows = []
    used = []
    ok = True
    for name, role in used_weights:
        entry = index.lookup(models, name)
        record = by_id.get(entry["record"]) if entry and entry.get("record") else None
        if record is None:
            rows.append({"name": name, "role": role, "record": None, "families": []})
            ok = False
            continue
        used.append(record)
        rows.append(
            {
                "name": name,
                "role": role,
                "record": record["id"],
                "licence": ledger.field(record, "licence.id"),
                "families": sorted(policy.families(record)),
            }
        )

    findings = []
    if used:
        findings = policy.evaluate(
            used, manifest.rules(data) if data else [],
            manifest.build_facts(data) if data else {},
            ship=True,
        )
        if policy.failed(findings):
            ok = False
    return rows, findings, ok
