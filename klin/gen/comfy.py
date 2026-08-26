"""ComfyUI: patch a workflow by role, queue it, and index what comes back.

The template is the project's, not klin's. A workflow encodes an art direction,
and klin has no opinion about one; what it does is put this run's values into
the right slots and record what came out.

**Roles are located by traversal, not by node title.** The driver this was
ported from found the positive prompt by looking for a `CLIPTextEncode` whose
title contained "POSITIVE", which works until somebody renames a node in the
editor and then fails on a template that is perfectly valid. `klin.index.comfy`
already locates every role by walking the graph from the sampler, because it
had to read graphs nobody wrote for it. The same walk works for writing, so a
template needs no special titles and the reader and the writer cannot disagree
about which node is which.

**An unused LoRA slot is zeroed rather than unwired.** Rewiring a graph by hand
is where a sweep quietly starts producing something other than what its record
says it produced. Setting the strength to zero leaves the chain intact and the
contribution nil, which is both safer and legible in the output's own metadata.
"""

import json
import os
import time
import urllib.error
import urllib.request

from ..index import comfy as reader
from . import GenError, load_graph, posture

NAME = "comfy"
HELP = "a local ComfyUI server (needs the workflow in API format)"

#: Where the server lives. An environment variable rather than a manifest key,
#: for the same reason the cache is one: which port a local service listens on
#: is a fact about the machine.
URL_ENV = "COMFY_URL"
DEFAULT_URL = "http://127.0.0.1:8188"

#: Where the server writes. Needed to turn the filenames `/history` reports
#: into paths the index can scan.
OUTPUT_ENV = "COMFY_OUTPUT"

#: How often to ask whether a queued prompt has finished.
POLL_SECONDS = 1.5


def configure(parser):
    parser.add_argument("--workflow", required=True, help="API-format workflow json")
    parser.add_argument("--prompt", default=None, help="the positive prompt")
    parser.add_argument(
        "--prompt-file", default=None, help="read the positive prompt from a file"
    )
    parser.add_argument("--negative", default=None, help="the negative prompt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--size", default=None, help="WxH, for example 768x448")
    parser.add_argument(
        "--lora",
        action="append",
        default=None,
        metavar="NAME@STRENGTH",
        help="fills the template's LoRA slots in chain order; repeatable",
    )
    parser.add_argument("--prefix", default=None, help="filename prefix for outputs")
    parser.add_argument(
        "--count", type=int, default=1, help="queue this many, incrementing the seed"
    )
    parser.add_argument("--url", default=None, help="ComfyUI base url")
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds to wait per image"
    )


def base_url(args):
    return args.url or os.environ.get(URL_ENV) or DEFAULT_URL


def output_dir(ctx):
    value = os.environ.get(OUTPUT_ENV) or (ctx.manifest.get("comfy_output") or "")
    if not value:
        return None
    return os.path.normpath(os.path.expanduser(os.path.expandvars(str(value))))


def parse_lora(spec):
    """`name@strength`, or a bare name meaning full strength."""
    text = str(spec)
    if "@" in text:
        name, _, strength = text.rpartition("@")
        try:
            return name, float(strength)
        except ValueError:
            raise GenError("--lora %r: %r is not a number" % (spec, strength))
    return text, 1.0


def parse_size(value):
    text = str(value).lower().replace(" ", "")
    for sep in ("x", "*", ","):
        if sep in text:
            width, _, height = text.partition(sep)
            try:
                return int(width), int(height)
            except ValueError:
                break
    raise GenError("--size wants WxH, for example 768x448, got %r" % (value,))


def _sampler_of(graph):
    found = [
        node_id
        for node_id, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") in reader.SAMPLERS
    ]
    if not found:
        raise GenError(
            "this workflow has no sampler node, so klin cannot find the prompt, "
            "the seed or the latent. Classes it recognises: %s"
            % ", ".join(reader.SAMPLERS)
        )
    if len(found) > 1:
        raise GenError(
            "this workflow has %d sampler nodes and klin will not guess which "
            "one carries the run's settings. Split it, or patch it by hand."
            % len(found)
        )
    return found[0]


def _encoder_behind(graph, node_id, prefer="positive"):
    """Walk a conditioning input back to the text encoder that feeds it.

    `prefer` names which branch this walk came down, and it is not cosmetic. A
    `ControlNetApplyAdvanced` takes a positive *and* a negative conditioning,
    so a walk that always tried `positive` first followed the wrong edge for
    the negative and reported both prompts as living in the same node. Checking
    the matching key first keeps the two branches apart.
    """
    order = ("conditioning", "conditioning_to", prefer) + tuple(
        key for key in ("positive", "negative") if key != prefer
    )
    hops = 0
    seen = set()
    while node_id is not None and node_id not in seen and hops < reader.MAX_HOPS:
        seen.add(node_id)
        inputs, kind = reader.node_of(graph, node_id)
        if kind in reader.TEXT_ENCODERS:
            return node_id
        nxt = None
        for key in order:
            nxt = reader.link(inputs.get(key))
            if nxt:
                break
        node_id = nxt
        hops += 1
    return None


def _latent_behind(graph, node_id):
    seen = set()
    while node_id is not None and node_id not in seen:
        seen.add(node_id)
        inputs, kind = reader.node_of(graph, node_id)
        if kind in reader.LATENTS:
            return node_id
        node_id = reader.link(inputs.get("samples")) or reader.link(
            inputs.get("latent_image")
        )
    return None


def _lora_slots(graph, sampler_id):
    """The LoRA loaders actually wired into this sampler, in chain order."""
    slots = []
    seen = set()
    node_id = reader.link(reader.node_of(graph, sampler_id)[0].get("model"))
    while node_id is not None and node_id not in seen:
        seen.add(node_id)
        inputs, kind = reader.node_of(graph, node_id)
        if kind in reader.LORA_LOADERS:
            slots.append((node_id, kind))
        elif kind in reader.MODEL_LOADERS:
            break
        node_id = reader.link(inputs.get("model"))
    slots.reverse()
    return slots


def patch(graph, args, seed):
    """This run's values, in the slots the traversal found."""
    import copy

    graph = copy.deepcopy(graph)
    sampler_id = _sampler_of(graph)
    sampler = graph[sampler_id]["inputs"]

    sampler["seed" if "seed" in sampler else "noise_seed"] = seed
    if args.steps is not None:
        sampler["steps"] = args.steps
    if args.cfg is not None:
        sampler["cfg"] = args.cfg

    positive_id = _encoder_behind(graph, reader.link(sampler.get("positive")), "positive")
    negative_id = _encoder_behind(graph, reader.link(sampler.get("negative")), "negative")

    text = _prompt_text(args)
    if text is not None:
        if positive_id is None:
            raise GenError(
                "no text encoder is reachable from the sampler's positive "
                "input, so klin cannot place the prompt"
            )
        graph[positive_id]["inputs"]["text"] = text

    if args.negative is not None:
        if negative_id is None:
            raise GenError("no text encoder is reachable from the negative input")
        if negative_id == positive_id:
            # The Flux and Z-Image pattern: the negative is a ConditioningZeroOut
            # of the positive, so the graph has one encoder and no negative
            # prompt to write. Writing here would overwrite the positive, which
            # is what happened before this check existed, silently, leaving a
            # plate whose record said one prompt and whose pixels came from
            # another.
            raise GenError(
                "this workflow has no separate negative prompt: its negative "
                "conditioning is derived from the positive one (the "
                "ConditioningZeroOut pattern Flux and Z-Image use), so both "
                "resolve to node %s. Writing --negative there would overwrite "
                "the prompt. Drop --negative, or use a template with its own "
                "negative encoder." % positive_id
            )
        graph[negative_id]["inputs"]["text"] = args.negative

    if args.size:
        width, height = parse_size(args.size)
        node_id = _latent_behind(graph, reader.link(sampler.get("latent_image")))
        if node_id is None:
            raise GenError(
                "this workflow's latent does not come from an empty-latent "
                "node, which is normal for img2img and means --size has "
                "nothing to set: the size comes from the input image"
            )
        graph[node_id]["inputs"]["width"] = width
        graph[node_id]["inputs"]["height"] = height

    if args.lora:
        wanted = [parse_lora(spec) for spec in args.lora]
        slots = _lora_slots(graph, sampler_id)
        if len(slots) < len(wanted):
            raise GenError(
                "asked for %d LoRA(s) and this workflow wires %d slot(s) into "
                "its sampler. Add loaders to the template rather than having "
                "klin rewire the graph." % (len(wanted), len(slots))
            )
        for (node_id, kind), (name, strength) in zip(slots, wanted):
            name_key, strength_key = reader.LORA_LOADERS[kind]
            graph[node_id]["inputs"][name_key] = name
            graph[node_id]["inputs"][strength_key] = strength
        for node_id, kind in slots[len(wanted):]:
            _name_key, strength_key = reader.LORA_LOADERS[kind]
            graph[node_id]["inputs"][strength_key] = 0.0

    if args.prefix:
        for node in graph.values():
            if isinstance(node, dict) and node.get("class_type") == "SaveImage":
                node["inputs"]["filename_prefix"] = args.prefix

    return graph


def _prompt_text(args):
    if args.prompt_file:
        if not os.path.isfile(args.prompt_file):
            raise GenError("no prompt file at %s" % args.prompt_file)
        import io as _io

        return _io.open(args.prompt_file, encoding="utf-8").read().strip()
    return args.prompt


def post(url, path, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise GenError(
            "cannot reach ComfyUI at %s (%s). Start it, or point klin at it "
            "with --url or %s." % (url, exc, URL_ENV)
        )


def get(url, path):
    try:
        with urllib.request.urlopen(url + path, timeout=60) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise GenError("cannot reach ComfyUI at %s (%s)" % (url, exc))


def queue(url, graph, out_dir, timeout=900, sleep=None):
    """Queue one graph and wait. Returns the paths it wrote."""
    sleep = sleep or (lambda seconds: time.sleep(seconds))
    prompt_id = post(url, "/prompt", {"prompt": graph}).get("prompt_id")
    if not prompt_id:
        raise GenError("ComfyUI accepted the graph but returned no prompt_id")

    waited = 0.0
    while waited <= timeout:
        entry = (get(url, "/history/" + prompt_id) or {}).get(prompt_id)
        if entry:
            status = (entry.get("status") or {}).get("status_str")
            if status == "error":
                raise GenError(
                    "ComfyUI reported an error: %s"
                    % json.dumps(entry.get("status"))[:1200]
                )
            images = []
            for output in (entry.get("outputs") or {}).values():
                for image in output.get("images", []) or []:
                    name = image.get("filename")
                    if not name:
                        continue
                    parts = [out_dir] if out_dir else []
                    if image.get("subfolder"):
                        parts.append(image["subfolder"])
                    parts.append(name)
                    images.append(os.path.join(*parts) if out_dir else name)
            if images:
                return images
        sleep(POLL_SECONDS)
        waited += POLL_SECONDS
    raise GenError("timed out after %ss waiting for %s" % (timeout, prompt_id))


def run(args, ctx):
    graph = load_graph(args.workflow)
    rows, findings, ok = posture(ctx, graph)

    ctx.say("workflow %s" % args.workflow)
    for row in rows:
        if row["record"]:
            ctx.say(
                "  %-6s %-44s %s [%s]"
                % (
                    row["role"],
                    os.path.basename(str(row["name"]))[:44],
                    row.get("licence") or "(unrecorded)",
                    ", ".join(row["families"]) or "no family applies",
                )
            )
        else:
            ctx.say(
                "  %-6s %-44s no ledger record, so its terms are unknown"
                % (row["role"], os.path.basename(str(row["name"]))[:44])
            )
    ctx.say("")
    ctx.say("output could ship: %s" % ("yes" if ok else "NO"))
    for finding in findings:
        if finding.level == "fail":
            ctx.say("  %s: %s" % (finding.rule_id, finding.summary))
    ctx.say("")

    if args.check and not ok:
        ctx.say(
            "refusing to queue, because --check was given and this graph's "
            "output could not ship. Drop --check to generate anyway; a "
            "prototype plate from a non-commercial model is a legitimate "
            "thing to want."
        )
        return 1

    if args.dry_run:
        ctx.say("dry run: nothing queued")
        return 0

    url = base_url(args)
    out_dir = output_dir(ctx)
    if not out_dir:
        ctx.say(
            "note: no output directory configured, so klin cannot index what "
            "comes back. Set %s or a `comfy_output` manifest key." % OUTPUT_ENV
        )

    seed = args.seed if args.seed is not None else 0
    written = []
    for number in range(max(1, args.count)):
        this_seed = seed + number
        ctx.say("queueing seed %d (%d/%d)" % (this_seed, number + 1, max(1, args.count)))
        images = queue(
            url, patch(graph, args, this_seed), out_dir, timeout=args.timeout
        )
        for path in images:
            ctx.say("  %s" % path)
        written.extend(images)

    if written and out_dir and not args.no_index:
        _index_them(ctx, written)

    ctx.say("")
    ctx.say("next: klin ls --since %s" % time.strftime("%Y-%m-%d"))
    return 0


def _index_them(ctx, paths):
    """Put what was just made into the index, so it is never a stray file.

    The index is the reason this is worth doing here rather than leaving the
    scan for later: an output that is indexed the moment it exists cannot
    accumulate into an unattributed pile, which is the state this project found
    itself in with 1,384 images.
    """
    from .. import index

    data = ctx.manifest
    try:
        conn = index.connect(index.db_path(data))
    except Exception as exc:
        ctx.say("note: could not open the index (%s); the files are still there" % exc)
        return

    project = data.get("product_name") or "(unnamed)"
    roots = index.roots(data) if data.get("index") else []
    patterns = index.claims(data) if data.get("index") else []
    seen = set()
    for path in paths:
        for root in roots:
            if os.path.normcase(os.path.normpath(path)).startswith(
                os.path.normcase(os.path.normpath(root)) + os.sep
            ):
                seen.add(root)
    for root in sorted(seen):
        index.scan(conn, root, project, patterns)
    if seen:
        ctx.say("indexed into %s" % index.db_path(data))
