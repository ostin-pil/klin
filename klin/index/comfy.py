"""ComfyUI outputs: read the graph the image carries about itself.

ComfyUI writes the whole API-format graph into a `tEXt` chunk keyed `prompt`,
so a generated PNG is self-describing. Every model, every LoRA and its
strength, the seed, the sampler settings and both prompts are in the file. This
is the reason `klin index` is a scan rather than an archaeology project: the
provenance was never lost, only unindexed.

The container is `png`'s business; this module only reads the graph it finds
and makes sense of it.

**Traversal, not pattern matching.** The naive read collects every node whose
class name contains "Lora" and calls that the stack. That is wrong in two ways
a graph will eventually exhibit: it loses the order the LoRAs were applied in,
and it counts loaders that sit in the graph wired to nothing. This module
starts at the sampler and walks its `model` input backwards, so the stack it
reports is the chain that actually produced the image, in order, with the
strengths that were in force.

**A guess is reported, never made silently.** Where the graph cannot be
resolved to one chain the read returns what it found and names the ambiguity in
`notes`. An index row saying "two samplers, could not tell which" is worth more
than one that quietly picked the first.
"""

import hashlib
import json

from . import png

NAME = "comfy"
HELP = "ComfyUI PNG outputs, read from the embedded prompt graph"

#: The chunk ComfyUI writes its API-format graph into.
CHUNK = "prompt"

#: Nodes that terminate a model chain: the input holding the weight name, and
#: the role klin records it under.
MODEL_LOADERS = {
    "UNETLoader": ("unet_name", "unet"),
    "UnetLoaderGGUF": ("unet_name", "unet"),
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoint"),
    "CheckpointLoader": ("ckpt_name", "checkpoint"),
    "DiffusersLoader": ("model_path", "diffusers"),
}

#: Nodes that add a LoRA to a chain: the inputs naming it and its strength.
LORA_LOADERS = {
    "LoraLoader": ("lora_name", "strength_model"),
    "LoraLoaderModelOnly": ("lora_name", "strength_model"),
    "LoraLoaderGGUF": ("lora_name", "strength_model"),
}

#: Nodes carrying the sampling settings.
SAMPLERS = (
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
)

#: Nodes holding the output dimensions.
LATENTS = (
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyQwenImageLayeredLatentImage",
)

#: Nodes whose text input is a prompt a person typed.
TEXT_ENCODERS = (
    "CLIPTextEncode",
    "TextEncodeQwenImageEdit",
    "TextEncodeQwenImageEditPlus",
)

#: How far the conditioning walk will follow transform nodes before giving up.
#: Guidance, zero-out and controlnet application all sit between a sampler and
#: the encoder, so the encoder is rarely one hop away, but a cycle must not
#: spin forever.
MAX_HOPS = 16


class ReadError(Exception):
    pass


def graph_of(path):
    """The API-format graph a ComfyUI output carries, or None."""
    raw = png.text_chunks(path).get(CHUNK)
    if raw is None:
        return None
    try:
        graph = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReadError("%s: prompt chunk does not parse: %s" % (path, exc))
    if not isinstance(graph, dict):
        raise ReadError("%s: prompt chunk is not an object" % path)
    return graph


def link(value):
    """A wired input is `[node_id, slot]`; a literal is anything else."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return None


def node_of(graph, node_id):
    node = graph.get(str(node_id))
    if not isinstance(node, dict):
        return {}, ""
    inputs = node.get("inputs")
    return (inputs if isinstance(inputs, dict) else {}), node.get("class_type") or ""


def _walk_model(graph, node_id, notes):
    """Follow a `model` input backwards, collecting the LoRA stack in order.

    Returns `(base, loras)`. The list is ordered as the graph applies them,
    because a stack's order changes the result and a set would throw that away.
    """
    seen = set()
    loras = []
    base = None
    while node_id is not None and node_id not in seen:
        seen.add(node_id)
        inputs, kind = node_of(graph, node_id)
        if kind in MODEL_LOADERS:
            key, role = MODEL_LOADERS[kind]
            base = {"name": inputs.get(key), "role": role, "loader": kind}
            break
        if kind in LORA_LOADERS:
            name_key, strength_key = LORA_LOADERS[kind]
            loras.append(
                {
                    "name": inputs.get(name_key),
                    "strength": inputs.get(strength_key),
                    "loader": kind,
                }
            )
        node_id = link(inputs.get("model"))
    loras.reverse()
    if base is None:
        notes.append("model chain ended without a recognised loader")
    return base, loras


def _text_from(graph, node_id):
    """Resolve a conditioning input back to the prompt a person typed."""
    hops = 0
    while node_id is not None and hops < MAX_HOPS:
        inputs, kind = node_of(graph, node_id)
        if kind in TEXT_ENCODERS:
            text = inputs.get("text")
            if text is None:
                text = inputs.get("prompt")
            return text if isinstance(text, str) else None
        nxt = None
        for key in ("conditioning", "conditioning_to", "positive", "negative"):
            nxt = link(inputs.get(key))
            if nxt:
                break
        node_id = nxt
        hops += 1
    return None


def _latent_size(graph, node_id):
    """Walk a latent input back to whichever node declared the dimensions.

    None is the right answer for an img2img graph, where the latent comes from
    an encoded image rather than a declared size. That is not a gap: the index
    takes the output's real dimensions from the PNG header, which is exact for
    every file and needs no graph at all. This field is only what the *graph*
    asked for, which can differ from what was written when an upscale sits
    between the sampler and the save.
    """
    seen = set()
    while node_id is not None and node_id not in seen:
        seen.add(node_id)
        inputs, kind = node_of(graph, node_id)
        if kind in LATENTS:
            return [inputs.get("width"), inputs.get("height")]
        node_id = link(inputs.get("samples")) or link(inputs.get("latent_image"))
    return None


def workflow_sha256(graph):
    """A stable hash of the graph with the seed removed.

    The point of this field is to group runs of one setup, and a seed sweep is
    one setup. Hashing the graph verbatim would give every image in a sweep its
    own workflow, which is the opposite of useful.
    """
    stripped = {}
    for node_id, node in graph.items():
        node = node if isinstance(node, dict) else {}
        inputs = dict(node.get("inputs") or {})
        inputs.pop("seed", None)
        inputs.pop("noise_seed", None)
        stripped[str(node_id)] = {
            "class_type": node.get("class_type"),
            "inputs": inputs,
        }
    canonical = json.dumps(stripped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read(path):
    """Provenance for one ComfyUI output, or None when the file carries none.

    A file with no `prompt` chunk is not an error. Contact sheets, masks and
    anything Pillow wrote are legitimate images in the same directory, and the
    index records those with what it knows: a path, a size and a hash.
    """
    graph = graph_of(path)
    if graph is None:
        return None

    notes = []
    found = [
        node_id
        for node_id, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") in SAMPLERS
    ]
    if not found:
        notes.append("no sampler node, so there is no model chain to trace")
        return {
            "source": NAME,
            "model": None,
            "loras": [],
            "graph": graph,
            "workflow_sha256": workflow_sha256(graph),
            "notes": notes,
        }
    if len(found) > 1:
        notes.append(
            "%d sampler nodes in this graph; reporting the first rather than "
            "guessing which one produced this file" % len(found)
        )

    sampler_id = sorted(found, key=lambda n: (len(str(n)), str(n)))[0]
    inputs, _kind = node_of(graph, sampler_id)
    base, loras = _walk_model(graph, link(inputs.get("model")), notes)
    seed = inputs.get("seed")
    if seed is None:
        seed = inputs.get("noise_seed")

    return {
        "source": NAME,
        "model": base,
        "loras": loras,
        "seed": seed,
        "steps": inputs.get("steps"),
        "cfg": inputs.get("cfg"),
        "sampler": inputs.get("sampler_name"),
        "scheduler": inputs.get("scheduler"),
        "denoise": inputs.get("denoise"),
        "prompt": _text_from(graph, link(inputs.get("positive"))),
        "negative": _text_from(graph, link(inputs.get("negative"))),
        "latent_size": _latent_size(graph, link(inputs.get("latent_image"))),
        "workflow_sha256": workflow_sha256(graph),
        "graph": graph,
        "notes": notes,
    }
