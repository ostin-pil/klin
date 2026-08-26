"""Generation: the pre-flight check, and patching a graph by role.

No test here reaches a network or a GPU. The server is a stub and the graphs
are built in-process, which matters because the claim under test is that klin
locates roles by walking the graph rather than by recognising node titles, and
a fixture copied out of one project's workflow directory would only prove those
particular titles happen to be present.
"""

import io
import json
import os

import pytest

from klin import cli, gen
from klin.gen import comfy

from conftest import record


def graph(base="flux1-dev-fp8.safetensors", loras=(), latent=True, titles=False):
    """A template with no helpful titles, wired the way ComfyUI wires one."""
    nodes = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": base}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad hands"}},
        "12": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["10", 0]}},
    }
    if titles:
        nodes["10"]["_meta"] = {"title": "POSITIVE"}
    if latent:
        nodes["20"] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 512, "height": 512},
        }
    upstream = "1"
    for number, (name, strength) in enumerate(loras, start=30):
        nodes[str(number)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": name,
                "strength_model": strength,
                "model": [upstream, 0],
            },
        }
        upstream = str(number)
    nodes["50"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 3.5,
            "model": [upstream, 0],
            "positive": ["12", 0],
            "negative": ["11", 0],
            "latent_image": ["20", 0] if latent else ["21", 0],
        },
    }
    nodes["60"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "old", "images": ["50", 0]},
    }
    if not latent:
        nodes["21"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["22", 0]}}
        nodes["22"] = {"class_type": "LoadImage", "inputs": {"image": "in.png"}}
    return nodes


class Args(object):
    def __init__(self, **kw):
        self.prompt = None
        self.prompt_file = None
        self.negative = None
        self.seed = None
        self.steps = None
        self.cfg = None
        self.size = None
        self.lora = None
        self.prefix = None
        self.count = 1
        self.url = None
        self.timeout = 5
        for key, value in kw.items():
            setattr(self, key, value)


def write(tmp_path, name, payload):
    path = str(tmp_path / name)
    io.open(path, "w", encoding="utf-8").write(json.dumps(payload))
    return path


# ------------------------------------------------------- loading a workflow


def test_the_editor_format_is_named_rather_than_failing_obscurely(tmp_path):
    """It looks close enough to try, and fails in a way that reads like a bug
    in klin rather than a wrong export."""
    path = write(tmp_path, "wf.json", {"nodes": [], "links": []})
    with pytest.raises(gen.GenError) as exc:
        gen.load_graph(path)
    assert "Export (API)" in str(exc.value)


def test_a_missing_or_broken_workflow_says_which(tmp_path):
    with pytest.raises(gen.GenError):
        gen.load_graph(str(tmp_path / "nope.json"))
    path = str(tmp_path / "bad.json")
    io.open(path, "w").write("{not json")
    with pytest.raises(gen.GenError):
        gen.load_graph(path)


# ------------------------------------------------------------- the pre-flight


def test_every_weight_the_graph_would_load_is_listed():
    found = gen.weights_in(graph(loras=(("a.safetensors", 0.8),)))
    assert ("flux1-dev-fp8.safetensors", "checkpoint") in found
    assert ("a.safetensors", "lora") in found


def test_a_slot_at_strength_zero_is_not_a_licence_risk():
    """It contributes nothing, and reporting it would train the reader to
    ignore this output."""
    found = gen.weights_in(graph(loras=(("off.safetensors", 0.0),)))
    assert [name for name, _role in found] == ["flux1-dev-fp8.safetensors"]


# --------------------------------------------------------- patching by role


def test_the_prompt_lands_without_any_node_being_titled(tmp_path):
    """The ported driver found the positive prompt by looking for a node
    titled POSITIVE, which fails on a valid template nobody titled."""
    patched = comfy.patch(graph(titles=False), Args(prompt="a warm bar car"), 101)
    assert patched["10"]["inputs"]["text"] == "a warm bar car"
    assert patched["11"]["inputs"]["text"] == "bad hands"  # negative untouched


def test_the_walk_passes_through_a_conditioning_transform():
    """FluxGuidance sits between the sampler and the encoder, so the encoder
    is not one hop away."""
    patched = comfy.patch(graph(), Args(prompt="x"), 1)
    assert patched["10"]["inputs"]["text"] == "x"


def test_seed_steps_cfg_and_size_go_where_they_belong():
    patched = comfy.patch(
        graph(), Args(prompt="x", steps=4, cfg=1.0, size="768x448"), 202
    )
    assert patched["50"]["inputs"]["seed"] == 202
    assert patched["50"]["inputs"]["steps"] == 4
    assert patched["50"]["inputs"]["cfg"] == 1.0
    assert patched["20"]["inputs"]["width"] == 768
    assert patched["20"]["inputs"]["height"] == 448


def test_size_on_an_img2img_graph_explains_itself():
    with pytest.raises(gen.GenError) as exc:
        comfy.patch(graph(latent=False), Args(prompt="x", size="768x448"), 1)
    assert "comes from the input image" in str(exc.value)


def test_lora_slots_fill_in_chain_order():
    patched = comfy.patch(
        graph(loras=(("one.safetensors", 0.1), ("two.safetensors", 0.2))),
        Args(prompt="x", lora=["psx.safetensors@0.6", "iso.safetensors@0.3"]),
        1,
    )
    assert patched["30"]["inputs"]["lora_name"] == "psx.safetensors"
    assert patched["30"]["inputs"]["strength_model"] == 0.6
    assert patched["31"]["inputs"]["lora_name"] == "iso.safetensors"
    assert patched["31"]["inputs"]["strength_model"] == 0.3


def test_an_unused_slot_is_zeroed_and_never_unwired():
    """Rewiring by hand is where a sweep quietly renders something other than
    what its record says."""
    patched = comfy.patch(
        graph(loras=(("one.safetensors", 0.1), ("two.safetensors", 0.2))),
        Args(prompt="x", lora=["only.safetensors@0.5"]),
        1,
    )
    assert patched["31"]["inputs"]["strength_model"] == 0.0
    assert patched["31"]["inputs"]["model"] == ["30", 0]  # still in the chain


def test_asking_for_more_loras_than_the_template_wires_is_refused():
    with pytest.raises(gen.GenError) as exc:
        comfy.patch(graph(), Args(prompt="x", lora=["a@1.0"]), 1)
    assert "rather than having klin rewire" in str(exc.value)


def test_two_samplers_are_refused_rather_than_guessed():
    nodes = graph()
    nodes["51"] = {"class_type": "KSampler", "inputs": {"seed": 0, "model": ["1", 0]}}
    with pytest.raises(gen.GenError) as exc:
        comfy.patch(nodes, Args(prompt="x"), 1)
    assert "will not guess" in str(exc.value)


def test_a_graph_with_no_sampler_says_what_it_looked_for():
    with pytest.raises(gen.GenError) as exc:
        comfy.patch({"1": {"class_type": "SaveImage", "inputs": {}}}, Args(), 1)
    assert "KSampler" in str(exc.value)


def test_the_prefix_reaches_every_save_node():
    patched = comfy.patch(graph(), Args(prompt="x", prefix="probe"), 1)
    assert patched["60"]["inputs"]["filename_prefix"] == "probe"


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("a.safetensors@0.6", ("a.safetensors", 0.6)),
        ("a.safetensors", ("a.safetensors", 1.0)),
        ("sub/dir/a.safetensors@0", ("sub/dir/a.safetensors", 0.0)),
    ],
)
def test_lora_specs_parse(spec, expected):
    assert comfy.parse_lora(spec) == expected


def test_a_lora_spec_with_a_bad_strength_is_refused():
    with pytest.raises(gen.GenError):
        comfy.parse_lora("a.safetensors@loud")


@pytest.mark.parametrize("value", ["768x448", "768X448", "768*448", "768,448"])
def test_sizes_parse(value):
    assert comfy.parse_size(value) == (768, 448)


def test_a_bad_size_is_refused_rather_than_ignored():
    with pytest.raises(gen.GenError):
        comfy.parse_size("big")


# ----------------------------------------------------------- queue and poll


class Server(object):
    """A ComfyUI that answers without a socket."""

    def __init__(self, outputs=None, status=None, never=False):
        self.posted = []
        self.outputs = outputs
        self.status = status
        self.never = never
        self.polls = 0

    def post(self, url, path, payload):
        self.posted.append((path, payload))
        return {"prompt_id": "abc"}

    def get(self, url, path):
        self.polls += 1
        if self.never:
            return {}
        entry = {"outputs": self.outputs or {}}
        if self.status:
            entry["status"] = self.status
        return {"abc": entry}


@pytest.fixture
def server(monkeypatch):
    def install(stub):
        monkeypatch.setattr(comfy, "post", stub.post)
        monkeypatch.setattr(comfy, "get", stub.get)
        return stub

    return install


def test_a_finished_prompt_returns_the_paths_it_wrote(server):
    stub = server(
        Server(outputs={"60": {"images": [{"filename": "a.png", "subfolder": "mock"}]}})
    )
    got = comfy.queue("http://x", graph(), r"D:\out", sleep=lambda s: None)
    assert got == [os.path.join(r"D:\out", "mock", "a.png")]
    assert stub.posted[0][0] == "/prompt"


def test_an_error_from_the_server_is_reported_not_swallowed(server):
    server(Server(status={"status_str": "error", "messages": ["oom"]}))
    with pytest.raises(gen.GenError) as exc:
        comfy.queue("http://x", graph(), r"D:\out", sleep=lambda s: None)
    assert "ComfyUI reported an error" in str(exc.value)


def test_a_prompt_that_never_finishes_times_out(server):
    server(Server(never=True))
    with pytest.raises(gen.GenError) as exc:
        comfy.queue("http://x", graph(), r"D:\out", timeout=3, sleep=lambda s: None)
    assert "timed out" in str(exc.value)


# ------------------------------------------------------------------- the cli


def run(made, argv):
    stream = io.StringIO()
    code = cli.main(["--repo", made.root] + argv, stream=stream)
    return code, stream.getvalue()


@pytest.fixture(autouse=True)
def no_real_machine(tmp_path, monkeypatch):
    monkeypatch.delenv("KLIN_MODELS", raising=False)
    monkeypatch.delenv("COMFY_OUTPUT", raising=False)
    monkeypatch.setenv("KLIN_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("KLIN_INDEX", str(tmp_path / "index.sqlite3"))


def test_an_untraceable_model_is_not_a_pass(repo, tmp_path):
    made = repo()
    path = write(tmp_path, "wf.json", graph())
    code, out = run(made, ["gen", "comfy", "--workflow", path, "--dry-run"])
    assert code == 0
    assert "no ledger record" in out
    assert "output could ship: NO" in out


def test_check_refuses_to_queue_and_says_the_flag_is_optional(repo, tmp_path):
    made = repo()
    path = write(tmp_path, "wf.json", graph())
    code, out = run(made, ["gen", "comfy", "--workflow", path, "--check"])
    assert code == 1
    assert "refusing to queue" in out
    assert "legitimate thing to want" in out


def test_a_clean_graph_reports_that_it_could_ship(repo, tmp_path, monkeypatch):
    made = repo()
    weights = tmp_path / "models" / "checkpoints"
    os.makedirs(str(weights))
    blob = str(weights / "flux1-dev-fp8.safetensors")
    io.open(blob, "wb").write(b"\x00" * 16)
    monkeypatch.setenv("KLIN_MODELS", str(tmp_path / "models"))
    made.write_records([record("clean-model", "CC0-1.0", paths=[blob])])

    path = write(tmp_path, "wf.json", graph())
    code, out = run(made, ["gen", "comfy", "--workflow", path, "--dry-run"])
    assert code == 0
    assert "output could ship: yes" in out
    assert "nothing queued" in out


# ------------------------ two defects the synthetic graphs did not reach, and
# ------------------------ the real workflow templates did


def zeroout_graph():
    """The Flux and Z-Image pattern: the negative is the positive, zeroed.

    There is one encoder in this graph and no negative prompt to write.
    """
    nodes = graph()
    nodes["7"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["10", 0]}}
    nodes["50"]["inputs"]["negative"] = ["7", 0]
    del nodes["11"]
    return nodes


def controlnet_graph():
    """A node taking both conditionings, which a direction-blind walk
    resolves to the same encoder for both branches."""
    nodes = graph()
    nodes["15"] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {"positive": ["12", 0], "negative": ["11", 0]},
    }
    nodes["50"]["inputs"]["positive"] = ["15", 0]
    nodes["50"]["inputs"]["negative"] = ["15", 0]
    return nodes


def test_a_shared_encoder_is_refused_rather_than_overwritten():
    """Before this check, --negative overwrote the prompt and the run produced
    a plate whose record said one thing and whose pixels came from another."""
    with pytest.raises(gen.GenError) as exc:
        comfy.patch(zeroout_graph(), Args(prompt="warm", negative="blurry"), 1)
    assert "no separate negative prompt" in str(exc.value)


def test_the_same_graph_is_fine_without_a_negative():
    patched = comfy.patch(zeroout_graph(), Args(prompt="warm"), 1)
    assert patched["10"]["inputs"]["text"] == "warm"


def test_the_walk_follows_the_branch_it_came_down():
    """`ControlNetApplyAdvanced` has a positive and a negative input. A walk
    that always tried `positive` first put both prompts in one node."""
    patched = comfy.patch(
        controlnet_graph(), Args(prompt="warm", negative="blurry"), 1
    )
    assert patched["10"]["inputs"]["text"] == "warm"
    assert patched["11"]["inputs"]["text"] == "blurry"


def test_a_cycle_in_the_conditioning_does_not_spin():
    nodes = graph()
    nodes["12"]["inputs"]["conditioning"] = ["12", 0]
    assert comfy._encoder_behind(nodes, "12", "positive") is None
