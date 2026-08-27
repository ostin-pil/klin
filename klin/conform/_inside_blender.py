"""The half of conform that runs inside Blender. klin never imports this.

This file's first working line is `import bpy`, which exists only in Blender's
own bundled interpreter. klin runs under a different Python entirely and cannot
import this module at all, which is why the name starts with an underscore:
adapter discovery skips it, and nothing in the test suite can reach it.

Its only caller is `klin/conform/blender.py`, by path, as

    blender --background --factory-startup --python <this> -- <job.json>

Everything it needs arrives in that json file already resolved: absolute paths,
the swatch table as numbers, and the flags. It makes no decision about where
anything lives, because it has no repository and no manifest to read.

It answers on stdout with one `CONFORM ` line of json. Blender prints a banner,
extension chatter and a farewell around it that cannot be turned off, so a
prefix is the only way the result is findable in that noise. Every failure is
caught and answered as `{"error": ...}` rather than allowed to become a
traceback, because a sentence crossing the process boundary is worth more than
a stack that klin would have to parse.

Two orderings here are load-bearing and neither is obvious:

- **Pin before collapsing the slots.** Pinning reads `polygon.material_index`
  to decide which swatch a face belongs to, and collapsing every slot into one
  shared material sets every index to zero. Done the other way round, every
  face silently pins to whichever swatch happened to be first.
- **Triangulate before counting.** The count that matters is the one in the
  file that ships, and a quad exports as two triangles. Counting the mesh
  before triangulating reports a number nothing else in the pipeline agrees
  with.
"""

import json
import os
import re
import sys
import traceback

import bpy

PREFIX = "CONFORM "
NOTE = "conform-note: "

#: Blender appends `.001` when a name collides, and glTF import carries source
#: names in, so a mesh with two `wood_mid` slots arrives with one of them
#: renamed. Stripping the suffix is what stops that reading as an unknown slot.
SUFFIX = re.compile(r"\.\d{3}$")


def note(text):
    print("%s%s" % (NOTE, text), flush=True)


def answer(payload):
    print("%s%s" % (PREFIX, json.dumps(payload, sort_keys=True)), flush=True)


def job_path():
    if "--" not in sys.argv:
        raise RuntimeError("no job file: expected `-- <job.json>` on the command line")
    rest = sys.argv[sys.argv.index("--") + 1:]
    if not rest:
        raise RuntimeError("no job file after `--`")
    return rest[0]


def empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def ensure_gltf():
    """Fail by name when the glTF addon is not there.

    `--factory-startup` loads no user preferences, and while glTF ships enabled
    in stock Blender that is a fact about a version rather than a guarantee.
    It cannot be feature-detected: `hasattr` on anything under `bpy.ops`
    answers True whether the operator exists or not. So this enables the addon
    and lets a genuinely absent importer fail at the call, where the message
    names the operator.
    """
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    except Exception as exc:
        note("could not enable io_scene_gltf2 (%s); assuming it is built in" % exc)


def load(path):
    """Open or import the source, leaving the scene holding only it."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
        return
    empty_scene()
    ensure_gltf()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise RuntimeError("no importer here for %s" % ext)


def meshes(names):
    found = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if names:
        wanted = set(names)
        found = [o for o in found if o.name in wanted]
        missing = wanted - set(o.name for o in found)
        if missing:
            raise RuntimeError("no mesh named %s" % ", ".join(sorted(missing)))
    if not found:
        raise RuntimeError("the source holds no mesh")
    return found


def swatch_of(slot_name, table):
    """Which swatch a material slot names, ignoring Blender's collision suffix."""
    if not slot_name:
        return None
    if slot_name in table:
        return slot_name
    stripped = SUFFIX.sub("", slot_name)
    return stripped if stripped in table else None


def atlas_material(name, image_path, filtering):
    """One material: the atlas, point-filtered, straight into base colour.

    Point filtering is the load-bearing setting. The atlas is a grid of flat
    cells and every uv this writes sits at a cell centre, so bilinear sampling
    would blend whatever is in the neighbouring cell into a prop that is meant
    to be one flat colour.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    shader = tree.nodes.get("Principled BSDF")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(image_path, check_existing=True)
    if filtering == "nearest":
        texture.interpolation = "Closest"
    texture.extension = "CLIP"
    if shader is not None:
        tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        for input_name, value in (("Roughness", 0.9), ("Metallic", 0.0)):
            if input_name in shader.inputs:
                shader.inputs[input_name].default_value = value
        if "Specular IOR Level" in shader.inputs:
            shader.inputs["Specular IOR Level"].default_value = 0.0
        elif "Specular" in shader.inputs:
            shader.inputs["Specular"].default_value = 0.0
    return material


def pin(obj, table, flip_v):
    """Set every loop uv of every face to the texel its slot names.

    Reads `polygon.material_index`, so this must run before the slots are
    collapsed. Faces whose slot names no swatch are left exactly as they
    arrived, and reported, because guessing one for them would produce a prop
    that looks deliberately wrong rather than one that stops the run.
    """
    mesh = obj.data
    # Down to exactly one uv layer before anything is written. A mesh can carry
    # several, only one of which is active, and pinning the active one leaves
    # the others shipping the layout the mesh arrived with: a second TEXCOORD
    # set in the exported file holding coordinates that point anywhere at all.
    # Once every face is one texel a second set cannot mean anything, so the
    # honest move is to drop them rather than to pin each in turn.
    dropped = 0
    while len(mesh.uv_layers) > 1:
        for layer in list(mesh.uv_layers):
            if layer is not mesh.uv_layers.active:
                mesh.uv_layers.remove(layer)
                dropped += 1
                break
    if dropped:
        note("dropped %d extra uv layer(s) from %s" % (dropped, obj.name))
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    layer = mesh.uv_layers.active

    by_index = []
    for slot in obj.material_slots:
        name = slot.material.name if slot.material else None
        by_index.append((name, swatch_of(name, table)))

    counts = {}
    pinned = {}
    for polygon in mesh.polygons:
        index = polygon.material_index
        if index >= len(by_index):
            continue
        name, swatch = by_index[index]
        key = name or "(no material)"
        counts[key] = counts.get(key, 0) + 1
        if swatch is None:
            continue
        u, v = table[swatch]
        # glTF's uv origin is the top left and Blender's is the bottom left, so
        # a coordinate measured in the exported file has to be written flipped
        # to come back out where it was measured.
        stored = (u, 1.0 - v) if flip_v else (u, v)
        for loop in polygon.loop_indices:
            layer.data[loop].uv = stored
        pinned[swatch] = [u, v]
    return by_index, counts, pinned


def collapse(obj, material):
    """Replace every material slot with the one shared atlas material."""
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0


def triangulate(obj):
    """Triangulate in place, so the count reported is the count that ships."""
    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def export(path, keep_external):
    """Write the glTF binary, tolerating a version that renamed a keyword.

    Blender's exporter keywords drift between releases, and a `TypeError` from
    one of them is otherwise a traceback in a subprocess. Dropping the optional
    ones and retrying turns that into a working export plus a warning.
    """
    warnings = []
    options = {
        "filepath": path,
        "export_format": "GLB",
        "export_materials": "EXPORT",
        "export_apply": True,
        "export_yup": True,
    }
    if keep_external:
        # The atlas must stay a reference rather than a copy. Embedding it
        # gives every conformed prop a private image, which is exactly what
        # breaks a project-wide re-grade of the one shared atlas.
        options["export_keep_originals"] = True
    try:
        bpy.ops.export_scene.gltf(**options)
        return warnings
    except TypeError as exc:
        warnings.append("exporter rejected a keyword (%s); retrying with fewer" % exc)
    for key in ("export_keep_originals", "export_yup", "export_apply"):
        options.pop(key, None)
    bpy.ops.export_scene.gltf(**options)
    return warnings


def verify(path):
    """Re-import what was just written and measure the uvs that shipped.

    The point of reading the file back rather than the mesh in memory is that
    the export is where a coordinate convention can change. Blender's uv origin
    and glTF's differ, so a pin that is correct in the scene can be upside down
    in the file, and both look equally clean from inside Blender.
    """
    empty_scene()
    ensure_gltf()
    bpy.ops.import_scene.gltf(filepath=path)

    seen = set()
    image_uri = None
    filtering = None
    layers = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not obj.data.uv_layers:
            continue
        # Every layer, not the active one. Reading only the active layer is how
        # a stale second uv set survives a gate that was meant to catch it.
        layers = max(layers, len(obj.data.uv_layers))
        for layer in obj.data.uv_layers:
            for datum in layer.data:
                seen.add((round(datum.uv[0], 6), round(1.0 - datum.uv[1], 6)))
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.type != "TEX_IMAGE" or not node.image:
                    continue
                image_uri = node.image.filepath or node.image.name
                filtering = 9728 if node.interpolation == "Closest" else 9729
    return sorted(seen), image_uri, filtering, layers


def main():
    payload = json.loads(open(job_path(), encoding="utf-8").read())
    table = dict((name, tuple(uv)) for name, uv in payload["swatches"].items())
    flip_v = bool(payload.get("flip_v", True))

    load(payload["source"])
    objects = meshes(payload.get("objects"))
    note("%d mesh object(s): %s" % (len(objects), ", ".join(o.name for o in objects)))

    material = atlas_material(
        payload.get("material_name") or "klin_atlas",
        payload["atlas"],
        payload.get("filter") or "nearest",
    )

    slots = []
    pinned = {}
    for obj in objects:
        by_index, counts, got = pin(obj, table, flip_v)
        pinned.update(got)
        for name, swatch in by_index:
            slots.append({
                "object": obj.name,
                "name": name,
                "swatch": swatch,
                "polygons": counts.get(name or "(no material)", 0),
            })
        collapse(obj, material)
        triangulate(obj)

    warnings = export(payload["output"], payload.get("keep_image_external", True))

    triangles = sum(len(o.data.polygons) for o in objects)
    vertices = sum(len(o.data.vertices) for o in objects)
    names = [o.name for o in objects]

    measured, image_uri, filtering, layers = ([], None, None, 0)
    if payload.get("verify", True):
        measured, image_uri, filtering, layers = verify(payload["output"])

    answer({
        "ok": True,
        "tool": "blender",
        "blender": bpy.app.version_string.split()[0],
        "objects": names,
        "slots": slots,
        "pinned": pinned,
        "triangles": triangles,
        "vertices": vertices,
        "uvs_measured": [list(uv) for uv in measured],
        "image_uri": image_uri,
        "mag_filter": filtering,
        "uv_layers": layers,
        "flip_v": flip_v,
        "warnings": warnings,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        answer({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc()})
        sys.exit(1)
