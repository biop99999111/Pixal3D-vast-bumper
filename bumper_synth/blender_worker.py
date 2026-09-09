"""Run with Blender --background --python this_file -- --job job.json.

RGB and each visible-instance mask share the exact same deformed geometry.
One damage per image is intentional in v1: no ambiguous overlap labels.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector, Matrix


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def node(tree, kind, **attrs):
    result = tree.nodes.new(kind)
    for key, value in attrs.items():
        setattr(result, key, value)
    return result


def socket(tree, target, value):
    if isinstance(value, (int, float, tuple, list)):
        target.default_value = value
    else:
        tree.links.new(value, target)


def math_node(tree, operation, a, b=0):
    result = node(tree, "ShaderNodeMath", operation=operation)
    socket(tree, result.inputs[0], a)
    socket(tree, result.inputs[1], b)
    return result.outputs[0]


def vec_node(tree, operation, a, b=(0, 0, 0)):
    result = node(tree, "ShaderNodeVectorMath", operation=operation)
    socket(tree, result.inputs[0], a)
    socket(tree, result.inputs[1], b)
    return result.outputs["Value" if operation in ("DISTANCE", "DOT_PRODUCT", "LENGTH") else "Vector"]


def mask_nodes(tree, damage):
    attr = node(tree, "ShaderNodeAttribute", attribute_name="synth_rest")
    pos = attr.outputs["Vector"]
    delta = vec_node(tree, "SUBTRACT", pos, damage["center"])
    if damage["class_id"] == 2:
        # Distance to a finite 3-D line segment (not an infinite stripe).
        tangent = damage["tangent"]
        projection = vec_node(tree, "DOT_PRODUCT", delta, tangent)
        projection = math_node(tree, "MAXIMUM", projection, -damage["length"]/2)
        projection = math_node(tree, "MINIMUM", projection, damage["length"]/2)
        scale = node(tree, "ShaderNodeVectorMath", operation="SCALE")
        scale.inputs[0].default_value = tangent
        tree.links.new(projection, scale.inputs["Scale"])
        distance = vec_node(tree, "LENGTH", vec_node(tree, "SUBTRACT", delta, scale.outputs[0]))
        radius = damage["width"] / 2
    else:
        distance = vec_node(tree, "LENGTH", delta)
        radius = damage["radius"]
    return math_node(tree, "LESS_THAN", distance, radius)


def damage_material(original, damage, mask_only=False):
    material = original.copy() if original and not mask_only else bpy.data.materials.new("damage_pass")
    material.use_nodes = True
    tree = material.node_tree
    mask = mask_nodes(tree, damage)
    outputs = [n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
    output = outputs[0] if outputs else node(tree, "ShaderNodeOutputMaterial")
    if mask_only:
        tree.nodes.remove(next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"))
        emission = node(tree, "ShaderNodeEmission")
        tree.links.new(mask, emission.inputs["Color"])
        tree.links.new(emission.outputs[0], output.inputs["Surface"])
    elif damage["class_id"] == 2:
        shaders = [n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"]
        if not shaders:
            raise ValueError("Scratch rendering requires a Principled BSDF material")
        for shader in shaders:
            base = shader.inputs["Base Color"]
            old = base.links[0].from_socket if base.is_linked else tuple(base.default_value)
            mix = node(tree, "ShaderNodeMixRGB", blend_type="MIX")
            tree.links.new(mask, mix.inputs[0])
            socket(tree, mix.inputs[1], old)
            mix.inputs[2].default_value = (0.14, 0.13, 0.12, 1)
            tree.links.new(mix.outputs[0], base)
            roughness = shader.inputs["Roughness"]
            old_rough = roughness.links[0].from_socket if roughness.is_linked else roughness.default_value
            combined = math_node(tree, "ADD", math_node(tree, "MULTIPLY", old_rough,
                                       math_node(tree, "SUBTRACT", 1, mask)),
                                 math_node(tree, "MULTIPLY", mask, .8))
            tree.links.new(combined, roughness)
            bump = node(tree, "ShaderNodeBump")
            bump.inputs["Strength"].default_value = .35
            bump.inputs["Distance"].default_value = .002
            bump.invert = True
            if shader.inputs["Normal"].is_linked:
                tree.links.new(shader.inputs["Normal"].links[0].from_socket, bump.inputs["Normal"])
            tree.links.new(mask, bump.inputs["Height"])
            tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return material


def plain_material(color=(.25, .3, .38, 1), emission=False):
    material = bpy.data.materials.new("plain")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = .24
    if emission:
        tree = material.node_tree
        light = node(tree, "ShaderNodeEmission")
        light.inputs["Color"].default_value = color
        tree.links.new(light.outputs[0], tree.nodes.get("Material Output").inputs["Surface"])
    return material


def fixture(path):
    # Explicitly synthetic curved panel; verifies software, not reconstruction quality.
    vertices, faces = [], []
    nx, nz = 101, 41
    for z in range(nz):
        for x in range(nx):
            xx = (x/(nx-1)-.5)*2
            zz = (z/(nz-1)-.5)*.6
            vertices.append((xx, .22*xx*xx + .05*(zz/.3)**2, zz))
    for z in range(nz-1):
        for x in range(nx-1):
            i = z*nx+x
            faces.append((i, i+1, i+1+nx, i+nx))
    mesh = bpy.data.meshes.new("fixture_panel")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = ((co.x+1)/2, (co.z+.3)/.6)
    obj = bpy.data.objects.new("FIXTURE_NOT_A_REAL_BUMPER", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(plain_material())
    for poly in mesh.polygons:
        poly.use_smooth = True
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")


def import_asset(path, cfg):
    bpy.ops.import_scene.gltf(filepath=str(path))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise ValueError("GLB contains no mesh")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.convert(target="MESH")
    bpy.ops.object.join()
    obj = bpy.context.object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    camera_path = Path(path).parent / "camera.json"
    alignment = Matrix.Identity(3)
    if cfg.get("auto_align_input_camera", True) and camera_path.exists():
        camera_info = json.loads(camera_path.read_text(encoding="utf-8"))
        grid = Matrix(((1,0,0),(0,0,-1),(0,1,0)))
        exported = Matrix(camera_info["export_transform"]).to_3x3()
        imported_camera_rotation = grid @ exported
        alignment = grid @ imported_camera_rotation.inverted()
        for vertex in obj.data.vertices:
            vertex.co = alignment @ vertex.co
    obj.rotation_euler = [math.radians(v) for v in cfg["rotation_degrees"]]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    points = [v.co.copy() for v in obj.data.vertices]
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    center, scale = (low+high)/2, max(high-low)
    if scale <= 0:
        raise ValueError("Degenerate asset bounds")
    for v in obj.data.vertices:
        v.co = (v.co-center) / scale * 2
    obj.data.update()
    if not obj.data.materials:
        obj.data.materials.append(plain_material())
    for i, material in enumerate(obj.data.materials):
        if material is None:
            obj.data.materials[i] = plain_material()
    attr = obj.data.attributes.new("synth_rest", "FLOAT_VECTOR", "POINT")
    for v in obj.data.vertices:
        attr.data[v.index].vector = v.co
    return obj, {"center": list(center), "scale_factor": 2/scale,
                 "rotation_degrees": cfg["rotation_degrees"],
                 "input_camera_alignment": [list(row) for row in alignment]}


def look_at(obj, target):
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_scene(cfg, rng):
    scene = bpy.context.scene
    scene.render.engine = cfg["engine"]
    scene.render.resolution_x = scene.render.resolution_y = cfg["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    if cfg["engine"] == "CYCLES":
        scene.cycles.samples = cfg["samples"]
        scene.cycles.use_denoising = True
        if cfg["device"] != "CPU":
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = cfg["device"]
            preferences.get_devices()
            found = False
            for dev in preferences.devices:
                dev.use = dev.type == cfg["device"]
                found |= bool(dev.use)
            if not found:
                raise RuntimeError(f"No Blender {cfg['device']} device; no silent CPU fallback")
            scene.cycles.device = "GPU"
    world = bpy.data.worlds.new("background")
    world.use_nodes = True
    scene.world = world
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.08, .08, .08, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .35
    bpy.ops.object.camera_add(location=(0, -4, 0))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.6
    scene.camera = camera
    for location, energy, size in (((-1.5, -3, 2), 150, 2), ((2, -1, .4), 80, 1)):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        look_at(light, (0, 0, 0))
    return scene, camera


def choose_damage(obj, cfg, rng, class_id):
    if cfg["front_axis"] not in ("-Y", "+Y"):
        raise ValueError("front_axis must be -Y or +Y; use rotation_degrees for other orientations")
    direction = Vector((0, -1 if cfg["front_axis"] == "-Y" else 1, 0))
    # Ray hits the first surface, preventing damage placement behind the front surface.
    for _ in range(100):
        origin = Vector((rng.uniform(-.85, .85), direction.y*4, rng.uniform(-.8, .8)))
        hit, center, normal, face = obj.ray_cast(origin, -direction)
        if hit and abs(normal.dot(direction)) > .45:
            if normal.dot(direction) < 0:
                normal = -normal
            break
    else:
        raise ValueError("Cannot find front surface; check GLB orientation and front_axis")
    damage = {"class_id": class_id, "instance_id": 1, "center": list(center),
              "normal": list(normal), "units": "normalized asset longest dimension = 2"}
    if class_id == 2:
        angle = rng.uniform(-math.pi, math.pi)
        tangent = Vector((math.cos(angle), 0, math.sin(angle)))
        tangent = (tangent-normal*tangent.dot(normal)).normalized()
        damage.update(tangent=list(tangent), length=rng.uniform(*cfg["damage"]["scratch"]["length"]),
                      width=rng.uniform(*cfg["damage"]["scratch"]["width"]))
    else:
        settings = cfg["damage"]["dent" if class_id == 0 else "ding"]
        damage.update(radius=rng.uniform(*settings["radius"]), depth=rng.uniform(*settings["depth"]))
        affected = []
        for vertex in obj.data.vertices:
            ratio = (vertex.co-center).length/damage["radius"]
            if ratio < 1:
                affected.append(vertex.index)
                vertex.co -= normal * damage["depth"] * (1-ratio*ratio)**2
        if len(affected) < 8:
            raise ValueError("Mesh too coarse for local damage; use higher-density mesh or larger damage")
        obj.data.update()
        damage["affected_vertices"] = len(affected)
    return damage


def render_job(job):
    cfg, output = job["config"], Path(job["output"])
    rng = random.Random(job["seed"])
    obj, normalization = import_asset(job["asset"], cfg)
    scene, camera = setup_scene(cfg, rng)
    azimuth = job.get("azimuth", rng.choice(cfg["views_degrees"]))
    elevation = rng.uniform(*cfg["elevation_degrees"])
    a, e = math.radians(azimuth), math.radians(elevation)
    sign = -1 if cfg["front_axis"] == "-Y" else 1
    camera.location = (4*math.sin(a)*math.cos(e), sign*4*math.cos(a)*math.cos(e), 4*math.sin(e))
    look_at(camera, (0, 0, 0))
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        rng.uniform(.015, .12), rng.uniform(.015, .12), rng.uniform(.015, .12), 1)
    lights = [o for o in scene.objects if o.type == "LIGHT"]
    for light in lights:
        light.location.x += rng.uniform(-.5, .5)
        light.data.energy *= rng.uniform(.65, 1.4)
        if sign == 1:
            light.location.y *= -1
        look_at(light, (0, 0, 0))
    damage = None
    originals = list(obj.data.materials)
    if job["class_id"] >= 0:
        scene.render.filepath = str(output / "clean.png")
        bpy.ops.render.render(write_still=True)
        damage = choose_damage(obj, cfg, rng, job["class_id"])
        for i, original in enumerate(originals):
            obj.data.materials[i] = damage_material(original, damage)
    scene.render.filepath = str(output / "rgb.png")
    bpy.ops.render.render(write_still=True)
    if damage:
        for i, original in enumerate(originals):
            obj.data.materials[i] = damage_material(original, damage, mask_only=True)
        # Other mesh objects do not exist after join, but all non-damaged surfaces
        # of the same mesh remain black opaque occluders, including holes/rear faces.
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
        for light in lights:
            light.hide_render = True
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
        if cfg["engine"] == "CYCLES":
            scene.cycles.samples = 1
            scene.cycles.use_denoising = False
        scene.render.filepath = str(output / "mask.png")
        bpy.ops.render.render(write_still=True)
    save_json(output / "metadata.json", {
        "seed": job["seed"], "damage": damage, "normalization": normalization,
        "camera_matrix_world": [list(row) for row in camera.matrix_world],
        "camera_type": "ORTHO", "ortho_scale": camera.data.ortho_scale,
        "azimuth": azimuth, "elevation": elevation, "blender": bpy.app.version_string,
        "lights": [{"location": list(o.location), "energy": o.data.energy} for o in lights],
        "label_policy": "Visible projected modification support; shadows excluded. Human appearance review required."
    })


def reference_job(job):
    bpy.ops.import_scene.gltf(filepath=job["asset"])
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = job["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.world = bpy.data.worlds.new("black")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
    bpy.ops.object.camera_add()
    scene.camera = bpy.context.object
    scene.camera.matrix_world = Matrix(job["camera_matrix"])
    scene.camera.data.type = "PERSP"
    scene.camera.data.sensor_fit = "HORIZONTAL"
    scene.camera.data.angle = job["fov"]
    materials = []
    for obj in list(scene.objects):
        if obj.type != "MESH":
            continue
        if not obj.data.materials:
            obj.data.materials.append(plain_material())
        for index, original in enumerate(list(obj.data.materials)):
            mat = (original or plain_material()).copy()
            mat.use_nodes = True
            tree = mat.node_tree
            shader = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            output = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
            emission = node(tree, "ShaderNodeEmission")
            if shader:
                base = shader.inputs["Base Color"]
                socket(tree, emission.inputs["Color"], base.links[0].from_socket if base.is_linked else tuple(base.default_value))
            tree.links.new(emission.outputs[0], output.inputs["Surface"])
            obj.data.materials[index] = mat
            materials.append((tree, emission))
    scene.render.filepath = str(Path(job["output"]) / "base_color.png")
    bpy.ops.render.render(write_still=True)
    for tree, emission in materials:
        for link in list(emission.inputs["Color"].links):
            tree.links.remove(link)
        emission.inputs["Color"].default_value = (1,1,1,1)
    scene.render.filepath = str(Path(job["output"]) / "silhouette.png")
    bpy.ops.render.render(write_still=True)
    for tree, emission in materials:
        geometry = node(tree, "ShaderNodeNewGeometry")
        normal = vec_node(tree, "ADD", geometry.outputs["Normal"], (1,1,1))
        scale = node(tree, "ShaderNodeVectorMath", operation="SCALE")
        tree.links.new(normal, scale.inputs[0])
        scale.inputs["Scale"].default_value = .5
        tree.links.new(scale.outputs[0], emission.inputs["Color"])
    scene.render.filepath = str(Path(job["output"]) / "normal.png")
    bpy.ops.render.render(write_still=True)


def retexture_job(job):
    from bpy_extras.object_utils import world_to_camera_view
    bpy.ops.import_scene.gltf(filepath=job["asset"])
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.bake.margin = 8
    scene.render.bake.use_clear = True
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.matrix_world = Matrix(job["camera_matrix"])
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = job["fov"]
    scene.camera = camera
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    projection = bpy.data.images.load(job["image"])
    objects = [o for o in scene.objects if o.type == "MESH"]
    dependency_graph = bpy.context.evaluated_depsgraph_get()
    for index, obj in enumerate(objects):
        if not obj.data.uv_layers:
            raise ValueError("Retexture requires existing UVs")
        original_uv = obj.data.uv_layers.active.name
        projected_uv = obj.data.uv_layers.new(name="synth_projection")
        visible_attr = obj.data.attributes.new("synth_visible", "FLOAT", "CORNER")
        for poly in obj.data.polygons:
            center = obj.matrix_world @ poly.center
            direction = center-camera.location
            distance = direction.length
            hit, location, _, _, hit_object, _ = scene.ray_cast(dependency_graph, camera.location,
                                                               direction.normalized())
            normal = (obj.matrix_world.to_3x3().inverted().transposed() @ poly.normal).normalized()
            visible = (hit and hit_object == obj and (location-center).length < max(1e-4,distance*1e-4)
                       and abs(normal.dot(-direction.normalized())) > .25)
            for loop_index in poly.loop_indices:
                vertex = obj.data.vertices[obj.data.loops[loop_index].vertex_index]
                uv = world_to_camera_view(scene, camera, obj.matrix_world @ vertex.co)
                projected_uv.data[loop_index].uv = (uv.x,uv.y)
                visible_attr.data[loop_index].value = float(visible and uv.z > 0 and 0 <= uv.x <= 1 and 0 <= uv.y <= 1)
        obj.data.uv_layers.active = obj.data.uv_layers[original_uv]
        obj.data.uv_layers[original_uv].active_render = True
        atlas = bpy.data.images.new(f"refined_{index}", width=job["texture_size"], height=job["texture_size"])
        pairs = []
        for slot_index, original in enumerate(list(obj.data.materials)):
            material = original.copy()
            material.use_nodes = True
            tree = material.node_tree
            shader = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if shader is None:
                raise ValueError("Retexture requires Principled BSDF")
            base = shader.inputs["Base Color"]
            old = base.links[0].from_socket if base.is_linked else tuple(base.default_value)
            uvnode = node(tree, "ShaderNodeUVMap", uv_map="synth_projection")
            texture = node(tree, "ShaderNodeTexImage", image=projection, extension="CLIP")
            tree.links.new(uvnode.outputs["UV"], texture.inputs["Vector"])
            visible = node(tree, "ShaderNodeAttribute", attribute_name="synth_visible")
            mix = node(tree, "ShaderNodeMixRGB")
            tree.links.new(visible.outputs["Fac"], mix.inputs[0])
            socket(tree, mix.inputs[1], old)
            tree.links.new(texture.outputs["Color"], mix.inputs[2])
            emission = node(tree, "ShaderNodeEmission")
            tree.links.new(mix.outputs[0], emission.inputs["Color"])
            output = next(n for n in tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
            original_surface = output.inputs["Surface"].links[0].from_socket
            tree.links.new(emission.outputs[0], output.inputs["Surface"])
            target = node(tree, "ShaderNodeTexImage", image=atlas)
            tree.nodes.active = target
            obj.data.materials[slot_index] = material
            pairs.append((tree,shader,output,original_surface,target))
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.bake(type="EMIT")
        atlas.filepath_raw = str(Path(job["output"]) / f"base_color_{index}.png")
        atlas.file_format = "PNG"
        atlas.save()
        for tree,shader,output,surface,target in pairs:
            tree.links.new(surface, output.inputs["Surface"])
            uvnode = node(tree, "ShaderNodeUVMap", uv_map=original_uv)
            tree.links.new(uvnode.outputs[0],target.inputs["Vector"])
            tree.links.new(target.outputs["Color"],shader.inputs["Base Color"])
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(Path(job["output"]) / "bumper_refined.glb"),
                             export_format="GLB", use_selection=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    job = json.loads(Path(args.job).read_text(encoding="utf-8-sig"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if job.get("fixture"):
        fixture(job["output"])
    elif job.get("reference"):
        reference_job(job)
    elif job.get("retexture"):
        retexture_job(job)
    else:
        render_job(job)


if __name__ == "__main__":
    main()
