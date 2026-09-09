"""Thin adapter over the provided Pixal3D; never silently falls back after OOM."""
import copy
import os
from pathlib import Path

from .common import checked_config, digest, object_digest, read_json, safe_error, source_digest, write_json


def run(config_path, image_path, output_dir, mask_path=None, models_lock=None):
    cfg = checked_config(config_path, "inference")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if models_lock is None:
        raise ValueError("Run prepare_models.py first and provide --models-lock for reproducibility")
    lock = read_json(models_lock)
    if lock.get("model") != cfg["model_path"]:
        raise ValueError("Model lock does not match configured model_path")
    if cfg["background_mode"] == "auto" and lock.get("background_mode") != "auto":
        raise ValueError("Prepare models with --background-mode auto before using auto removal")
    signature = object_digest({"config": cfg, "image": digest(image_path),
                               "mask": digest(mask_path) if mask_path else None,
                               "models": lock, "code": source_digest()})
    manifest = output / "run.json"
    if manifest.exists():
        previous = read_json(manifest)
        if previous["signature"] != signature:
            raise ValueError("Output belongs to another input/config/version; use a new directory")
        if previous["status"] == "complete" and (output / "bumper.glb").exists():
            if digest(output / "bumper.glb") == previous.get("glb_sha256"):
                print(f"Reusing completed run: {output}")
                return
    state = {"signature": signature, "status": "running", "config": cfg,
             "input_sha256": digest(image_path), "models": lock}
    write_json(manifest, state)
    os.environ["ATTN_BACKEND"] = cfg["attention_backend"]
    os.environ["BUMPER_NAF_REF"] = lock["naf_ref"]
    try:
        import torch
        import inference as upstream
        from .preprocess import prepare
        from .telemetry import Recorder
        from .export import save_mesh_state, export_state, EXPORT_TRANSFORM
        from .quality import inspect_glb
        recorder = Recorder(output / "stages.json")
        # Only change per-process config copies. Upstream source defaults stay intact.
        upstream.IMAGE_COND_CONFIGS = copy.deepcopy(upstream.IMAGE_COND_CONFIGS)
        for item in upstream.IMAGE_COND_CONFIGS.values():
            item["model_name"] = lock["paths"]["dino"]
        with recorder.stage("model_loading"):
            pipeline = upstream.init_pipeline(lock["paths"]["pixal"], low_vram=cfg["low_vram"],
                                               load_rembg=cfg["background_mode"] == "auto")
            pipeline.projection_chunk_size = cfg.get("projection_chunk_size", 8192)
            pipeline.require_requested_resolution = cfg.get("require_requested_resolution", False)
        with recorder.stage("preprocess"):
            remover = pipeline.rembg_model
            if remover is not None:
                remover.to("cuda")
            image = prepare(image_path, output, mode=cfg["background_mode"], mask_path=mask_path,
                            remover=remover, padding=cfg["padding"])
            if remover is not None:
                remover.cpu()
            torch.cuda.empty_cache()
        with recorder.stage("camera"):
            if cfg["manual_fov"] > 0:
                import math
                fov = cfg["manual_fov"]
                if fov >= math.pi:
                    raise ValueError("manual_fov must be in radians, 0 < FOV < pi")
                distance = upstream.distance_from_fov(fov, torch.tensor([-1., 0., 0.]),
                            torch.tensor([0., 511.]), 1., 512)["distance_from_x"]
                camera = {"camera_angle_x": fov, "distance": distance, "mesh_scale": 1.}
            else:
                model = upstream.load_moge_model(model_name=lock["paths"]["moge"])
                camera = upstream.get_camera_params_wild_moge(str(output / "model_input.png"), model)
                model.cpu()
                del model
                torch.cuda.empty_cache()
        write_json(output / "camera.json", {**camera, "export_transform": EXPORT_TRANSFORM.tolist(),
                                           "note": "Camera in upstream coordinates; apply export transform when rendering GLB."})
        if cfg.get("bumper_constraint"):
            from .geometry_constraint import FrontPartConstraint
            pipeline.support_constraint = FrontPartConstraint(
                output / "input_mask.png", camera, cfg["bumper_constraint"])
        for name in ("get_proj_cond_ss", "sample_sparse_structure", "get_proj_cond_shape",
                     "sample_tex_slat", "decode_latent"):
            if hasattr(pipeline, name):
                recorder.wrap(pipeline, name)
        recorder.wrap(pipeline.shape_slat_sampler, "sample", label="shape_sampling")
        recorder.wrap(pipeline.models["shape_slat_decoder"], "upsample", label="shape_upsample")
        with torch.inference_mode():
            meshes, (_, _, resolution) = pipeline.run(
                image, camera_params=camera, seed=cfg["seed"], preprocess_image=False, return_latent=True,
                sparse_structure_sampler_params=cfg["samplers"]["ss"],
                shape_slat_sampler_params=cfg["samplers"]["shape"],
                tex_slat_sampler_params=cfg["samplers"]["texture"],
                pipeline_type=f"{cfg['resolution']}_cascade", max_num_tokens=cfg["max_num_tokens"])
        del _
        with recorder.stage("save_intermediate"):
            save_mesh_state(meshes[0], resolution, pipeline.pbr_attr_layout, output / "mesh_state.npz")
        del meshes, pipeline
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        with recorder.stage("glb_export"):
            export_state(output / "mesh_state.npz", output / "bumper.glb", cfg["export"])
        inspect_glb(output / "bumper.glb", output / "quality.json")
        state.update(status="complete", glb_sha256=digest(output / "bumper.glb"),
                     semantic_status="needs_human_review")
    except Exception as error:
        state.update(status="failed", error=safe_error(error))
        raise
    finally:
        write_json(manifest, state)
