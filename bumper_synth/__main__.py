import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pixal3D bumper experiments and synthetic YOLO data")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--config", default="configs/vast/baseline.json")
    infer.add_argument("--image", required=True)
    infer.add_argument("--mask")
    infer.add_argument("--output", required=True)
    infer.add_argument("--models-lock", required=True)
    export = commands.add_parser("export")
    export.add_argument("--state", required=True)
    export.add_argument("--config", default="configs/vast/baseline.json")
    export.add_argument("--output", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--asset", required=True)
    inspect.add_argument("--output", required=True)
    preview = commands.add_parser("preview", help="Four unlabelled views for asset review")
    preview.add_argument("--asset", required=True)
    preview.add_argument("--output", required=True)
    preview.add_argument("--blender", default="blender")
    review = commands.add_parser("review", help="Record YOUR visual review, bound to exact asset hash")
    review.add_argument("--asset", required=True)
    review.add_argument("--note", required=True)
    fixture = commands.add_parser("fixture", help="Create a curved test panel, not training data")
    fixture.add_argument("--blender", default="blender")
    fixture.add_argument("--output", required=True)
    render = commands.add_parser("render")
    render.add_argument("--asset", required=True)
    render.add_argument("--config", default="configs/vast/render.json")
    render.add_argument("--output", required=True)
    render.add_argument("--blender", default="blender")
    render.add_argument("--group", required=True, help="Original part/vehicle/capture group, same group for all derivatives")
    render.add_argument("--split", choices=("train", "val", "test"), default="train")
    render.add_argument("--demo", action="store_true")
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--datasets", nargs="+", required=True)
    assemble.add_argument("--output", required=True)
    align = commands.add_parser("align", help="Render GLB using saved input camera and compare foreground")
    align.add_argument("--run", required=True)
    align.add_argument("--blender", default="blender")
    refine = commands.add_parser("retexture", help="Experimental clean front-view SD inpaint + UV bake")
    refine.add_argument("--run", required=True)
    refine.add_argument("--output", required=True)
    refine.add_argument("--model", required=True, help="Diffusers-compatible inpainting model ID")
    refine.add_argument("--revision", default="main")
    refine.add_argument("--prompt", required=True)
    refine.add_argument("--strength", type=float, default=.25)
    refine.add_argument("--seed", type=int, default=42)
    refine.add_argument("--blender", default="blender")
    args = parser.parse_args()
    if args.command == "infer":
        from .inference import run
        run(args.config, args.image, args.output, args.mask, args.models_lock)
    elif args.command == "export":
        from .common import checked_config
        from .export import export_state
        export_state(args.state, args.output, checked_config(args.config, "inference")["export"])
    elif args.command == "inspect":
        from .quality import inspect_glb
        print(inspect_glb(args.asset, args.output))
    elif args.command == "preview":
        from .preview import preview
        preview(args.asset, args.output, args.blender)
    elif args.command == "review":
        from .quality import approve_asset
        print(approve_asset(args.asset, args.note))
    elif args.command == "fixture":
        from .dataset import fixture
        fixture(args.blender, args.output)
    elif args.command == "render":
        from .dataset import build
        build(args.asset, args.config, args.output, args.blender, args.group, args.split, args.demo)
    elif args.command == "assemble":
        from .dataset import assemble
        assemble(args.datasets, args.output)
    elif args.command == "align":
        from .alignment import compare
        compare(args.run, args.blender)
    elif args.command == "retexture":
        from .retexture import run
        run(args.run, args.output, args.model, args.prompt, args.blender, args.seed, args.strength, args.revision)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import json
        import sys
        from .common import safe_error
        print(json.dumps(safe_error(error), ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
