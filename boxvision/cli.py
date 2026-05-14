#!/usr/bin/env python3
"""
BoxVision CLI — Train, evaluate, export, and run inference.

Usage:
    python -m boxvision.cli datasets
    python -m boxvision.cli train --dataset road-signs --epochs 100
    python -m boxvision.cli export --checkpoint ./runs/best.pt --output ./model.onnx
    python -m boxvision.cli detect --model ./model.onnx --image ./test.jpg
    python -m boxvision.cli benchmark --model ./model.onnx --image ./test.jpg
    python -m boxvision.cli info
"""

import argparse
import sys


def cmd_train(args):
    """Train the BoxVision model."""
    from .config import ModelConfig, TrainConfig, tiny_config, small_config
    from .train import Trainer

    if args.preset == "tiny":
        model_config = tiny_config(
            input_size=(args.input_size, args.input_size),
            pretrained_backbone=True,
        )
    elif args.preset == "small":
        model_config = small_config(
            input_size=(args.input_size, args.input_size),
            pretrained_backbone=True,
        )
    else:
        model_config = ModelConfig(
            input_size=(args.input_size, args.input_size),
            fpn_out_channels=args.fpn_channels,
        )

    train_config = TrainConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.workers,
        save_dir=args.save_dir,
        device=args.device,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )

    trainer = Trainer(model_config, train_config)
    trainer.train(resume_from=args.resume)


def cmd_datasets(args):
    """List datasets registered in datasets.yaml."""
    from .registry import list_datasets, load_dataset, verify_dataset

    entries = list_datasets()
    if not entries:
        print("No datasets registered. Add entries to datasets.yaml.")
        return

    print(f"{'NAME':22s} {'FORMAT':8s} {'STATUS':12s} DESCRIPTION")
    print("-" * 80)
    for entry in entries:
        try:
            spec = load_dataset(entry["name"])
            errors = verify_dataset(spec)
            status = "ready" if not errors else f"{len(errors)} missing"
        except Exception as exc:
            status = f"error: {exc.__class__.__name__}"
        print(f"{entry['name']:22s} {entry['format']:8s} {status:12s} {entry['description']}")


def cmd_export(args):
    """Export model to ONNX."""
    from .config import ModelConfig, ExportConfig
    from .model import build_model
    from .export import BoxVisionONNXExporter

    model_config = ModelConfig(
        pretrained_backbone=False,
        input_size=(args.input_size, args.input_size),
    )

    export_config = ExportConfig(
        output_path=args.output,
        quantize_int8=args.quantize,
        input_size=(args.input_size, args.input_size),
    )

    model = build_model(model_config)
    exporter = BoxVisionONNXExporter(model, export_config)
    exporter.export(checkpoint_path=args.checkpoint)


def cmd_detect(args):
    """Run inference on an image."""
    import cv2
    from .export import BoxVisionONNXInference

    detector = BoxVisionONNXInference(
        model_path=args.model,
        input_size=(args.input_size, args.input_size),
        confidence_threshold=args.confidence,
        nms_threshold=args.nms,
    )

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        sys.exit(1)

    drawn, boxes, scores = detector.detect_and_draw(image)

    print(f"Detected {len(boxes)} objects:")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.astype(int)
        print(f"  [{i+1}] Box: ({x1}, {y1}, {x2}, {y2})  Score: {score:.3f}")

    # Save output
    output_path = args.output or args.image.replace(".", "_detected.")
    cv2.imwrite(output_path, drawn)
    print(f"\nSaved detection result to: {output_path}")


def cmd_benchmark(args):
    """Benchmark inference speed."""
    import cv2
    from .export import BoxVisionONNXInference

    detector = BoxVisionONNXInference(
        model_path=args.model,
        input_size=(args.input_size, args.input_size),
    )

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        sys.exit(1)

    print(f"Benchmarking on {args.runs} runs...")
    results = detector.benchmark(image, num_runs=args.runs)

    print(f"\nResults:")
    print(f"  Average: {results['avg_ms']:.2f} ms")
    print(f"  Min:     {results['min_ms']:.2f} ms")
    print(f"  Max:     {results['max_ms']:.2f} ms")
    print(f"  Std:     {results['std_ms']:.2f} ms")
    print(f"  FPS:     {results['fps']:.1f}")


def cmd_info(args):
    """Print model architecture info."""
    from .config import tiny_config, small_config
    from .model import build_model

    nc = args.num_classes
    mode = "class-agnostic" if nc == 1 else f"multi-class ({nc} classes)"

    for name, config_fn in [("tiny", tiny_config), ("small", small_config)]:
        config = config_fn(pretrained_backbone=False, num_classes=nc)
        model = build_model(config)
        params = model.count_parameters()

        ghost = "Ghost-" if config.fpn_use_ghost else ""
        print(f"BoxVision-{name}  ({mode})")
        print(f"{'='*48}")
        print(f"Backbone:     {config.backbone}")
        print(f"Neck:         {ghost}FPN ({config.fpn_out_channels}ch, top-down)")
        print(f"Head:         FCOS ({config.head_num_convs}-conv, DW-sep)")
        print(f"Classes:      {nc}")
        print(f"Input size:   {config.input_size}")
        print(f"Total params: {params['total']:,}")
        print(f"Model size:   {params['total_mb']:.2f} MB (FP32)")
        print(f"{'='*48}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="boxvision — Lightweight CPU Box Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Datasets ---
    subparsers.add_parser("datasets", help="List datasets registered in datasets.yaml")

    # --- Train ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--preset", type=str, default="small", choices=["tiny", "small"],
                              help="Model preset: tiny (~162K params) or small (~838K params)")
    train_parser.add_argument("--dataset", type=str, default="road-signs",
                              help="Dataset name from datasets.yaml (run `boxvision datasets` to list)")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_parser.add_argument("--input-size", type=int, default=320, help="Input image size")
    train_parser.add_argument("--fpn-channels", type=int, default=48, help="FPN output channels")
    train_parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    train_parser.add_argument("--save-dir", type=str, default="./runs", help="Checkpoint save directory")
    train_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--eval-interval", type=int, default=5, help="Epochs between validation")
    train_parser.add_argument("--save-interval", type=int, default=5, help="Epochs between checkpoint saves")

    # --- Export ---
    export_parser = subparsers.add_parser("export", help="Export to ONNX")
    export_parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    export_parser.add_argument("--output", type=str, default="./boxvision_model.onnx", help="Output ONNX path")
    export_parser.add_argument("--input-size", type=int, default=320, help="Input size")
    export_parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")

    # --- Detect ---
    detect_parser = subparsers.add_parser("detect", help="Run detection on an image")
    detect_parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    detect_parser.add_argument("--image", type=str, required=True, help="Path to input image")
    detect_parser.add_argument("--output", type=str, default=None, help="Output image path")
    detect_parser.add_argument("--input-size", type=int, default=320, help="Input size")
    detect_parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold")
    detect_parser.add_argument("--nms", type=float, default=0.50, help="NMS threshold")

    # --- Benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark inference speed")
    bench_parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    bench_parser.add_argument("--image", type=str, required=True, help="Path to test image")
    bench_parser.add_argument("--input-size", type=int, default=320, help="Input size")
    bench_parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")

    # --- Info ---
    info_parser = subparsers.add_parser("info", help="Print model info")
    info_parser.add_argument("--num-classes", type=int, default=1,
                             help="Number of classes (1 = class-agnostic, >1 = multi-class)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "datasets": cmd_datasets,
        "export": cmd_export,
        "detect": cmd_detect,
        "benchmark": cmd_benchmark,
        "info": cmd_info,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
