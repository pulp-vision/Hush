#!/usr/bin/env python3
"""
ONNX export for DeepFilterNet-SE.

Exports the model to ONNX format with dynamic time axis for streaming inference.

Usage:
    python export/onnx_export.py --checkpoint checkpoints/best.pt --output-dir exports
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import onnx.checker
import torch
from torch import Tensor

# Optional: onnxruntime for validation
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def load_model(checkpoint_path: Path, device: str = "cpu"):
    """Load trained model from checkpoint."""
    from model.dfnet_se import create_model
    from training.config import get_default_config
    
    config = get_default_config()
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, config


class DfNetSEExport(torch.nn.Module):
    """Wrapper for ONNX export with explicit input/output handling."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, mixture: Tensor) -> Tensor:
        """
        Args:
            mixture: Audio waveform [B, 1, T]
        
        Returns:
            Enhanced audio [B, 1, T]
        """
        return self.model(mixture)


def export_full_model(
    model: torch.nn.Module,
    output_path: Path,
    sample_rate: int = 16000,
    opset_version: int = 14,
    check: bool = True,
) -> Path:
    """
    Export full model to ONNX.
    
    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        sample_rate: Sample rate
        opset_version: ONNX opset version
        check: Whether to validate the exported model
    
    Returns:
        Path to exported ONNX file
    """
    model.eval()
    export_model = DfNetSEExport(model)
    
    # Create dummy input (3 seconds of audio)
    duration = 3.0
    batch_size = 1
    n_samples = int(duration * sample_rate)
    dummy_input = torch.randn(batch_size, 1, n_samples)
    
    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting model to {output_path}...")
    
    torch.onnx.export(
        export_model,
        (dummy_input,),
        str(output_path),
        input_names=["mixture"],
        output_names=["enhanced"],
        dynamic_axes={
            "mixture": {0: "batch", 2: "time"},
            "enhanced": {0: "batch", 2: "time"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"Exported model to {output_path}")
    
    # Validate
    if check:
        print("Validating ONNX model...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
        if HAS_ORT:
            print("Testing with ONNX Runtime...")
            validate_onnx(model, output_path, dummy_input)
    
    return output_path


def validate_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    test_input: Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    """Validate ONNX model outputs match PyTorch."""
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX Runtime inference
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = sess.run(
        ["enhanced"],
        {"mixture": test_input.numpy()},
    )[0]
    
    # Compare
    try:
        np.testing.assert_allclose(
            pytorch_output, onnx_output, rtol=rtol, atol=atol
        )
        print(f"ONNX validation passed! (rtol={rtol}, atol={atol})")
    except AssertionError as e:
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        print(f"Warning: ONNX output differs from PyTorch. Max diff: {max_diff:.6f}")
        print(f"Error: {e}")


def benchmark_onnx(
    onnx_path: Path,
    sample_rate: int = 16000,
    duration: float = 10.0,
    n_runs: int = 100,
):
    """Benchmark ONNX model inference speed."""
    if not HAS_ORT:
        print("ONNX Runtime not available, skipping benchmark")
        return
    
    import time
    
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    
    # Warm up
    n_samples = int(duration * sample_rate)
    test_input = np.random.randn(1, 1, n_samples).astype(np.float32)
    
    for _ in range(5):
        sess.run(["enhanced"], {"mixture": test_input})
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run(["enhanced"], {"mixture": test_input})
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    rtf = avg_time / duration
    
    print(f"\nBenchmark Results:")
    print(f"  Audio duration: {duration:.1f}s")
    print(f"  Avg inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Real-time factor (RTF): {rtf:.4f}")
    print(f"  {'✓ REAL-TIME' if rtf < 1.0 else '✗ NOT REAL-TIME'}")


def export_streaming_model(
    model: torch.nn.Module,
    output_dir: Path,
    sample_rate: int = 16000,
    frame_size: int = 128,  # hop_size
    opset_version: int = 14,
):
    """
    Export model components for streaming inference.
    
    Exports encoder, decoder separately for frame-by-frame processing.
    """
    # TODO: Implement component-wise export similar to original DeepFilterNet
    # For now, export the full model which works for batch streaming
    print("Note: Full streaming export with separate components not yet implemented")
    print("Using full model export instead...")
    
    return export_full_model(
        model,
        output_dir / "dfnet_se.onnx",
        sample_rate=sample_rate,
        opset_version=opset_version,
    )


def main():
    parser = argparse.ArgumentParser(description="Export DeepFilterNet-SE to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip ONNX validation",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark",
    )
    args = parser.parse_args()
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint)
    
    # Export
    onnx_path = export_full_model(
        model,
        args.output_dir / "dfnet_se.onnx",
        sample_rate=config.model.stft.sample_rate,
        opset_version=args.opset,
        check=not args.no_check,
    )
    
    # Benchmark
    if args.benchmark:
        benchmark_onnx(onnx_path, sample_rate=config.model.stft.sample_rate)
    
    # Save config
    import json
    config_path = args.output_dir / "config.json"
    with open(config_path, "w") as f:
        from training.config import config_to_dict
        json.dump(config_to_dict(config), f, indent=2)
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
