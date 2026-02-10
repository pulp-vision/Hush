
import time
import torch
import sys
import os

# Add the current directory to sys.path to allow imports from model/
sys.path.append(os.getcwd())

from model.dfnet_se import DfNetSE, ModelConfig

def main():
    print("Initializing model...")
    try:
        model = DfNetSE()
        model.eval()
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Create dummy input: 10 seconds of audio at 16 kHz
    sr = 16000
    duration = 10.0
    input_len = int(sr * duration)
    x = torch.randn(1, 1, input_len)
    
    print(f"Input shape: {x.shape} ({duration} seconds)")

    # Warm up
    print("Warming up...")
    try:
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
    except Exception as e:
        print(f"Error during warm-up: {e}")
        return

    # Measure execution time
    num_runs = 10
    total_time = 0.0
    
    print(f"Running {num_runs} iterations...")
    try:
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                _ = model(x)
                end_time = time.time()
                total_time += (end_time - start_time)
                print(f"Run {i+1}: {end_time - start_time:.4f} s")
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    avg_time = total_time / num_runs
    rtf = avg_time / duration
    
    print("-" * 30)
    print(f"Average execution time: {avg_time:.4f} s")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")
    print("-" * 30)
    
    if rtf < 1.0:
        print("Model runs faster than real-time.")
    else:
        print("Model runs slower than real-time.")

if __name__ == "__main__":
    main()
