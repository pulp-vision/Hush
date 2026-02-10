
import json
import os

def extract_paths(json_path, output_txt):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    elif isinstance(data, list):
        samples = data
    else:
        print(f"Unknown format for {json_path}")
        return

    # Convert to absolute paths
    # Assuming the paths in json are relative to current working directory (project root)
    paths = [os.path.abspath(s['path']) for s in samples]
    
    with open(output_txt, 'w') as f:
        f.write('\n'.join(paths))
    print(f"Extracted {len(paths)} paths to {output_txt}")

os.makedirs('data/hdf5_prep', exist_ok=True)
extract_paths('data/manifests/train.json', 'data/hdf5_prep/train_speech.txt')
extract_paths('data/manifests/val.json', 'data/hdf5_prep/val_speech.txt')
extract_paths('data/manifests/noise.json', 'data/hdf5_prep/train_noise.txt')
# For RIRs, we have a manifest text file, not json.
# We should probably handle RIRs separately or manually ensure they are absolute too.
# But likely prepare_data.py for RIR handles it if we pass absolute path to it?
# Actually, the user command for RIR used data/rir/rir_manifest.txt
# Let's check if rir_manifest.txt has absolute paths.
