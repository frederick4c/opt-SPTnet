import argparse
import os
import glob
import time
import tempfile
from collections import defaultdict
from os.path import dirname, basename

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from sptnet import SPTnet, Transformer, Transformer3d
from sptnet.data import FileSampleDataset, SubsetByIndices, collate_inference
from sptnet.inference import (
    extract_state_dict,
    get_num_frames,
    get_num_queries,
    load_checkpoint_strict_enough,
    run_batched_inference,
)

# Set up processing device
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Device selection logic
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def parse_args():
    p = argparse.ArgumentParser(description="SPTnet Parallel Inference on Colab/CSD3")
    p.add_argument('-m', '--model-path', type=str, required=True, help="Path to the trained model file (e.g. .../trained_model)")
    p.add_argument('-d', '--data', type=str, nargs='+', required=True, help="Path(s) to test data files (.h5/.hdf5, MATLAB v7.3 .mat, or .tif)")
    p.add_argument('-b', '--batch-size', type=int, default=8, help="Batch size for parallel inference (default: 8)")
    p.add_argument(
        '--hdf5-clip-index',
        '--mat-clip-index',
        dest='mat_clip_index',
        type=int,
        default=0,
        help="For 4D HDF5 files (N,T,H,W), only run this clip index (default: 0).",
    )
    return p.parse_args()

def main():
    t0_all = time.time()
    args = parse_args()
    
    model_path = args.model_path
    if not os.path.isfile(model_path):
        possible_file = os.path.join(model_path, "trained_model")
        if os.path.isfile(possible_file):
            model_path = possible_file
            print(f"Warning: Directory provided. Using model file: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"Selected device = {device}")
    if torch.cuda.is_available():
        print(f"CUDA device name = {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count = {torch.cuda.device_count()}")

    # Expand data file patterns
    filename_test = []
    if isinstance(args.data, list):
        for data_arg in args.data:
            filename_test.extend(glob.glob(data_arg))
    else:
        filename_test.extend(glob.glob(args.data))

    # Remove duplicates and sort
    filename_test = sorted(list(set(filename_test)))

    if not filename_test:
        print(f"Error: No data files found matching patterns: {args.data}")
        return

    print(f"Found {len(filename_test)} test files.")
    for i, fp in enumerate(filename_test, start=1):
        print(f"  [{i:03d}] {fp}")

    t0 = time.time()
    ckpt_cpu = torch.load(model_path, map_location="cpu")
    state_dict_cpu = extract_state_dict(ckpt_cpu)
    num_q = get_num_queries(state_dict=state_dict_cpu)
    num_frames = get_num_frames(state_dict=state_dict_cpu)
    print(f"Read checkpoint metadata in {time.time() - t0:.2f}s (num_queries={num_q}, num_frames={num_frames})")

    infer_batch_size = args.batch_size

    print("Initializing FileSampleDataset...")
    print(f"HDF5 clip index for 4D files: {args.mat_clip_index}")
    all_samples = FileSampleDataset(filename_test, mat_clip_index=args.mat_clip_index)
    bad_shapes = [shape_key for shape_key in all_samples.shape_groups if shape_key[0] != num_frames]
    if bad_shapes:
        raise ValueError(
            f"Checkpoint expects {num_frames} frames, but inference data contains shape group(s): "
            f"{bad_shapes}. Use data with matching T or a checkpoint trained for that frame count."
        )

    transformer3d = Transformer3d(
        d_model=256,
        dropout=0,
        nhead=8,
        dim_feedforward=1024,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False
    )
    transformer = Transformer(
        d_model=256,
        dropout=0,
        nhead=8,
        dim_feedforward=1024,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False
    )
    model = SPTnet(
        num_classes=1,
        num_queries=num_q,
        num_frames=num_frames,
        spatial_t=transformer,
        temporal_t=transformer3d,
        input_channel=512
    ).to(device)

    t0 = time.time()
    load_checkpoint_strict_enough(model, state_dict=state_dict_cpu)
    print(f"Loaded checkpoint into model in {time.time() - t0:.2f}s")
    model.eval()

    # Optional: if you really have multiple GPUs, uncomment the next 2 lines.
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    all_results = []

    print(f"Starting batched inference (batch_size={infer_batch_size})...")
    # Batch only samples with the same (T, H, W), otherwise torch.stack will fail.
    t0_inf = time.time()
    for shape_key, indices in all_samples.shape_groups.items():
        print(f"  Shape group {shape_key}: {len(indices)} sample(s)")
        subset = SubsetByIndices(all_samples, indices)
        test_dataloader = DataLoader(
            subset,
            batch_size=infer_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_inference
        )
        all_results.extend(run_batched_inference(model, test_dataloader, device))
    print(f"Inference pass completed in {time.time() - t0_inf:.2f}s")

    results_by_file = defaultdict(lambda: {
        'obj_estimation': [],
        'estimation_xy': [],
        'estimation_H': [],
        'estimation_C': [],
    })

    for rec in sorted(all_results, key=lambda x: (x['file_path'], x['sample_idx'])):
        results_by_file[rec['file_path']]['obj_estimation'].append(rec['obj_estimation'])
        results_by_file[rec['file_path']]['estimation_xy'].append(rec['estimation_xy'])
        results_by_file[rec['file_path']]['estimation_H'].append(rec['estimation_H'])
        results_by_file[rec['file_path']]['estimation_C'].append(rec['estimation_C'])

    save_dir = os.path.join(dirname(model_path), 'inference_results')
    os.makedirs(save_dir, exist_ok=True)

    t0_save = time.time()
    for file_path, rec in results_by_file.items():
        base = os.path.splitext(basename(file_path))[0] + '.mat'
        estimation_obj = np.vstack(rec['obj_estimation'])
        estimation_obj = np.expand_dims(estimation_obj, axis=1) # Shape: [N, 1, Q, T] to match MATLAB GUI expectation
        estimation_xy = np.vstack(rec['estimation_xy'])
        estimation_H = np.vstack(rec['estimation_H'])
        estimation_C = np.vstack(rec['estimation_C'])

        output_path = os.path.join(save_dir, 'result_' + basename(base))
        if os.path.exists(output_path):
            print(f"Overwriting existing result file: {output_path}")
        else:
            print(f"Writing new result file: {output_path}")

        # Write atomically: save to temp file in same directory, then replace.
        # This prevents stale/partial files if a job is interrupted during save.
        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix='.mat',
            prefix='tmp_result_',
            dir=save_dir,
            delete=False
        ) as tf:
            tmp_output_path = tf.name

        sio.savemat(
            tmp_output_path,
            mdict={
                'obj_estimation': estimation_obj,
                'estimation_xy': estimation_xy,
                'estimation_H': estimation_H,
                'estimation_C': estimation_C,
            }
        )
        os.replace(tmp_output_path, output_path)

    print(f"Result saving completed in {time.time() - t0_save:.2f}s")
    print(f'Done. Saved inference results to: {save_dir}')
    print(f"Total runtime: {time.time() - t0_all:.2f}s")

if __name__ == "__main__":
    main()
