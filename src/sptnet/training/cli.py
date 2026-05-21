import os
import time
import argparse
import glob
import csv
import random
from types import SimpleNamespace
if 'MPLBACKEND' in os.environ:
    del os.environ['MPLBACKEND']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sptnet import SPTnet, Transformer, Transformer3d, TransformerMatDataset
from sptnet.data import create_train_val_loaders
from sptnet.training import hungarian_matched_loss, normalize_training_inputs
from sptnet.training.crlb import load_or_generate_crlb_matrix
from tqdm import tqdm
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename
# from tkinter.filedialog import askdirectory
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- Ampere / Turing GPU optimizations ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

current_folder = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 68


def _default_crlb_path():
    candidates = [
        os.path.join(os.getcwd(), 'CRLB_H_D_frame.mat'),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'CRLB_H_D_frame.mat')),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def set_random_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_loss_history(csv_log_path):
    history = {
        'epoch': [],
        't_loss': [],
        'v_loss': [],
        't_cls': [],
        'v_cls': [],
        't_coor': [],
        'v_coor': [],
        't_hurst': [],
        'v_hurst': [],
        't_diff': [],
        'v_diff': [],
        't_bg': [],
        'v_bg': [],
    }
    if not csv_log_path or not os.path.exists(csv_log_path):
        return history

    with open(csv_log_path, 'r', newline='') as csv_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            try:
                history['epoch'].append(int(float(row['epoch'])))
                for key in history:
                    if key != 'epoch':
                        history[key].append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                continue
    return history


def _copy_loss_history(src_csv, dst_csv):
    if not src_csv or not os.path.exists(src_csv) or os.path.abspath(src_csv) == os.path.abspath(dst_csv):
        return
    with open(src_csv, 'r') as src, open(dst_csv, 'w') as dst:
        dst.write(src.read())

def parse_args():
    #define training parameters based user's inputs
    p = argparse.ArgumentParser(description="Training SPTnet with user defined parameters")
    p.add_argument('-b', '--batch-size',  type=int,   default=16,     help="training batch size")
    p.add_argument('-g', '--gpus',        type=int,   default=1,      help="number of GPUs to use")
    p.add_argument('-lr','--learning-rate',type=float, default=0.0001, help="initial learning rate")
    p.add_argument('-m','--model-dir',    type=str,   default='.',    help="where to save/load model")
    p.add_argument('-q', '--query', type=int, default=20, help="number of query")
    p.add_argument('-dc', '--max_dc', type=float, default=0.5, help="the maximum diffusion coefficient among all training data")
    p.add_argument('--val-every', type=int, default=1, help="run validation every N epochs")
    p.add_argument('--patience', type=int, default=6, help="early-stopping patience measured in validation checks")
    p.add_argument('--max-epochs', type=int, default=0, help="maximum epochs to run; 0 means no explicit cap")
    p.add_argument('--grad-clip', type=float, default=1.0, help="max gradient norm; <=0 disables clipping")
    p.add_argument('--max-train-batches', type=int, default=0, help="maximum train batches per epoch; 0 means all")
    p.add_argument('--max-val-batches', type=int, default=0, help="maximum validation batches per validation pass; 0 means all")
    p.add_argument('--resume', type=str, default='', help="path to model weights to resume from")
    p.add_argument('--resume-optimizer', type=str, default='', help="path to optimizer state; defaults to <resume>optimizer_stat")
    p.add_argument('--resume-history', type=str, default='', help="path to existing loss_history.csv; defaults to output model dir CSV")
    p.add_argument('--crlb-path', type=str, default='', help="path to CRLB_H_D_frame.mat; defaults to ./CRLB_H_D_frame.mat")
    p.add_argument('-d', '--data', type=str, nargs='+', help="Path to training data HDF5 files (.h5/.hdf5 or MATLAB v7.3 .mat)")
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seeds()
    
    model_name = "trained_model"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    full_path = os.path.join(args.model_dir, model_name)
    print(f"Model will be saved to: {os.path.abspath(full_path)}")
    print(f"Random seed set to: {RANDOM_SEED}")
    
    # Verify write permissions immediately
    try:
        test_file_path = full_path + "_write_test.tmp"
        with open(test_file_path, 'w') as f:
            f.write("Write test successful.")
        os.remove(test_file_path)
        print("✅ Write check passed: Output directory is writable.")
    except Exception as e:
        print(f"❌ Write check failed: Cannot write to {os.path.abspath(full_path)}")
        raise RuntimeError(f"Output directory is not writable: {e}")

    if args.gpus > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    use_amp = device.type == 'cuda'

    # Tk().withdraw() # keep the root window from appearing
    # filename_train = askopenfilename(multiple=True, initialdir=current_folder, title='#######Please select all the Training Data File########') # show an "Open" dialog box and return the path to the selected file
    
    if args.data:
        filename_train = []
        for pattern in args.data:
            filename_train.extend(glob.glob(pattern))
        filename_train = sorted(filename_train)
    else:
        raise RuntimeError("No training data provided! Use --data argument.")

    data_train = []

    if not filename_train:
        raise RuntimeError("No training data selected!")

    # Normalize to a Python list
    if isinstance(filename_train, tuple):
        training_files = list(filename_train)
    else:
        # In this specific case, filename_train is already a list of strings
        training_files = filename_train
    # Only read the FIRST file to get image dimensions — reading all 1000 files just for H/W
    # is a huge bottleneck (~minutes) and completely unnecessary.
    with h5py.File(training_files[0], 'r') as f:
        data = f['timelapsedata']
        if data.ndim == 4:
            (n_videos, n_frames, H, W) = data.shape
        elif data.ndim == 3:
            (n_frames, H, W) = data.shape
        else:
            raise ValueError(f"Unexpected timelapsedata shape: {data.shape}")

    spt = SimpleNamespace(
        path_saved_model=full_path,
        momentum=0.9,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_gpu=(args.gpus > 0 and torch.cuda.is_available()),
        image_size=H,
        number_of_frame=n_frames,
        num_queries= args.query,
        diff_max = args.max_dc
    )

    file_path = args.crlb_path or _default_crlb_path()
    CRLB_matrix = load_or_generate_crlb_matrix(
        file_path,
        frame_number=spt.number_of_frame,
        diff_max=spt.diff_max,
    )

    print(f"Loading training data from {len(training_files)} files...")
    all_datasets = []
    for i, file in enumerate(training_files):
        print(f"Processing file {i+1}/{len(training_files)}: {file}")
        datafile = TransformerMatDataset(config=spt, dataset_path=file)
        all_datasets.append(datafile)
    # Build a FLAT ConcatDataset in one call. Nesting ConcatDataset inside ConcatDataset
    # on every iteration creates O(N) recursion depth, which overflows Python's limit at ~1000 files.
    data_train = torch.utils.data.ConcatDataset(all_datasets)
    print(f"Data loading complete. Total samples: {len(data_train)}")
    train_size = int(len(data_train) * 0.8)
    val_size = len(data_train) - train_size  # remainder goes to val, guarantees they sum correctly
    train_dataloader, val_dataloader, _, _ = create_train_val_loaders(
        data_train,
        [train_size, val_size],
        batch_size=spt.batch_size,
    )

    def train_step(batch_idx, data):
        model.train()
        inputs, Hlabel, Clabel, position_label, class_label = data['video'], data['Hlabel'], data['Clabel'], data['position'], data['class_label']
        inputs = torch.unsqueeze(inputs, 1).float().to(device) # float64 is actually "double"
        inputs = normalize_training_inputs(inputs)

        class_label = class_label.float().to(device)
        position_label = (position_label / (spt.image_size / 2)).float().to(device)
        Hlabel = Hlabel.float().to(device)
        Clabel = (Clabel / spt.diff_max).float().to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            class_out, center_out, H_out, C_out = model(inputs)
        # Cast back to float32 for loss computation (BCE is unsafe under autocast)
        class_out = class_out.float()
        center_out = center_out.float()
        H_out = H_out.float().squeeze(-1)  # (B, Q, 1) -> (B, Q)
        C_out = C_out.float().squeeze(-1)  # (B, Q, 1) -> (B, Q)
        t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = hungarian_matched_loss(
            class_out,
            center_out,
            H_out,
            C_out,
            class_label,
            position_label,
            Hlabel,
            Clabel,
            num_queries=spt.num_queries,
            diff_max=spt.diff_max,
            num_frames=spt.number_of_frame,
            crlb_matrix=CRLB_matrix,
        )
        if not torch.isfinite(t_loss):
            raise RuntimeError(
                f"Non-finite training loss at batch {batch_idx}: "
                f"loss={t_loss}, class={cl_ls}, coord={coor_ls}, H={h_ls}, D={diff_ls}, bg={bg_ls}"
            )
        scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(grad_norm):
                print(f"Skipping batch {batch_idx}: non-finite gradient norm {grad_norm}.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                return float(t_loss.detach().cpu()), 0.0, 0.0, 0.0, 0.0, 0.0
        scaler.step(optimizer)
        scaler.update()
        t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = float(t_loss), float(cl_ls), float(coor_ls), float(h_ls), float(diff_ls), float(bg_ls)
        return t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls

    def val_step(batch_idx, data):
        model.eval()
        with torch.no_grad():
            inputs, Hlabel, Clabel, position_label, class_label = data['video'], data['Hlabel'], data['Clabel'], data['position'], data['class_label']
            inputs = torch.unsqueeze(inputs, 1).float().to(device) # float64 is actually "double"
            inputs = normalize_training_inputs(inputs)

            class_label = class_label.float().to(device)
            position_label = (position_label / (spt.image_size / 2)).float().to(device)
            Hlabel = Hlabel.float().to(device)
            Clabel = (Clabel / spt.diff_max).float().to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                class_out, center_out, H_out, C_out = model(inputs)
            # Cast back to float32 for loss computation (BCE is unsafe under autocast)
            class_out = class_out.float()
            center_out = center_out.float()
            H_out = H_out.float().squeeze(-1)  # (B, Q, 1) -> (B, Q)
            C_out = C_out.float().squeeze(-1)  # (B, Q, 1) -> (B, Q)
            v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = hungarian_matched_loss(
                class_out,
                center_out,
                H_out,
                C_out,
                class_label,
                position_label,
                Hlabel,
                Clabel,
                num_queries=spt.num_queries,
                diff_max=spt.diff_max,
                num_frames=spt.number_of_frame,
                crlb_matrix=CRLB_matrix,
            )
            v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = float(v_loss), float(cl_ls), float(coor_ls), float(h_ls), float(diff_ls), float(bg_ls)
        return v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls

    torch.backends.cudnn.benchmark = True  # use the fastest convolution methods when the inputs size are fixed improves performance
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    transformer3d = Transformer3d(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
    transformer = Transformer(d_model=256,dropout=0,nhead=8,dim_feedforward=1024,num_encoder_layers=6,num_decoder_layers=6,normalize_before=False)
    print("Initializing model...")
    model = SPTnet(num_classes=1, num_queries=spt.num_queries, num_frames=spt.number_of_frame, spatial_t=transformer,
                       temporal_t=transformer3d, input_channel=512).to(device)
    # torch.autograd.set_detect_anomaly(True)  # Disabled for performance; re-enable only for debugging
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # AMP gradient scaler for mixed-precision training

    if args.gpus > 1:
        device_ids = list(range(args.gpus))
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=spt.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # scheduler_rdpl = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=5, verbose=True,
    #                                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                                  eps=1e-08)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)
    ##############################################
    csv_log_path = spt.path_saved_model + 'loss_history.csv'
    resume_history_path = args.resume_history or csv_log_path
    if args.resume_history:
        _copy_loss_history(args.resume_history, csv_log_path)
    loss_history = _load_loss_history(resume_history_path)

    t_loss_append = loss_history['t_loss']
    t_loss_epoch_cls_append = loss_history['t_cls']
    t_loss_epoch_coor_append = loss_history['t_coor']
    t_loss_epoch_hurst_append = loss_history['t_hurst']
    t_loss_epoch_diff_append = loss_history['t_diff']
    t_loss_epoch_bg_append = loss_history['t_bg']

    v_loss_append = loss_history['v_loss']
    v_loss_epoch_cls_append = loss_history['v_cls']
    v_loss_epoch_coor_append = loss_history['v_coor']
    v_loss_epoch_hurst_append = loss_history['v_hurst']
    v_loss_epoch_diff_append = loss_history['v_diff']
    v_loss_epoch_bg_append = loss_history['v_bg']

    epoch_list = loss_history['epoch']

    no_improvement = 0
    finite_v_losses = [v for v in v_loss_append if np.isfinite(v)]
    min_v_loss = min(finite_v_losses) if finite_v_losses else 99999999999
    max_num_of_epoch_without_improving = args.patience
    epoch = (max(epoch_list) + 1) if epoch_list else 1

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming model weights from: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device)
        target_model = model.module if args.gpus > 1 else model
        incompatible = target_model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"Resume checkpoint loaded with missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )

        optimizer_path = args.resume_optimizer or (args.resume + 'optimizer_stat')
        if optimizer_path and os.path.exists(optimizer_path):
            print(f"Resuming optimizer state from: {optimizer_path}")
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            for group in optimizer.param_groups:
                group['lr'] = spt.learning_rate
            print(f"Optimizer state loaded; learning rate set to {spt.learning_rate}.")
        else:
            print(f"No optimizer state found at {optimizer_path}; continuing with a fresh optimizer.")

    if epoch_list:
        print(f"Loaded {len(epoch_list)} previous loss-history rows; next epoch is {epoch}.")
        print(f"Best previous finite validation loss: {min_v_loss}")
    #
    start = time.time()
    lr = []

    modelrecord = open(spt.path_saved_model + 'training_log.txt', 'a')
    # Lightweight CSV log — writing a matplotlib figure on every epoch is slow on a headless node.
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as csv_f:
            csv_f.write('epoch,t_loss,v_loss,t_cls,v_cls,t_coor,v_coor,t_hurst,v_hurst,t_diff,v_diff,t_bg,v_bg\n')
    while no_improvement < max_num_of_epoch_without_improving and (args.max_epochs <= 0 or epoch <= args.max_epochs):
    # for epoch in range(n_epochs):
        print(f"Starting Epoch {epoch}...")
        epoch_list.append(epoch)
        t_loss_total = 0
        t_loss_total_cls = 0
        t_loss_total_coor = 0
        t_loss_total_hurst = 0
        t_loss_total_diff = 0
        t_loss_total_bg = 0
        v_loss_total = 0
        v_loss_total_cls = 0
        v_loss_total_coor = 0
        v_loss_total_hurst = 0
        v_loss_total_diff = 0
        v_loss_total_bg = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} [train]")
        for batch_idx, data in enumerate(pbar):
            t_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = train_step(batch_idx, data)
            t_loss_total += t_loss
            t_loss_total_cls += cl_ls
            t_loss_total_coor += coor_ls
            t_loss_total_hurst += h_ls
            t_loss_total_diff += diff_ls
            t_loss_total_bg += bg_ls
            pbar.set_postfix({'loss': f'{t_loss:.4f}'})
            if args.max_train_batches > 0 and (batch_idx + 1) >= args.max_train_batches:
                print(f"Stopping train epoch early after {args.max_train_batches} batches (--max-train-batches).")
                break
        t_loss_epoch = t_loss_total/(batch_idx+1)
        t_loss_epoch_cls  = t_loss_total_cls/(batch_idx+1)
        t_loss_epoch_coor = t_loss_total_coor/(batch_idx+1)
        t_loss_epoch_hurst = t_loss_total_hurst/(batch_idx+1)
        t_loss_epoch_diff = t_loss_total_diff/(batch_idx+1)
        t_loss_epoch_bg = t_loss_total_bg / (batch_idx + 1)
            # lr.append(scheduler_rdpl.get_lr()[0])

        did_validate = (epoch == 1) or (args.val_every > 0 and epoch % args.val_every == 0)
        if did_validate:
            for batch_idx, data in enumerate(val_dataloader):
                v_loss, cl_ls, coor_ls, h_ls, diff_ls, bg_ls = val_step(batch_idx, data)
                v_loss_total+=v_loss
                v_loss_total_cls += cl_ls
                v_loss_total_coor += coor_ls
                v_loss_total_hurst += h_ls
                v_loss_total_diff += diff_ls
                v_loss_total_bg += bg_ls
                if args.max_val_batches > 0 and (batch_idx + 1) >= args.max_val_batches:
                    print(f"Stopping validation early after {args.max_val_batches} batches (--max-val-batches).")
                    break
            v_loss_epoch = v_loss_total / (batch_idx + 1)
            v_loss_epoch_cls = v_loss_total_cls / (batch_idx + 1)
            v_loss_epoch_coor = v_loss_total_coor / (batch_idx + 1)
            v_loss_epoch_hurst = v_loss_total_hurst / (batch_idx + 1)
            v_loss_epoch_diff = v_loss_total_diff / (batch_idx + 1)
            v_loss_epoch_bg = v_loss_total_bg / (batch_idx + 1)
            print(f"Finished validation for Epoch {epoch}. Loss: {v_loss_epoch:.4f}")
        else:
            v_loss_epoch = float('nan')
            v_loss_epoch_cls = float('nan')
            v_loss_epoch_coor = float('nan')
            v_loss_epoch_hurst = float('nan')
            v_loss_epoch_diff = float('nan')
            v_loss_epoch_bg = float('nan')
            print(f"Skipping validation for Epoch {epoch}; next validation check is controlled by --val-every {args.val_every}.")

        if did_validate and v_loss_epoch < min_v_loss:
            min_v_loss = v_loss_epoch
            no_improvement = 0
            if args.gpus > 1:
                torch.save(model.module.state_dict(), spt.path_saved_model) #Save model.module.state_dict() in DP case!!!
            else:
                torch.save(model.state_dict(), spt.path_saved_model)
            torch.save(optimizer.state_dict(), spt.path_saved_model+'optimizer_stat')
            print('==> Saving a new best model')
        elif did_validate:
            no_improvement+=1
        else:
            print(f"Early stopping patience unchanged ({no_improvement}/{max_num_of_epoch_without_improving}) because validation was skipped.")
        lr.append(optimizer.param_groups[0]['lr'])
        # scheduler_rdpl.step(v_loss_epoch)
        print('learning rate is: %f' %lr[-1])

        # ---- Collect per-epoch metrics into lists ----
        t_loss_append.append(t_loss_epoch)
        v_loss_append.append(v_loss_epoch)
        t_loss_epoch_cls_append.append(t_loss_epoch_cls)
        v_loss_epoch_cls_append.append(v_loss_epoch_cls)
        t_loss_epoch_coor_append.append(t_loss_epoch_coor)
        v_loss_epoch_coor_append.append(v_loss_epoch_coor)
        t_loss_epoch_hurst_append.append(t_loss_epoch_hurst)
        v_loss_epoch_hurst_append.append(v_loss_epoch_hurst)
        t_loss_epoch_diff_append.append(t_loss_epoch_diff)
        v_loss_epoch_diff_append.append(v_loss_epoch_diff)
        t_loss_epoch_bg_append.append(t_loss_epoch_bg)
        v_loss_epoch_bg_append.append(v_loss_epoch_bg)

        # ---- Write to text log (identical format to original) ----
        modelrecord.write('\nepoch %d, t_loss: %s, v_loss: %s' % (epoch, t_loss_epoch, v_loss_epoch))
        modelrecord.write(', t_cls_loss: %s, v_cls_loss: %s' % (t_loss_epoch_cls, v_loss_epoch_cls))
        modelrecord.write(', t_coor_loss: %s, v_coor_loss: %s' % (t_loss_epoch_coor, v_loss_epoch_coor))
        modelrecord.write(', t_hurst_loss: %s, v_hurst_loss: %s' % (t_loss_epoch_hurst, v_loss_epoch_hurst))
        modelrecord.write(', t_diff_loss: %s, v_diff_loss: %s' % (t_loss_epoch_diff, v_loss_epoch_diff))
        modelrecord.write(', t_bg_loss: %s, v_bg_loss: %s' % (t_loss_epoch_bg, v_loss_epoch_bg))
        modelrecord.flush()  # ensure log is written even if the job is killed

        # ---- Write to CSV log (easy to load with pandas later) ----
        with open(csv_log_path, 'a') as csv_f:
            csv_f.write(f'{epoch},{t_loss_epoch},{v_loss_epoch},{t_loss_epoch_cls},{v_loss_epoch_cls},'
                        f'{t_loss_epoch_coor},{v_loss_epoch_coor},{t_loss_epoch_hurst},{v_loss_epoch_hurst},'
                        f'{t_loss_epoch_diff},{v_loss_epoch_diff},{t_loss_epoch_bg},{v_loss_epoch_bg}\n')

        # ---- Save learning curve plot every 5 epochs or when a new best model is saved ----
        if epoch % 5 == 0 or (did_validate and no_improvement == 0):
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
            ax[0,0].plot(epoch_list, t_loss_append, 'r', lw=2, label='train')
            ax[0,0].plot(epoch_list, v_loss_append, 'b', lw=2, label='val')
            ax[0,0].set_title('Total loss'); ax[0,0].legend()
            ax[0,1].plot(epoch_list, t_loss_epoch_cls_append, 'r', lw=2)
            ax[0,1].plot(epoch_list, v_loss_epoch_cls_append, 'b', lw=2)
            ax[0,1].set_title('cls loss')
            ax[0,2].plot(epoch_list, t_loss_epoch_coor_append, 'r', lw=2)
            ax[0,2].plot(epoch_list, v_loss_epoch_coor_append, 'b', lw=2)
            ax[0,2].set_title('coordinate loss')
            ax[1,0].plot(epoch_list, t_loss_epoch_hurst_append, 'r', lw=2)
            ax[1,0].plot(epoch_list, v_loss_epoch_hurst_append, 'b', lw=2)
            ax[1,0].set_title('hurst loss')
            ax[1,1].plot(epoch_list, t_loss_epoch_diff_append, 'r', lw=2)
            ax[1,1].plot(epoch_list, v_loss_epoch_diff_append, 'b', lw=2)
            ax[1,1].set_title('diffusion loss')
            ax[1,2].plot(epoch_list, t_loss_epoch_bg_append, 'r', lw=2)
            ax[1,2].plot(epoch_list, v_loss_epoch_bg_append, 'b', lw=2)
            ax[1,2].set_title('bg loss')
            plt.tight_layout()
            plt.savefig(spt.path_saved_model + 'learning_curve.png')
            plt.close('all')
        print("(""epoch", epoch, ")", "Training Loss", t_loss_epoch, "Validation Loss", v_loss_epoch)
        epoch+=1
    end = time.time()
    print("...Done Training...")
    print("...Training takes %d s..." % (end - start))

    modelrecord.write('\n...Training for %d epoch...\nThe minimal validation loss is %s\n' % (
    epoch, min_v_loss))
    modelrecord.close()

if __name__ == '__main__':
    main()
