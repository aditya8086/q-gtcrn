import os, time, csv, random, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from dataset import get_dataloaders
from q_gtcrn import QGTCRN
from loss import HybridLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def noisy_spec_from_feat(noisy_feat):
    # noisy_feat channels are [mag, real, imag]; the raw spectrum is (real, imag).
    return torch.stack([noisy_feat[:, 1], noisy_feat[:, 2]], dim=-1)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total = 0.0
    with torch.set_grad_enabled(training):
        for noisy_feat, clean_spec, _ in loader:
            noisy_feat = noisy_feat.to(device)
            clean_spec = clean_spec.to(device)
            enhanced = model(noisy_feat, noisy_spec_from_feat(noisy_feat))
            loss = loss_fn(enhanced, clean_spec)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def save_curve(history, path):
    plt.figure(figsize=(7, 4))
    plt.plot(history['epoch'], history['train'], label='train')
    plt.plot(history['epoch'], history['val'],   label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',        default='VoiceBank-DEMAND')
    p.add_argument('--epochs',      type=int,   default=180)   # <-- was 100
    p.add_argument('--batch-size',  type=int,   default=4)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--weight-decay',type=float, default=0.01)  # <-- new, AdamW
    p.add_argument('--warmup-epochs', type=int, default=10)    # <-- new, ~5-6% of 180
    p.add_argument('--eta-min',     type=float, default=1e-6)  # <-- new, cosine floor
    p.add_argument('--workers',     type=int,   default=0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--ckpt-dir',    default='checkpoints_qgtcrn_recipeA')
    p.add_argument('--results-dir', default='results_qgtcrn_recipeA')
    p.add_argument('--resume',      default=None)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=args.workers)

    model = QGTCRN().to(device)          # architecture UNCHANGED -- same dual-axis + conv-TRA
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    loss_fn = HybridLoss().to(device)

    # --- Recipe fix: AdamW + linear warmup -> cosine annealing ---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                            total_iters=args.warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                                     eta_min=args.eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                             milestones=[args.warmup_epochs])

    log_path   = os.path.join(args.results_dir, 'train_log.csv')
    curve_path = os.path.join(args.results_dir, 'loss_curve.png')
    history    = {'epoch': [], 'train': [], 'val': []}

    start_epoch, best_val = 1, float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt['best_val']
        print(f"Resumed from epoch {ckpt['epoch']}")
        if os.path.exists(log_path):
            with open(log_path) as f:
                for row in csv.DictReader(f):
                    if int(row['epoch']) < start_epoch:
                        history['epoch'].append(int(row['epoch']))
                        history['train'].append(float(row['train_loss']))
                        history['val'].append(float(row['val_loss']))

    if not args.resume or not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer)
        vl = run_epoch(model, val_loader,   loss_fn, device)
        scheduler.step()          # <-- epoch-based, no val-loss argument (not plateau-based)
        lr = optimizer.param_groups[0]['lr']
        print(f"epoch {epoch:3d} | train {tr:.4f} | val {vl:.4f} "
              f"| lr {lr:.2e} | {time.time()-t0:.0f}s")

        history['epoch'].append(epoch)
        history['train'].append(tr)
        history['val'].append(vl)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{tr:.6f}", f"{vl:.6f}", f"{lr:.2e}"])
        save_curve(history, curve_path)

        state = {'epoch': epoch, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'best_val':  best_val}
        torch.save(state, os.path.join(args.ckpt_dir, 'last.tar'))
        if vl < best_val:
            best_val = vl
            state['best_val'] = best_val
            torch.save(state, os.path.join(args.ckpt_dir, 'best.tar'))
            print(f"  saved best (val {vl:.4f})")


if __name__ == '__main__':
    main()