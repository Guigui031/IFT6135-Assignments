import json
import matplotlib.pyplot as plt
import numpy as np

# Load all experiment logs
logs = {}
for i in range(1, 9):
    with open(f'../assignment/log/{i}/args.json', 'r') as f:
        logs[i] = json.load(f)

def plot_single(eid, filename):
    """Plot train loss, val loss, val accuracy for one experiment in a single figure."""
    log = logs[eid]
    desc = log['description']
    train_iters = np.arange(len(log['train_loss_logfreq']))
    eval_iters = np.arange(len(log['eval_acc_logfreq']))
    best_val = max(log['eval_acc_logfreq'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.subplots_adjust(wspace=0.35)

    # Left: train loss and val loss
    ax1.plot(train_iters, log['train_loss_logfreq'], color='C0', linewidth=0.8, label='Train Loss')
    ax1.plot(eval_iters, log['eval_loss_logfreq'], color='C1', linewidth=0.8, label='Val Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Training Iteration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: val accuracy
    ax2.plot(eval_iters, log['eval_acc_logfreq'], color='C2', linewidth=0.8, label='Val Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Training Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {filename}')

# Generate one plot per experiment
for i in range(1, 9):
    plot_single(i, f'exp{i}.png')

print('Done!')
