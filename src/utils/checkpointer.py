import torch
import os

class CheckpointManager:
    def __init__(self, save_dir, save_best_only=False, mode='min'):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        os.makedirs(save_dir, exist_ok=True)

    def save(self, epoch, model, metric):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metric': metric,
        }

        if not self.save_best_only:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch:03d}.pth')
            torch.save(checkpoint, path)

        is_best = (self.mode == 'min' and metric < self.best_metric) or \
                  (self.mode == 'max' and metric > self.best_metric)

        if is_best:
            self.best_metric = metric
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
