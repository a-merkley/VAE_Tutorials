import csv
import os
import torch
import numpy as np


def check_dir_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def get_run_id(dir):
    runs = os.listdir(dir)
    if len(runs) > 0:
        ids = [int(run.split("_")[1]) for run in runs]
        next_id = max(ids) + 1
    else:
        next_id = 0
    run_dir = "run_" + str(next_id).zfill(2)
    return next_id, run_dir

def batch_mean(val):
    # val is a list of tensors of variables sizes. If shape is 2D, first dim assumed to be batchsize
    val_batched = []
    for v in val:
        if isinstance(v, int):
            val_batched.append(v)
        elif isinstance(v, list):
            batched_lst = []
            for i in range(len(v)):
                batched_lst.append(v[i].mean(0).detach().numpy())
            val_batched.append(batched_lst)
        elif len(v.shape) > 1:
            val_batched.append(v.mean(0))
        else:
            val_batched.append(v)
    return val_batched


class Logger:
    def __init__(self, root):
        # Create all (sub)directories for saving
        check_dir_exists(root)
        run_id, run_dir = get_run_id(root)
        self.dir = os.path.join(root, run_dir)
        os.mkdir(self.dir)
        # Initialize save structures
        self.model = {}
        self.params = {}
        self.loss = {}
        self.metrics = {}

    @staticmethod
    def record(dict_obj, obj, header, batch_flag):
        # Obj is a list of the values, header is a list of descriptions
        if batch_flag:
            obj = batch_mean(obj)
        for i, h in enumerate(header):
            o = obj[i]
            if isinstance(o, torch.Tensor):
                o = o.detach().cpu().numpy()
            if h not in dict_obj:
                dict_obj[h] = [o]
            else:
                dict_obj[h].append(o)

    def record_params(self, obj, header):
        self.record(self.params, obj, header, batch_flag=True)

    def record_loss(self, obj, header):
        self.record(self.loss, obj, header, batch_flag=False)

    def record_metrics(self, obj, header):
        self.record(self.metrics, obj, header, batch_flag=False)

    def record_model(self, d):
        self.model = d

    def save(self):
        # Save parameters per iteration (I assume all parameters are vectors, not matrices)
        for key, val in self.params.items():
            self.params[key] = np.array(val)
        np.savez(os.path.join(self.dir, 'params.npz'), **self.params)

        # Save metrics per iteration
        for key, val in self.metrics.items():
            self.metrics[key] = np.array(val)
        np.savez(os.path.join(self.dir, 'metrics.npz'), **self.metrics)

        # Save loss per iteration
        np.savez(os.path.join(self.dir, 'loss.npz'), **self.loss)

        # Save model settings
        with open(os.path.join(self.dir, 'model.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.model.items():
                writer.writerow([key, value])
