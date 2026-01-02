import os
import time
import torch

from collections import OrderedDict

def save_weights(state_dict, work_dir, out_folder='weights'):
    weights = OrderedDict([
        [k.split('module.')[-1], v.cpu()]
        for k, v in state_dict.items()
    ])

    weights_name = f'weights.pt'
    save_states(weights, work_dir, out_folder, weights_name)


def save_states(states, work_dir, out_folder, out_name):
    out_folder_path = os.path.join(work_dir, out_folder)
    out_path = os.path.join(out_folder_path, out_name)
    os.makedirs(out_folder_path, exist_ok=True)
    torch.save(states, out_path)


def load_weights(model, weights_path, work_dir = "./output_dir/"):
    weights = torch.load(weights_path)
    weights = OrderedDict(
        [[k.split('module.')[-1],
          v.cuda()] for k, v in weights.items()])

    try:
        model.load_state_dict(weights, strict=False)
    except:
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        print_log('Can not find these weights:', work_dir)
        for d in diff:
            print_log('  ' + d)
        state.update(weights)
        model.load_state_dict(state, strict=False)

    return model

def print_log(s, work_dir, print_time=True, print_log_bool=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        s = f'[ {localtime} ] {s}'
    print(s)
    if print_log_bool:
        path = os.path.join(work_dir, 'log.txt')

        with open(path, 'a') as f:
            print(s, file=f)

def compute_duration(start, end):
    return end - start

### Based on timm's accuracy
def compute_batch_acc(output, target, maxk = (1,)):

    result = []

    for k in maxk:
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()

        true = pred.eq(target.view(1, -1).expand_as(pred))
        true_k = true[:k].reshape(-1).float().sum(0)
        result.append(true_k.mul(100.0/output.size(0)))

    return result
