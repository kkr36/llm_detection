import subprocess
from tqdm import tqdm
import shlex

hyperparameters = {
    "lr": [.0001, .00001, .000001, .000005],
    "momentum": [0, .5, .95],
    "wd": [0, 5e-5, 5e-4, 5e-3]
}

seeds = [0, 42, 100, 410, 1000, 155, 829, 9901, 284, 777][:1]

if __name__ == "__main__":

    train_methods = ['TEDn', 'PN'][:1]
    pretrain_paths = [None]

    for train_method in train_methods:
        for pretrain_path in pretrain_paths:
            for seed in seeds:
                # default settings
                lr = hyperparameters['lr'][1]
                momentum = hyperparameters['momentum'][0]
                wd = hyperparameters['wd'][-2]
                cmd = f"python ft_PU.py --lr={lr} --momentum={momentum} --wd={wd} --data-type='SemEval' --train-method={train_method} --net-type='microsoft/codebert-base' --epochs=30 --optimizer=AdamW --alpha={seed} --beta=.6 --log-dir=logging_accuracy_double_cheating_id/{pretrain_path}/{seed}"

                print(cmd)

                subprocess.run(shlex.split(cmd))