import subprocess
from tqdm import tqdm
import shlex
from sklearn.model_selection import ParameterSampler

hyperparameters = {
    "lr": [.0001, .00001, .000005, .000001],
    "momentum": [0, .5, .95],
    "wd": [0, 5e-5, 5e-4, 5e-3]
}

sampler = ParameterSampler(hyperparameters, n_iter=5, random_state=42)
param_list = list(sampler)

special = {"lr": .00001, "momentum": 0., "wd": 5e-4}

if special not in param_list:
    param_list = [special] + param_list

param_list = param_list[:1]
print(param_list)

if __name__ == "__main__":

    train_methods = ['TEDn', 'PN'][:1]
    pretrain_paths = [None]

    for params in param_list:
        for pretrain_path in pretrain_paths:
            for train_method in train_methods:
                # default settings
                # lr = hyperparameters['lr'][1]
                # momentum = hyperparameters['momentum'][0]
                # wd = hyperparameters['wd'][-2]
                lr, momentum, wd = params["lr"], params["momentum"], params["wd"]
                cmd = f"python ft_PU.py --lr={lr} --momentum={momentum} --wd={wd} --data-type='SemEval' --train-method={train_method} --net-type='microsoft/codebert-base' --epochs=15 --optimizer=AdamW --alpha=0 --beta=.6 --log-dir=logging_accuracy_test_on_train_id/{pretrain_path}/{train_method}"

                print(cmd)

                subprocess.run(shlex.split(cmd))