import subprocess
from tqdm import tqdm
import shlex

hyperparameters = {
    "lr": [.0001, .00001, .000001],
    "momentum": [0, .5, .95],
    "wd": [5e-5, 5e-4, 5e-3]
}

if __name__ == "__main__":

    train_methods = ['TEDn', 'PN'][-1:]

    for train_method in train_methods:
        # default settings
        lr = hyperparameters['lr'][1]
        momentum = hyperparameters['momentum'][1]
        wd = hyperparameters['wd'][1]
        cmd = f"python train_PU.py --lr={lr} --momentum={momentum} --wd={wd} --data-type='SemEval' --train-method={train_method} --net-type='microsoft/codebert-base' --epochs=10 --optimizer=AdamW --alpha=0 --beta=.6 --log-dir=logging_accuracy_hyperparam_testing_full"

        print(cmd)

        subprocess.run(shlex.split(cmd))