import subprocess
from tqdm import tqdm
import shlex

if __name__ == "__main__":

    train_methods = ['TEDn', 'PN'][-1:]

    for train_method in train_methods:

        cmd = f"python train_PU.py --lr=0.00001 --momentum=0 --data-type='SemEval' --train-method={train_method} --net-type='microsoft/codebert-base' --epochs=3 --optimizer=AdamW --alpha=0 --beta=.6 --log-dir=logging_accuracy_hyperparam_testing"

        print(cmd)

        subprocess.run(shlex.split(cmd))