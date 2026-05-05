import argparse
import os
import shutil

KEEP_FILES = {
    "results_0_oss_test_50.csv",
    "results_0_oss_val_15.csv",
    "task_prompt.txt",
}

def clean_folder(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(base_dir, folder_name)

    if not os.path.isdir(target):
        print(f"Error: '{target}' is not a directory.")
        return

    removed = []
    for entry in os.listdir(target):
        entry_path = os.path.join(target, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
            removed.append(entry + "/")
        elif entry.endswith(".py") or entry in KEEP_FILES:
            continue
        else:
            os.remove(entry_path)
            removed.append(entry)

    if removed:
        print(f"Removed {len(removed)} item(s):")
        for name in removed:
            print(f"  {name}")
    else:
        print("Nothing to remove.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove non-essential files from a folder.")
    parser.add_argument("folder", help="Folder name (relative to this script's directory)")
    args = parser.parse_args()
    confirm = input(f"Clean folder '{args.folder}'? This will delete files. Type 'yes' to continue: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
    else:
        clean_folder(args.folder)
