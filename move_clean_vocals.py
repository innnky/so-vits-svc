import os
import shutil
from pathlib import Path
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, help="Root directory of ouput and clean_vocals folder.")
    args = parser.parse_args()
    path = args.path
    if path is None:
        print("Please specify --path as the root directory")
        exit(0)

    output_path = os.path.join(path, "output")
    if not os.path.isdir(output_path):
        print("{} does not exist!".format(output.path))

    clean_path = os.path.join(path, "clean_vocal")
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)

    for root, dirs, files in os.walk(output_path):
        for file in files:
            path = Path(os.path.join(root, file))
            stem, ext = os.path.splitext(os.path.join(root, file))
            if path.stem == "vocals":
                src = os.path.join(root, file)
                parent_folder = root.split(os.sep)[-1]
                dst = os.path.join(clean_path, parent_folder + ext)
                shutil.copy(src, dst)
                print("src: {}, target: {}".format(src, dst))
