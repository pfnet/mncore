import argparse
import csv
import os
import shutil
import subprocess
import sys

from tqdm import tqdm


def check_not_target(dir: str, target_folders: list[str]) -> bool:

    for target_folder in target_folders:
        # Compare target_folder name and input dir name
        if target_folder in dir:
            return False

    return True


def rename_folders(dataset_dir: str, target_folders: list[str] | None = None) -> None:
    # obtain image folders' name
    dir_list = os.listdir(dataset_dir)

    print("renaming directories' name for dataset...")
    # rename the folders' name ( remove initial numbers and `-101` term )
    for dir in tqdm(dir_list):
        if target_folders is not None and check_not_target(dir, target_folders):
            result = subprocess.run(["rm", "-rf", os.path.join(dataset_dir, dir)])
            if result.returncode != 0:
                print(f"Error deleting {dir}")
                sys.exit(1)
            continue
        new_dir_name = dir.split(".")[-1]
        if "101" in new_dir_name:
            new_dir_name = new_dir_name[0 : len(new_dir_name) - 4]
        shutil.move(
            os.path.join(dataset_dir, dir), os.path.join(dataset_dir, new_dir_name)
        )
    print("renaming done!")


def rename_imgs(dataset_dir: str) -> None:
    # obtain image folders' name
    dir_list = os.listdir(dataset_dir)

    print("renaming imgs' name for dataset...")
    for dir in tqdm(dir_list):
        img_dir_path = os.path.join(dataset_dir, dir)
        img_list = os.listdir(img_dir_path)
        new_img = ""
        for img in img_list:
            if "_" in img:
                new_img = img[4:]
            else:
                result = subprocess.run(["rm", "-rf", os.path.join(img_dir_path, img)])
                if result.returncode != 0:
                    print(f"Error deleteing {img}")
                    sys.exit(1)
                continue
            result = subprocess.run(
                [
                    "mv",
                    "-f",
                    os.path.join(img_dir_path, img),
                    os.path.join(img_dir_path, new_img),
                ]
            )
            if result.returncode != 0:
                print(f"Error moving {img}")
                sys.exit(1)
    print("renaming done!")


def add_metadata(dataset_dir: str) -> None:
    # obtain image folders' name
    dir_list = os.listdir(dataset_dir)

    print("creating metadata...")
    header = ["file_name", "caption"]
    for dir in tqdm(dir_list):
        img_dir_path = os.path.join(dataset_dir, dir)
        img_list = os.listdir(img_dir_path)
        if "metadata.csv" in img_list:
            print("'metadata.csv' already exist in", dir)
            continue

        with open(img_dir_path + "/metadata.csv", mode="w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
            for img in img_list:
                csv_writer.writerow([img, dir])
    print("creating metadata done!")


def main(args: argparse.Namespace) -> None:
    rename_folders(args.dataset_dir, args.target_folders)
    rename_imgs(args.dataset_dir)
    add_metadata(args.dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./data/train")
    parser.add_argument(
        "--target_folders", type=str, action="extend", nargs="+", default=None
    )

    args = parser.parse_args()
    main(args)
