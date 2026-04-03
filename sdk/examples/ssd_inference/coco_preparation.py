import argparse
import json
import os
from typing import Any


def fetch_imgs_and_ids(
    json_obj: dict[str, Any], license_id_list: list[int]
) -> tuple[list[list[dict[str, int | str]]], list[int]]:
    img_list = []
    id_list = []
    for i in json_obj["images"]:
        if i["license"] in license_id_list:
            img_list.append(i)
            id_list.append(i["id"])

    return img_list, id_list


def fetch_annotations(json_obj: dict[str, Any], id_list: list[int]) -> list[Any]:
    ann_list = []
    for i in json_obj["annotations"]:
        if i["image_id"] in id_list:
            ann_list.append(i)

    return ann_list


def remove_wasted_imgs(
    json_obj: dict[str, Any], fetched_dict: dict[str, Any], img_dir: str | os.PathLike
) -> None:
    org_file_names = set([i["file_name"] for i in json_obj["images"]])
    fetched_file_name = set([i["file_name"] for i in fetched_dict["images"]])
    diff_set = org_file_names - fetched_file_name
    for i in diff_set:
        wasted_img_path = os.path.join(img_dir, i)
        if os.path.exists(wasted_img_path):
            os.remove(wasted_img_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--annotation_file",
        type=str,
        default="./coco/annotations/instances_val2017.json",
    )
    parser.add_argument("-i", "--images_dir", type=str, default="./coco/val2017/")
    parser.add_argument(
        "-f",
        "--fetched_annotation_file",
        type=str,
        default="./coco/annotations/fetched_annotations.json",
    )
    args = parser.parse_args()

    # fetch images' infomation with no license problems
    json_obj = None
    with open(args.annotation_file, mode="r") as f:
        json_obj = json.load(f)
    license_id_list = [4]  # 4: CC-BY 2.0

    img_list, id_list = fetch_imgs_and_ids(json_obj, license_id_list)
    ann_list = fetch_annotations(json_obj, id_list)
    fetched_dict = (
        {"info": json_obj["info"]}
        | {"licenses": json_obj["licenses"]}
        | {"images": img_list}
        | {"annotations": ann_list}
        | {"categories": json_obj["categories"]}
    )

    # dump fetched infomations
    with open(args.fetched_annotation_file, mode="w") as f:
        json.dump(fetched_dict, f)

    # remove files unused in inference
    remove_wasted_imgs(json_obj, fetched_dict, args.images_dir)


if __name__ == "__main__":
    main()
