import argparse
import os
from pathlib import Path
from typing import Mapping, Optional

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
from skimage.metrics import structural_similarity as ssim

# isort: off

# MLSDK modules
from mlsdk import (
    MNDevice,
    Context,
    storage,
    CacheOptions,
    TensorLike,
)

# following modules are from:
# [ai-reference-models repository](https://github.com/intel/ai-reference-models/tree/main/models_v2/pytorch/ssd-resnet34/inference/cpu)  # NOQA: B950
from infer import dboxes_R34_coco
from ssd_r34 import SSD_R34
from utils import COCODetection, Encoder, SSDTransformer

# isort: on

# No GUI output
matplotlib.use("Agg")


# label_num = 81 and strides = [3, 3, 2, 2, 2, 2] are from original source in:
# https://github.com/intel/ai-reference-models/tree/main/models_v2/pytorch/ssd-resnet34/inference/cpu/infer.py
def load_model(
    path: str | os.PathLike,
    label_num: int = 81,
    strides: tuple[int, ...] = (3, 3, 2, 2, 2, 2),
) -> torch.nn.Module:
    # Create class object
    model = SSD_R34(label_num, strides=strides)
    # Load pretrained model's parameters
    model.load_state_dict(
        torch.load(path, map_location=lambda storage, loc: storage)["model"]
    )
    assert isinstance(model, torch.nn.Module)
    return model


# Load an input image given as a path
def load_image(
    img_path: str | os.PathLike,
    img_size: tuple[int, int] = (1200, 1200),
) -> tuple[ImageFile.ImageFile, torch.Tensor]:
    # Resize input img for model
    orig_img = Image.open(img_path)

    # Convert img to tensor and change shape
    resized_img = orig_img.resize(img_size)
    converted_img_np = np.array(resized_img).transpose(
        2, 0, 1
    )  # (C,H,W), assuming dtype=uint8
    converted_img = (
        torch.from_numpy(converted_img_np).contiguous().unsqueeze(dim=0)
    )  # (1,C,H,W)

    return orig_img, converted_img


# Decode a model's output for an input image
def decode_outputs(
    out_locs: torch.Tensor,
    out_labels: torch.Tensor,
    encoder: Encoder,
    criteria: float = 0.50,
    max_output: int = 200,
    device: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    try:
        decoded_outputs = encoder.decode_batch(
            out_locs,
            out_labels,
            criteria=criteria,
            max_output=max_output,
            device=device,
        )
    except Exception as e:
        print(f"Error in decode_outputs: {type(e)} - {e}")
        return []

    results = []
    idx = 0
    for i in range(decoded_outputs[3].size(0)):
        detection_num = decoded_outputs[3][i].item()
        idx_range = idx + detection_num
        results.append(
            (
                decoded_outputs[0][idx:idx_range],
                decoded_outputs[1][idx:idx_range],
                decoded_outputs[2][idx:idx_range],
            )
        )
        idx += detection_num

    return results


# This function is originally from:
# [ai-reference-models](https://github.com/intel/ai-reference-models/blob/main/models_v2/pytorch/ssd-resnet34/inference/cpu/utils.py)  # NOQA: B950
# Drawing bboxes and labels for detected objects
def draw_patches(  # NOQA: CFQ002
    img: ImageFile.ImageFile,
    img_path: Path,
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    order: str = "ltrb",
    label_map: Mapping[int, str] | None = None,
    bbox_alpha: float = 1.0,
    bbox_linewidth: int = 1,
    label_alpha: float = 0.3,
    label_linewidth: int = 2,
    text_size: int = 10,
) -> None:
    img_np = np.array(img)
    labels_np = labels.detach().numpy()
    bboxes_np = bboxes.detach().numpy()

    # From labels_np (ndarray[int]) to labels_list (list[str])
    if label_map is not None:
        labels_list = [
            label_map.get(int(label_i), "(no label)") for label_i in labels_np
        ]
    else:
        # Use an int value as a label str
        labels_list = [str(int(label_i)) for label_i in labels_np]

    if order == "ltrb":
        xmin, ymin, xmax, ymax = (
            bboxes_np[:, 0],
            bboxes_np[:, 1],
            bboxes_np[:, 2],
            bboxes_np[:, 3],
        )
        cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    else:
        cx, cy, w, h = (
            bboxes_np[:, 0],
            bboxes_np[:, 1],
            bboxes_np[:, 2],
            bboxes_np[:, 3],
        )

    htot, wtot, _ = img_np.shape
    cx *= wtot
    cy *= htot
    w *= wtot
    h *= htot

    plt.imshow(img_np)
    ax = plt.gca()
    for cx_i, cy_i, w_i, h_i, label_i in zip(cx, cy, w, h, labels_list):
        if label_i == "background":
            continue
        ax.add_patch(
            patches.Rectangle(
                (cx_i - 0.5 * w_i, cy_i - 0.5 * h_i),
                w_i,
                h_i,
                fill=False,
                color="r",
                alpha=bbox_alpha,
                linewidth=bbox_linewidth,
            )
        )
        bbox_props = dict(
            boxstyle="square,pad=0",
            fc="y",
            ec="y",
            alpha=label_alpha,
            linewidth=label_linewidth,
        )
        ax.text(
            cx_i - 0.5 * w_i,
            cy_i - 0.5 * h_i,
            label_i,
            ha="left",
            va="bottom",
            size=text_size,
            bbox=bbox_props,
        )

    plt.savefig(img_path)
    plt.clf()
    plt.close()


# Fetch results satisfying threshold and draw bounding box on the given input image
def draw_detection_result(  # NOQA: CFQ002
    out_locs: torch.Tensor,
    out_labels: torch.Tensor,
    img: ImageFile.ImageFile,
    img_path: Path,
    encoder: Encoder,
    threshold: float,
    label_info: dict[int, str],
) -> None:

    decoded_output = decode_outputs(out_locs, out_labels, encoder)
    if not decoded_output:
        print("no objects have been detected")
        return

    bboxes, labels, scores = decoded_output[0]
    detection_mask = scores > threshold
    fetched_bboxes = bboxes[detection_mask]
    fetched_labels = labels[detection_mask]
    draw_patches(img, img_path, fetched_bboxes, fetched_labels, label_map=label_info)


def calc_ssim(
    lhs_img_path: Path,
    rhs_img_path: Path,
) -> float:
    lhs_img = cv2.imread(lhs_img_path)
    rhs_img = cv2.imread(rhs_img_path)
    assert lhs_img.shape == rhs_img.shape

    # Convert to gray scale (only image structure is necessary for SSIM)
    lhs_img_gray = cv2.cvtColor(lhs_img, cv2.COLOR_BGR2GRAY)
    rhs_img_gray = cv2.cvtColor(rhs_img, cv2.COLOR_BGR2GRAY)

    return float(ssim(lhs_img_gray, rhs_img_gray))  # type: ignore[no-untyped-call]


def run_infer(  # NOQA: CFQ002
    *,
    img_path: Path,
    model_path: Path,
    coco_data_path: Path,
    coco_annotation_path: Path,
    out_img_dir: Path,
    threshold: float,
    device_name: str,
    outdir: str,
    option_json_path: Optional[Path] = None,
) -> None:
    # Create Dataset object
    IMG_SIZE = [1200, 1200]
    STRIDES = [3, 3, 2, 2, 2, 2]
    default_boxes = dboxes_R34_coco(IMG_SIZE, STRIDES)
    transformer = SSDTransformer(default_boxes, tuple(IMG_SIZE), val=True)
    coco = COCODetection(coco_data_path, coco_annotation_path, transformer)

    # Create encoder for decode process
    encoder = Encoder(default_boxes)

    # Load and prepare images
    orig_img, converted_img = load_image(img_path)
    infer_input = {"image": converted_img}
    img_name = img_path.name

    # Create pretrained model object
    model = load_model(model_path)
    model.eval()

    # Inference function for Context.compile
    def infer_fn(sample: dict[str, TensorLike]) -> dict[str, TensorLike]:
        with torch.no_grad():
            locs, labels = model(sample["image"].float() / 255.0)
        return {"locs": locs, "labels": labels}

    device = MNDevice(device_name)
    context = Context(device)
    Context.switch_context(context)

    context.registry.register("model", model)

    compile_options = {}
    if option_json_path is not None:
        compile_options = {"option_json": str(option_json_path)}

    compiled_infer_fn = context.compile(
        infer_fn,
        infer_input,
        storage.path(outdir),
        options=compile_options,
        cache_options=CacheOptions(outdir + "/cache"),
    )

    out_mncore = compiled_infer_fn(infer_input)
    out_locs_mncore, out_labels_mncore = (
        out_mncore["locs"].cpu(),
        out_mncore["labels"].cpu(),
    )

    mncore_img_path = out_img_dir / ("out_mncore_" + img_name)
    print(f"Drawing detection result => {mncore_img_path}")
    draw_detection_result(
        out_locs_mncore,
        out_labels_mncore,
        orig_img,
        mncore_img_path,
        encoder,
        threshold,
        label_info=coco.label_info,
    )

    out_torch = infer_fn(infer_input)
    out_locs_torch, out_labels_torch = (
        out_torch["locs"].cpu(),
        out_torch["labels"].cpu(),
    )

    torch_img_path = out_img_dir / ("out_torch_" + img_name)
    print(f"Drawing detection result => {torch_img_path}")
    draw_detection_result(
        out_locs_torch,
        out_labels_torch,
        orig_img,
        torch_img_path,
        encoder,
        threshold,
        label_info=coco.label_info,
    )

    score = calc_ssim(mncore_img_path, torch_img_path)
    print(f"SSIM score: {score}")
    assert score > 0.99, "Generated images differ."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SSD-ResNet34-1200 model for input image"
    )
    parser.add_argument("img_path", type=Path, help="Path to input image")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained SSD model (e.g. resnet34-ssd1200.pth)",
    )
    parser.add_argument(
        "--coco_data_path", type=Path, required=True, help="Path to the COCO dataset"
    )
    parser.add_argument(
        "--coco_annotation_path",
        type=Path,
        required=True,
        help="Path to annotation data (JSON) corresponding to the COCO dataset",
    )
    parser.add_argument(
        "--out_img_dir",
        type=Path,
        default=Path("."),
        help="Path to directory to output detection result images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="""
        Detection threshold (0.0-1.0), a smaller threshold make object detecting
        more sensitive""",
    )
    parser.add_argument("--device", type=str, default="mncore2:auto")
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_ssd_inference/out")
    parser.add_argument(
        "--option_json",
        type=Path,
        default="/opt/pfn/pfcomp/codegen/preset_options/O1.json",
    )
    args = parser.parse_args()

    # Validate arguments
    assert 0 <= args.threshold and args.threshold <= 1.0
    assert not args.out_img_dir.is_file(), "Please specify directory path."

    args.out_img_dir.mkdir(parents=True, exist_ok=True)

    run_infer(
        img_path=args.img_path,
        model_path=args.model_path,
        coco_data_path=args.coco_data_path,
        coco_annotation_path=args.coco_annotation_path,
        out_img_dir=args.out_img_dir,
        threshold=args.threshold,
        device_name=args.device,
        outdir=args.outdir,
        option_json_path=args.option_json,
    )


if __name__ == "__main__":
    main()
