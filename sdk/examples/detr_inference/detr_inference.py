import argparse
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from detr_eval import DETRSegmToBBoxAttn, MaskHeadPart, run_eval
from mlsdk import Context, MNDevice
from PIL import Image
from utility import apply_toml_defaults


def prepare_task_components(args: argparse.Namespace) -> dict[str, Any]:
    task_components = {}

    # Create model and post processor objs
    task_components["model"], task_components["postprocessors"] = torch.hub.load(
        "facebookresearch/detr",
        "detr_resnet50_panoptic",
        pretrained=True,
        return_postprocessor=True,
    )

    # standard PyTorch mean-std input image normalization
    transform = T.Compose(
        [
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    im = Image.open(args.img_path)
    task_components["orig_target_size"] = torch.as_tensor(
        T.functional.to_tensor(im).shape[-2:]
    ).unsqueeze(0)
    task_components["image"] = transform(im).unsqueeze(0)

    return task_components


def main(args: argparse.Namespace) -> None:
    # Pass device info to the Context obj
    device = MNDevice(args.device_name)
    context = Context(device)
    Context.switch_context(context)

    task_components = prepare_task_components(args)
    task_components["mask_head"] = MaskHeadPart(
        task_components["model"], num_split=args.num_split
    )
    task_components["model"] = DETRSegmToBBoxAttn(task_components["model"])

    run_eval(args, task_components, context)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("img_path", type=Path, help="Path to input image")
    parser.add_argument("--device", type=str, default="mncore2:auto")
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_detr_inference/out")
    parser.add_argument(
        "--option_json",
        type=Path,
        default="/opt/pfn/pfcomp/codegen/preset_options/O1.json",
    )

    # load configs from toml file
    apply_toml_defaults(Path(__file__).resolve().parent / "configs.toml", parser)

    args = parser.parse_args()

    # Set "device" attribute for detr module
    args.device, args.device_name = "cpu", args.device

    main(args)
