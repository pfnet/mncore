import io
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from mlsdk import CompiledFunction, Context
from models.segmentation import DETRsegm
from panopticapi.utils import rgb2id
from PIL import Image
from util.misc import nested_tensor_from_tensor_list
from utility import compile_fn


class DETRSegmToBBoxAttn(torch.nn.Module):
    def __init__(self, detr_segm: DETRsegm) -> None:
        super().__init__()
        self.detr = detr_segm.detr
        self.bbox_attention = detr_segm.bbox_attention

    def forward(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}

        sample = nested_tensor_from_tensor_list(sample)
        features, pos = self.detr.backbone(sample)

        src, mask = features[-1].decompose()
        src_proj = self.detr.input_proj(src)
        hs, memory = self.detr.transformer(
            src_proj, mask, self.detr.query_embed.weight, pos[-1]
        )

        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()

        # FIXME h_boxes takes the last one computed, keep this in mind
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        out.update(
            feat2=features[2].tensors,
            feat1=features[1].tensors,
            feat0=features[0].tensors,
            src_proj=src_proj,
            pred_logits=outputs_class[-1],
            pred_boxes=outputs_coord[-1],
            bbox_mask=bbox_mask,
        )

        return out


class MaskHeadPart(torch.nn.Module):
    def __init__(self, detr_segm: DETRsegm, num_split: int = 100) -> None:
        super().__init__()

        self.mask_head = detr_segm.mask_head
        self.num_queries = detr_segm.detr.num_queries  # 100
        self.num_split = num_split

    def forward(
        self, x: torch.Tensor, bbox_mask: torch.Tensor, fpns: list[torch.Tensor]
    ) -> torch.Tensor:
        seg_mask = self.mask_head(x, bbox_mask, fpns)
        output_seg_mask = seg_mask.view(
            x.shape[0],
            self.num_queries // self.num_split,
            seg_mask.shape[-2],
            seg_mask.shape[-1],
        )

        return output_seg_mask


def visualize_prediction(
    args: Namespace,
    task_components: dict[str, Any],
    outputs: dict[str, torch.Tensor],
) -> None:

    image = numpy.array(Image.open(args.img_path))[:, :, ::-1]
    result = task_components["postprocessors"](outputs, [image.shape[:2]])[0]

    # Panoptic predictions are stored in a special format png
    panoptic_seg = Image.open(io.BytesIO(result["png_string"]))

    # We convert the png into an segment id map
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

    # Detectron2 uses a different numbering of coco classes,
    # here we convert the class ids accordingly
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    for info in result["segments_info"]:
        c = info["category_id"]
        if info["isthing"]:
            info["category_id"] = meta.thing_dataset_id_to_contiguous_id[c]
        else:
            info["category_id"] = meta.stuff_dataset_id_to_contiguous_id[c]

    # Finally we visualize the prediction
    v = Visualizer(image, meta)
    v._default_font_size = 20
    v = v.draw_panoptic_seg_predictions(panoptic_seg, result["segments_info"])
    v.save(os.path.join(args.outdir, f"{Path(args.img_path).stem}.png"))


def compile_eval_fn(
    args: Namespace,
    task_components: dict[str, Any],
    context: Context,
) -> tuple[CompiledFunction, CompiledFunction, dict[str, torch.Tensor]]:

    # in case using a mncore2 backend, the DETR model is separated
    # into two halves to avoid the LM oom and the reshape errors:
    # the first half calculates the bbox_masks,
    # and the second half calculates the remaining MaskHead part
    def eval_to_bbox_or_full(
        sample: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:  # eval_all_or_bbox
        with torch.no_grad():
            return task_components["model"](sample["image"])

    def eval_mask_head(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            pred_mask = task_components["mask_head"](
                sample["src_proj"],
                sample["bbox_mask"],
                [sample["feat2"], sample["feat1"], sample["feat0"]],
            )

        return {"pred_mask": pred_mask}

    eval_fn = eval_to_bbox_or_full
    mask_fn = eval_mask_head

    # For compilation for MLSDK/MN-Core 2
    sample = {"image": task_components["image"]}

    eval_fn = compile_fn(
        context,
        eval_fn,
        task_components["model"],
        sample,
        os.path.join(args.outdir, "to_bbox"),
        model_name="detr_to_bbox",
        option_json=args.option_json,
    )

    sample.update(
        src_proj=torch.randn(1, 256, 25, 25),
        bbox_mask=torch.rand(
            1, task_components["mask_head"].num_queries // args.num_split, 8, 25, 25
        ),  # originally, 1, 100, 8, 25, 25
        feat2=torch.randn(1, 1024, 50, 50),
        feat1=torch.randn(1, 512, 100, 100),
        feat0=torch.randn(1, 256, 200, 200),
    )

    mask_fn = compile_fn(
        context,
        mask_fn,
        task_components["mask_head"],
        sample,
        os.path.join(args.outdir, "mask_head"),
        model_name="detr_mask_head",
        option_json=args.option_json,
    )

    return eval_fn, mask_fn


@torch.no_grad()
def evaluate(
    args: Namespace,
    eval_fn: CompiledFunction,
    mask_fn: CompiledFunction,
    task_components: dict[str, Any],
) -> dict[str, torch.Tensor]:

    sample = {"image": task_components["image"]}

    outputs = eval_fn(sample)

    num_queries = outputs["bbox_mask"].shape[1]
    bbox_masks = outputs["bbox_mask"].split(num_queries // args.num_split, dim=1)
    sample.update(
        src_proj=outputs["src_proj"],
        feat2=outputs["feat2"],
        feat1=outputs["feat1"],
        feat0=outputs["feat0"],
    )

    pred_masks = []
    for bbox_mask in bbox_masks:
        sample.update(bbox_mask=bbox_mask)
        pred_masks.append(mask_fn(sample)["pred_mask"].cpu())

    pred_masks = torch.cat(pred_masks, dim=0)
    outputs = {
        "pred_logits": outputs["pred_logits"],
        "pred_boxes": outputs["pred_boxes"],
        "pred_masks": pred_masks.view(
            1, num_queries, pred_masks.shape[-2], pred_masks.shape[-1]
        ),
    }

    return outputs


def run_eval(
    args: Namespace, task_components: dict[str, Any], context: Context
) -> None:

    task_components["model"].eval()
    task_components["mask_head"].eval()
    task_components["postprocessors"].eval()

    eval_fn, mask_fn = compile_eval_fn(args, task_components, context)
    outputs = evaluate(args, eval_fn, mask_fn, task_components)
    visualize_prediction(args, task_components, outputs)
