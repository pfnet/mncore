import argparse
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from datasets import load_dataset
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from mlsdk import CompiledFunction, Context, MNCoreLRScheduler, MNDevice
from stable_diffusion_components import StableDiffusionComponents
from stable_diffusion_eval import StableDiffusionMNCorePipeline
from stable_diffusion_utility import (
    Timer,
    apply_toml_defaults,
    create_optimizer,
    decide_outdir,
    output_result_times,
    set_deterministic_mode,
)
from torchvision import transforms
from transformers import CLIPTokenizer


def create_train_dataloader(  # noqa: CFQ004
    args: argparse.Namespace,
    tokenizer: CLIPTokenizer,
) -> torch.utils.data.DataLoader:

    # Create dataloader
    dataset = load_dataset(
        "imagefolder",
        data_dir=args.data_dir,
        cache_dir=args.data_cache_dir if args.data_cache_dir != "" else None,
    )

    column_names = dataset["train"].column_names
    # dataset_columns = None  # Unused variable, can be removed

    image_column = column_names[0]
    caption_column = column_names[1]

    # originally from [diffusers example](https://github.com/huggingface/diffusers/blob/464374fb87610c53b2cf81e08d3df628fada3ce4/examples/text_to_image/train_text_to_image.py)  # noqa: B950
    def tokenize_captions(
        samples: dict[str, Any], is_train: bool = True
    ) -> torch.Tensor:
        captions = []
        for caption in samples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` "
                    "should contain either strings or lists of strings."
                )

        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs.input_ids

    # originally from [diffusers example](https://github.com/huggingface/diffusers/blob/464374fb87610c53b2cf81e08d3df628fada3ce4/examples/text_to_image/train_text_to_image.py)  # noqa: B950
    def preprocess_train(samples: dict[str, Any]) -> dict[str, Any]:
        images = [image.convert("RGB") for image in samples[image_column]]
        samples["pixel_values"] = [train_transforms(image) for image in images]
        samples["input_ids"] = tokenize_captions(samples)

        return samples

    # originally from [diffusers example](https://github.com/huggingface/diffusers/blob/464374fb87610c53b2cf81e08d3df628fada3ce4/examples/text_to_image/train_text_to_image.py)  # noqa: B950
    def collate_fn(samples: dict[str, Any]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([sample["input_ids"] for sample in samples])

        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # Create training dataset object
    train_dataset = dataset["train"].with_transform(preprocess_train)

    img_hw = (args.height, args.width)
    # originally from [diffusers example](https://github.com/huggingface/diffusers/blob/464374fb87610c53b2cf81e08d3df628fada3ce4/examples/text_to_image/train_text_to_image.py)  # noqa: B950
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                img_hw, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomCrop(img_hw),
            transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Create dataloader object
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=2,
    )

    return dataloader


def create_sample_inputs(  # noqa: CFQ002
    sample: dict[str, torch.Tensor],
    text_encoder_fn: Callable,
    vae_encoder_fn: Callable,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDIMScheduler,
    device: str | torch.device | MNDevice,
    batch: dict[str, torch.Tensor] | None = None,
    train_dataloader: torch.utils.data.DataLoader | None = None,
) -> dict[str, torch.Tensor]:
    if batch is None:
        if train_dataloader is None:
            print("Failed to get batch from dataloder: given dataloader is None!")
            return
        batch = next(train_dataloader.__iter__())

    # The given batch should have two keys, "pixel_values" and "input_ids"
    sample["pixel_values"] = batch["pixel_values"].to(
        "cuda" if device == "cuda" else "cpu"
    )
    sample["input_ids"] = batch["input_ids"].to("cuda" if device == "cuda" else "cpu")
    # Create position_ids for text_encoder_fn
    sample["position_ids"] = (
        torch.arange(tokenizer.model_max_length)
        .expand((1, -1))
        .to("cuda" if device == "cuda" else "cpu")
    )
    # Create encoder_hidden_state, latents, timesteps, and target_noise for unet_fn
    sample["encoder_hidden_states"] = text_encoder_fn(sample)["embeddings"]
    latents = vae_encoder_fn(sample)["latents"]
    # If the device is not cuda, latents may be `TensorProxy`, which do not support
    # `.device` attribute. Calling `.cpu()` on a `TensorProxy` will return a
    # `torch.Tensor`, which has `.device` attribute
    if device != "cuda":
        latents = latents.cpu()
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device,
    ).long()
    noise = torch.randn_like(latents)
    sample["target_noise"] = noise
    sample["noisy_latents"] = noise_scheduler.add_noise(latents, noise, timesteps)
    sample["timesteps"] = timesteps

    return sample


def run_train(  # noqa: CFQ002
    text_encoder_fn: Callable | CompiledFunction,
    vae_encoder_fn: Callable | CompiledFunction,
    unet_fn: Callable | CompiledFunction,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | MNCoreLRScheduler,
    sd_components: StableDiffusionComponents,
    sample: dict[str, torch.Tensor],
    device: str | torch.device | MNDevice,
    context: Context | None,
    train_dataloader: torch.utils.data.DataLoader,
    num_epoch: int,
) -> list[float]:

    total_iter = len(train_dataloader)
    times_per_epoch = []
    for epoch in range(num_epoch):
        with Timer() as t:
            for step, batch in enumerate(train_dataloader):
                sample = create_sample_inputs(
                    sample,
                    text_encoder_fn,
                    vae_encoder_fn,
                    sd_components.tokenizer,
                    sd_components.noise_scheduler,
                    device,
                    batch=batch,
                )
                output = unet_fn(sample)
                print(
                    f"Epoch {epoch + 1}/{num_epoch}, "
                    f"Iteration {step + 1}/{total_iter}, "
                    f"Loss: {output["result"].item():.4}"
                )
                lr_scheduler.step()
                if step == 1:
                    break
        times_per_epoch.append(t.time)
    if context is not None:  # when using MLSDK
        context.synchronize()

    return times_per_epoch


def main(args: argparse.Namespace) -> None:  # noqa: CFQ001

    # Decide the device where the training runs
    device_name = args.backend
    outdir = (
        args.outdir
        if args.outdir is not None
        else decide_outdir(
            device_name, example_name="stable_diffusion_training", basedir="/tmp"
        )
    )
    if device_name in ["cpu", "cuda"]:
        os.makedirs(outdir, exist_ok=True)

    # Fix seed for reproducibility
    set_deterministic_mode(args.seed)

    # Create stable diffusion components
    # ( text_encoder, vae, unet, tokenizer, noise_scheduler, ...)
    sd_components = StableDiffusionComponents(
        args.model_path, use_pretrained_unet=args.use_pretrained_unet
    )

    context = None
    device = None
    if device_name in ["mncore2:auto", "pfvm:cuda", "pfvm:cpu"]:
        device = MNDevice(device_name)
        context = Context(device)
        Context.switch_context(context)
    else:
        device = device_name
        sd_components.text_encoder.to(device_name)
        sd_components.vae.to(device_name)
        sd_components.unet.to(device_name)

    sd_components.set_fixed_parameters(
        device,
        outdir,
        create_sample_inputs,
        args.skip_text_encoder_compilation,
        args.skip_vae_encoder_compilation,
        args.skip_unet_compilation,
        args.num_compiler_threads,
        args.do_quiet_compilation,
        args.optimize_option,
    )

    lora_layers = None
    if args.do_lora:
        from peft import LoraConfig

        sd_components.unet.requires_grad_(False)
        lora_weights_initializer = (
            False if args.init_lora_weights == "" else args.init_lora_weights
        )
        assert lora_weights_initializer in ["gaussian", "pissa", "olora", "eva"]
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights=lora_weights_initializer,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        sd_components.unet.add_adapter(unet_lora_config)
        lora_layers = filter(lambda p: p.requires_grad, sd_components.unet.parameters())

    # Create loss function object
    loss_fn = functional.mse_loss

    # Create dataloader, optimizer, and lr_scheduler
    train_dataloader = create_train_dataloader(args, sd_components.tokenizer)

    optimizer = create_optimizer(
        args.optimizer,
        sd_components.unet.parameters() if lora_layers is None else lora_layers,
        ("mncore" in args.backend or "pfvm" in args.backend),  # use_mncore
        args.learning_rate,
        args.momentum,
        args.weight_decay,
    )
    if optimizer is None:
        print("failed to create optimizer for given optimizer's name")
        return

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup,
        num_training_steps=args.epoch,
    )
    if args.use_mncore_lr_scheduler and (context is not None):
        lr_scheduler = MNCoreLRScheduler(lr_scheduler, context)

    sample = {}

    # Set unet to training mode
    sd_components.unet.train()

    text_encoder_fn, vae_encoder_fn, unet_fn = sd_components.create_training_components(
        context, train_dataloader, args.learning_rate, optimizer, loss_fn, sample
    )

    # Run training
    train_times = run_train(
        text_encoder_fn,
        vae_encoder_fn,
        unet_fn,
        lr_scheduler,
        sd_components,
        sample,
        device,
        context,
        train_dataloader,
        args.epoch,
    )

    output_result_times(
        train_times,
        [],  # eval_times
        len(train_dataloader),
        0,  # eval_iter
        args.batch_size,
        1,
        args.optimizer,
        device_name,
        sample_name="stable_diffusion",
        optimize_option=args.optimize_option,
    )

    # Save the model
    if args.save_model:
        save_model_path = os.path.join(outdir, "model")
        print(f"saving model to {save_model_path} ...")
        pipe = StableDiffusionMNCorePipeline(
            vae=sd_components.vae,
            text_encoder=sd_components.text_encoder,
            tokenizer=sd_components.tokenizer,
            unet=sd_components.unet,
            scheduler=sd_components.noise_scheduler,
            safety_checker=sd_components.safety_checker,
            feature_extractor=sd_components.feature_extractor,
        )
        pipe.save_pretrained(save_model_path)
        print("done saving!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # mlsdk options
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="mncore2:auto",
        choices=["mncore2:auto", "pfvm:cpu", "pfvm:gpu", "cpu", "cuda"],
        help="backend to run the training/evaluation on",
    )
    parser.add_argument(
        "--optimize_option",
        type=str,
        default="debug",
        choices=["debug", "O0", "O1", "O2", "O3", "O4"],
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default=None, help="path to output dir"
    )

    # train options: general
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="path to training dataset"
    )

    apply_toml_defaults("./configs.toml", parser)

    # Parse command line args and opts
    args = parser.parse_args()

    main(args)
