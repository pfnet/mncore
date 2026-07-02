import os
from collections.abc import Callable

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from mlsdk import CacheOptions, CompiledFunction, Context, MNCoreOptimizer, MNDevice
from stable_diffusion_utility import PRESET_OPTIONS_DIR, compile_fn
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


class StableDiffusionComponents:
    def __init__(  # noqa: CFQ002
        self,
        model_path: str,
        vae: AutoencoderKL | None = None,
        text_encoder: CLIPTextModel | None = None,
        tokenizer: CLIPTokenizer | None = None,
        unet: UNet2DConditionModel | None = None,
        noise_scheduler: DDIMScheduler | None = None,
        safety_checker: StableDiffusionSafetyChecker | None = None,
        feature_extractor: CLIPFeatureExtractor | None = None,
        use_pretrained_unet: bool = False,
    ) -> None:

        # Initialize stable diffusion component models
        self.vae = (
            AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            if vae is None
            else vae
        )
        self.text_encoder = (
            CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
            if text_encoder is None
            else text_encoder
        )
        if use_pretrained_unet:
            self.unet = (
                UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
                if unet is None
                else unet
            )
        else:
            self.unet = UNet2DConditionModel(cross_attention_dim=768, sample_size=64)
        self.noise_scheduler = (
            DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
            if noise_scheduler is None
            else noise_scheduler
        )
        self.tokenizer = (
            CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            if tokenizer is None
            else tokenizer
        )
        self.safety_checker = (
            StableDiffusionSafetyChecker.from_pretrained(
                model_path, subfolder="safety_checker"
            )
            if safety_checker is None
            else safety_checker
        )
        self.feature_extractor = (
            CLIPFeatureExtractor.from_pretrained(
                model_path, subfolder="feature_extractor"
            )
            if feature_extractor is None
            else feature_extractor
        )

        # Freeze text_encoder, and vae and set unet to training mode
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Create component functions' obj used in training
        self.created_text_encoder_fn = None
        self.created_vae_encoder_fn = None
        self.created_unet_fn = None

    def set_fixed_parameters(  # noqa: CFQ002
        self,
        device: str | torch.device | MNDevice,
        outdir: str,
        create_sample_fn: Callable,
        skip_text_encoder_compilation: bool = False,
        skip_vae_encoder_compilation: bool = False,
        skip_unet_compilation: bool = False,
        num_compiler_threads: int = -1,
        do_quiet_compilation: bool = False,
        optimize_option: str = "debug",
    ) -> None:
        self._device = device
        self.outdir = outdir
        self.create_sample_fn = create_sample_fn
        self.num_compiler_threads = num_compiler_threads
        self.skip_text_encoder_compilation = skip_text_encoder_compilation
        self.skip_vae_encoder_compilation = skip_vae_encoder_compilation
        self.skip_unet_compilation = skip_unet_compilation
        self.do_quiet_compilation = do_quiet_compilation
        self.optimize_option = optimize_option
        self.optimize_option_filepath = os.path.join(
            PRESET_OPTIONS_DIR, f"{optimize_option}.json"
        )
        self.compile_opts = {"option_json": self.optimize_option_filepath}

    def compile_text_encoder(
        self,
        context: Context,
        sample: dict[str, torch.Tensor],
        compiled_fn: Callable,
    ) -> CompiledFunction:
        return compile_fn(
            context,
            compiled_fn,
            {"text_encoder": self.text_encoder},
            sample,
            outdir=self.outdir,
            model_name="text_encoder",
            is_train=False,
            optimize_option=self.optimize_option,
            cache_options=CacheOptions(self.outdir + "/text_encoder_cache"),
            num_compiler_threads=self.num_compiler_threads,
            quiet=self.do_quiet_compilation,
        )

    def compile_vae_encoder(
        self,
        context: Context,
        sample: dict[str, torch.Tensor],
        compiled_fn: Callable,
    ) -> CompiledFunction:
        return compile_fn(
            context,
            compiled_fn,
            {"vae_quant_conv": self.vae.quant_conv, "vae_encoder": self.vae.encoder},
            sample,
            outdir=self.outdir,
            model_name="vae_encoder",
            is_train=False,
            optimize_option=self.optimize_option,
            cache_options=CacheOptions(self.outdir + "/vae_encoder_cache"),
            num_compiler_threads=self.num_compiler_threads,
            quiet=self.do_quiet_compilation,
            envs={
                "CODEGEN_GEMM_FORCE_WEIGHT_ON_DRAM": "1",
                "CODEGEN_CONV_AGGRESSIVE_TIME_SLICING": "1",
                "CODEGEN_GN_AGGRESSIVE_TIME_SLICING": "1",
            },
        )

    def compile_unet(
        self,
        context: Context,
        sample: dict[str, torch.Tensor],
        compiled_fn: Callable,
        optimizer: MNCoreOptimizer,
        learning_rate: float,
    ) -> CompiledFunction:
        return compile_fn(
            context,
            compiled_fn,
            {"unet": self.unet},
            sample,
            outdir=self.outdir,
            model_name="unet",
            is_train=True,
            optimizer_configs=[(optimizer, learning_rate)],
            optimize_option=self.optimize_option,
            cache_options=CacheOptions(self.outdir + "/unet_cache"),
            num_compiler_threads=self.num_compiler_threads,
            quiet=self.do_quiet_compilation,
            envs={
                "CODEGEN_SKIP_OPTIMIZER_FUSION": "1",
            },
        )

    def create_training_components(  # noqa: CFQ002 CFQ004
        self,
        context: Context,
        train_dataloader: torch.utils.data.DataLoader,
        learning_rate: float,
        optimizer: torch.optim.Optimizer | MNCoreOptimizer,
        loss_fn: Callable,
        sample: dict[str, torch.Tensor],
    ) -> tuple[
        Callable | CompiledFunction,
        Callable | CompiledFunction,
        Callable | CompiledFunction,
    ]:

        def text_encoder_fn(
            sample_d: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            embeddings = self.text_encoder(
                sample_d["input_ids"],
                position_ids=sample_d["position_ids"],
                return_dict=False,
            )[0]

            return {"embeddings": embeddings}

        def vae_encoder_fn(
            sample_d: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            latents = self.vae.encode(sample_d["pixel_values"]).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            return {"latents": latents}

        def unet_fn_fx2onnx(
            sample_d: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            optimizer.zero_grad()
            noise_pred = self.unet(
                sample_d["noisy_latents"],
                sample_d["timesteps"],
                encoder_hidden_states=sample_d["encoder_hidden_states"],
            ).sample
            loss = loss_fn(noise_pred.float(), sample_d["target_noise"].float())
            loss.backward()
            optimizer.step()

            return {"result": loss}

        # Create sample inputs to compile component models
        sample = self.create_sample_fn(
            sample,
            text_encoder_fn,
            vae_encoder_fn,
            self.tokenizer,
            self.noise_scheduler,
            self._device,
            train_dataloader=train_dataloader,
        )

        if self.skip_text_encoder_compilation or self._device in ["cpu", "cuda"]:
            self.created_text_encoder_fn = text_encoder_fn
        else:
            print("compiling text_encoder...")
            self.created_text_encoder_fn = self.compile_text_encoder(
                context,
                sample,
                text_encoder_fn,
            )
            print("end text_encoder's compilation!")

        if self.skip_vae_encoder_compilation or self._device in ["cpu", "cuda"]:
            self.created_vae_encoder_fn = vae_encoder_fn
        else:
            print("compiling vae encoder...")
            self.created_vae_encoder_fn = self.compile_vae_encoder(
                context,
                sample,
                vae_encoder_fn,
            )
            print("end vae encoder's compilation!")

        if self.skip_unet_compilation or self._device in ["cpu", "cuda"]:
            self.created_unet_fn = unet_fn_fx2onnx
        else:
            print("compiling unet...")
            self.created_unet_fn = self.compile_unet(
                context,
                sample,
                unet_fn_fx2onnx,
                optimizer,
                learning_rate,
            )
            print("end unet' compilation!")

        return (
            self.created_text_encoder_fn,
            self.created_vae_encoder_fn,
            self.created_unet_fn,
        )
