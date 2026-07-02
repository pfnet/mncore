import argparse
import inspect
import os
from collections.abc import Callable

import diffusers
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from mlsdk import CacheOptions, CompiledFunction, Context, MNDevice, storage
from PIL import Image
from stable_diffusion_utility import (
    PRESET_OPTIONS_DIR,
    Timer,
    apply_toml_defaults,
    decide_outdir,
    output_result_times,
    set_deterministic_mode,
)
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


# Use this class obj for model evaluation before and after training
class StableDiffusionMNCorePipeline(StableDiffusionPipeline):
    def __init__(  # noqa: CFQ002
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ) -> None:
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        self.eval_text_encoder_fn = None
        self.eval_unet_fn = None
        self.eval_vae_decoder_fn = None
        self.is_text_encoder_registered = False
        self.is_unet_registered = False
        self.is_vae_decoder_registered = False

    def set_models(  # noqa: CFQ002
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker | None = None,
        feature_extractor: CLIPFeatureExtractor | None = None,
    ) -> None:
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        if safety_checker is not None:
            self.safety_checker = safety_checker
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor

    def set_fixed_parameters(  # noqa: CFQ002
        self,
        outdir: str,
        prompt: str,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        skip_text_encoder_compilation: bool = False,
        skip_unet_compilation: bool = False,
        skip_vae_decoder_compilation: bool = False,
        num_compiler_threads: int = -1,
        do_quiet_compilation: bool = False,
        optimize_option: str = "debug",
    ) -> None:
        self.outdir = outdir
        self.prompt = prompt
        self.height = height
        self.width = width
        if int(diffusers.__version__.split(".")[1]) >= 22:  # diffuers >= 0.22.0
            self._guidance_scale = guidance_scale
        else:  # diffuers < 0.22.0
            self.guidance_scale = guidance_scale
            self.do_classifier_free_guidance = self.guidance_scale > 1.0
        self.skip_text_encoder_compilation = skip_text_encoder_compilation
        self.skip_unet_compilation = skip_unet_compilation
        self.skip_vae_decoder_compilation = skip_vae_decoder_compilation
        self.num_compiler_threads = num_compiler_threads
        self.do_quiet_compilation = do_quiet_compilation
        self.optimize_option_filepath = os.path.join(
            PRESET_OPTIONS_DIR, f"{optimize_option}.json"
        )
        self.compile_opts = {"option_json": self.optimize_option_filepath}

    def create_text_encoder_fn(
        self,
        context: Context | None,
        name_suffix: str,
        batch_size: int = 1,
    ) -> Callable | CompiledFunction:
        seq_len = self.tokenizer.model_max_length

        def text_encoder_fn(
            sample_d: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            input_ids = sample_d["input_ids"]
            position_ids = sample_d["position_ids"]
            embeddings = self.text_encoder(input_ids, position_ids=position_ids)[0]

            return {"embeddings": embeddings}

        if context is None or self.skip_text_encoder_compilation:
            return text_encoder_fn
        else:
            if not self.is_text_encoder_registered:
                context.registry.register("text_encoder", self.text_encoder)
                self.is_text_encoder_registered = True

            input_ids = torch.zeros((batch_size, seq_len), dtype=torch.int64)
            position_ids = torch.arange(seq_len).expand((1, -1))
            if self.device in ["cpu", "cuda"]:
                input_ids = input_ids.to(self.device)
                position_ids = position_ids.to(self.device)

            compile_args = {
                "function": text_encoder_fn,
                "inputs": {  # Pass the shape information to context for compilation
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                },
                "options": self.compile_opts,
                "cache_options": CacheOptions(self.outdir + "/text_encoder_eval_cache"),
                "num_compiler_threads": self.num_compiler_threads,
                "quiet": self.do_quiet_compilation,
            }
            codegen_base_dir = storage.path(self.outdir)
            name = "text_encoder_eval_" + name_suffix
            compile_args["codegen_dir"] = codegen_base_dir / name

            return context.compile(**compile_args)

    def create_unet_fn(
        self,
        context: Context | None,
        name_suffix: str,
        batch_size: int = 1,
    ) -> Callable | CompiledFunction:

        seq_len = self.tokenizer.model_max_length

        def unet_fn_fx2onnx(
            sample_d: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            noise_pred = self.unet(
                sample_d["latents"],
                sample_d["timesteps"],
                encoder_hidden_states=sample_d["encoder_hidden_states"],
            ).sample

            return {"result": noise_pred}

        if context is None or self.skip_unet_compilation:
            return unet_fn_fx2onnx
        else:  # Run in MLSDK/on MN-Core
            if not self.is_unet_registered:
                context.registry.register("unet", self.unet)
                self.is_unet_registered = True

            options = self.compile_opts.copy()
            compile_args = {
                "function": unet_fn_fx2onnx,
                "inputs": {
                    "latents": torch.zeros(
                        batch_size * 2 if self.do_classifier_free_guidance else 1,
                        self.unet.config.in_channels,
                        self.height // 8,
                        self.width // 8,
                    ).to(self.device),
                    "timesteps": torch.tensor([0], dtype=torch.long),
                    "encoder_hidden_states": torch.zeros(
                        batch_size * 2 if self.do_classifier_free_guidance else 1,
                        seq_len,
                        self.text_encoder.config.hidden_size,
                    ),
                },
                "options": options,
                "cache_options": CacheOptions(self.outdir + "/unet_eval_cache"),
                "num_compiler_threads": self.num_compiler_threads,
                "quiet": self.do_quiet_compilation,
            }
            codegen_base_dir = storage.path(self.outdir)
            name = "unet_eval_" + name_suffix
            compile_args["codegen_dir"] = codegen_base_dir / name

            return context.compile(**compile_args)

    def create_vae_decoder_fn(
        self,
        context: Context | None,
        name_suffix: str,
        batch_size: int = 1,
    ) -> Callable | CompiledFunction:

        def vae_decoder_fn(
            sample_d: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            image = self.vae.decode(sample_d["z"]).sample
            return {"image": image}

        if context is None or self.skip_vae_decoder_compilation:
            return vae_decoder_fn
        else:
            if not self.is_vae_decoder_registered:
                context.registry.register(
                    "vae_post_quant_conv", self.vae.post_quant_conv
                )
                context.registry.register("vae_decoder", self.vae.decoder)
                self.is_vae_decoder_registered = True

            z = torch.zeros(
                (
                    batch_size,
                    self.unet.config.in_channels,
                    self.height // 8,
                    self.width // 8,
                )
            )
            if self.device in ["cpu", "cuda"]:
                z = z.to(self.device)
            options = self.compile_opts.copy()
            options["envs"] = {
                "CODEGEN_GEMM_FORCE_WEIGHT_ON_DRAM": "1",
                "CODEGEN_CONV_AGGRESSIVE_TIME_SLICING": "1",
                "CODEGEN_GN_AGGRESSIVE_TIME_SLICING": "1",
                "CODEGEN_GENERIC_MOVE_CHUNK_SIZE": "32768",  # 32 * 1024 LWs
            }
            compile_args = {
                "function": vae_decoder_fn,
                "inputs": {
                    "z": torch.zeros(
                        (
                            batch_size,
                            self.unet.config.in_channels,
                            self.height // 8,
                            self.width // 8,
                        )
                    ).to(self.device)
                },
                "options": options,
                "cache_options": CacheOptions(self.outdir + "/vae_decoder_cache"),
                "num_compiler_threads": self.num_compiler_threads,
                "quiet": self.do_quiet_compilation,
                "codegen_dir": storage.path(self.outdir) / f"vae_decoder_{name_suffix}",
            }

            return context.compile(**compile_args)

    def create_eval_components(
        self,
        *,
        context: Context | None,
        name_suffix: str,
        batch_size: int = 1,
    ) -> None:

        self.eval_text_encoder_fn = self.create_text_encoder_fn(
            context,
            name_suffix,
            batch_size=batch_size,
        )

        self.eval_unet_fn = self.create_unet_fn(
            context,
            name_suffix,
            batch_size=batch_size,
        )

        self.eval_vae_decoder_fn = self.create_vae_decoder_fn(
            context,
            name_suffix,
            batch_size=batch_size,
        )

    # Ref https://github.com/huggingface/diffusers/blob/v0.2.4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py  # noqa: B950
    # Ref https://github.com/huggingface/diffusers/blob/v0.8.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py  # noqa: B950
    @torch.no_grad()
    def __call__(  # noqa: CFQ001 CFQ002
        self,
        num_inference_steps: int | None = 50,
        eta: float | None = 0.0,
        latents: (
            torch.Tensor | None
        ) = None,  # parameterize to allow its value to be fixed for model evaluation
        generator: torch.Generator | None = None,
        output_type: str | None = "pil",  # PIL.Image.Image
        **kwargs,
    ) -> tuple[list[Image.Image], list[bool]]:

        if isinstance(self.prompt, str):
            batch_size = 1
        elif isinstance(self.prompt, list):
            batch_size = len(self.prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(self.prompt)}."
            )

        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ValueError(
                "`height` and `width` have to be divisible by 8 but are "
                f"{self.height} and {self.width}."
            )

        sample = {}

        # get prompt text embeddings
        text_input = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        sample["input_ids"] = text_input.input_ids.to(self.device)
        sample["position_ids"] = (
            torch.arange(self.tokenizer.model_max_length)
            .expand((1, -1))
            .to(self.device)
        )

        text_embeddings = self.eval_text_encoder_fn(sample)["embeddings"]
        if self.do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids
            if self.device in ["cpu", "cuda"]:
                uncond_input_ids = uncond_input_ids.to(self.device)
            uncond_embeddings = self.eval_text_encoder_fn(
                {
                    "input_ids": uncond_input.input_ids.to(self.device),
                    "position_ids": sample["position_ids"],
                }
            )["embeddings"]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        sample["encoder_hidden_states"] = text_embeddings

        # get the inital random noise
        latents = (
            torch.randn(
                (
                    batch_size,
                    self.unet.config.in_channels,
                    self.height // 8,
                    self.width // 8,
                ),
                generator=generator,
                device=self.device,
            )
            if latents is None
            else latents
        )
        latents = latents * self.scheduler.init_noise_sigma

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if self.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            sample["latents"] = latent_model_input
            sample["timesteps"] = t
            noise_pred = self.eval_unet_fn(sample)["result"]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_{t-1}
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # scale and decode the image latents with vae
        latents = (
            1 / self.vae.config.scaling_factor * latents
        )  # vae.config.scaling_factor: defaults to 0.18215
        sample["z"] = latents
        image = self.eval_vae_decoder_fn(sample)["image"].cpu()
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).numpy()

        # run safety_checker
        has_nsfw_concept = None
        if self.safety_checker is not None:
            image, has_nsfw_concept = self.run_safety_checker(
                image, self.device, text_embeddings.dtype
            )

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}


def eval_model(
    pipe: StableDiffusionMNCorePipeline,
    context: Context | None,
    name_suffix: str,
) -> float:
    # Set unet to eval mode
    pipe.unet.eval()
    pipe.create_eval_components(
        context=context,
        name_suffix=name_suffix,
        batch_size=1,
    )

    with Timer() as t:
        image = pipe()["sample"][0]

    image.save(f"{pipe.outdir}/output_{name_suffix}.png")

    return t.time


# For test/example of StableDiffusionPipeline class
def main(args: argparse.Namespace) -> None:

    # Decide the device name and output dir
    device_name = args.backend
    outdir = (
        args.outdir
        if args.outdir is not None
        else decide_outdir(
            device_name, example_name="stable_diffusion_eval", basedir="/tmp"
        )
    )

    context = None
    if device_name in ["mncore2:0", "pfvm:cpu", "pfvm:gpu"]:
        device = MNDevice(device_name)
        context = Context(device)
        Context.switch_context(context)
    else:  # Run on CPU or GPU
        device = device_name
        os.makedirs(outdir, exist_ok=True)  # Create outdir when using CPU or GPU

    # Fix seeds for model evaluation
    set_deterministic_mode(args.seed)

    # Load pipeline components from pretrained model files
    pipe = StableDiffusionMNCorePipeline.from_pretrained(args.model_path)

    if args.skip_safety_check:
        pipe.safety_checker = None
        pipe.feature_extractor = None

    # Set fixed parameters
    pipe.set_fixed_parameters(
        outdir=outdir,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        skip_text_encoder_compilation=args.skip_text_encoder_compilation,
        skip_unet_compilation=args.skip_unet_compilation,
        skip_vae_decoder_compilation=args.skip_vae_decoder_compilation,
        num_compiler_threads=args.num_compiler_threads,
        optimize_option=args.optimize_option,
    )
    # Create components used in evaluation
    pipe.create_eval_components(
        context=context,
        name_suffix="eval",
        batch_size=len(args.prompt) if isinstance(args.prompt, list) else 1,
    )

    # Generate an image for the given prompt and save the image
    with Timer() as t:
        image = pipe()["sample"][0]

    batch_size = len(args.prompt) if isinstance(args.prompt, list) else 1
    output_result_times(
        None,
        [t.time],
        0,
        1,
        None,
        batch_size,
        None,
        device_name,
        sample_name="stable_diffusion_inference",
        optimize_option=args.optimize_option,
    )

    image.save(f"{pipe.outdir}/output_eval.png")


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

    # inference options: general
    parser.add_argument("--prompt", type=str, default="dog")

    apply_toml_defaults("./configs.toml", parser)

    # Parse given command line args and opts
    args = parser.parse_args()

    main(args)
