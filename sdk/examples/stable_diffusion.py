import argparse
import inspect
from typing import List, Optional, Union

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from mlsdk import CacheOptions, Context, MNDevice, storage
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


class StableDiffusionMNCorePipeline(StableDiffusionPipeline):
    def __init__(  # noqa: CFQ002
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
        )
        self.compiled_text_encoder = None
        self.compiled_unet = None
        self.compiled_vae_decoder = None

    def compile_encoder(
        self,
        context,
        batch_size: int,
        out_dir: str,
        num_compiler_threads: Optional[int] = None,
    ):
        seq_len = self.tokenizer.model_max_length

        def text_encoder_fn(inp):
            input_ids = inp["input_ids"]
            position_ids = inp["position_ids"]
            embeddings = self.text_encoder(input_ids, position_ids=position_ids)[0]
            return {"embeddings": embeddings}

        context.registry.register("text_encoder", self.text_encoder)

        return context.compile(
            text_encoder_fn,
            {
                "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64).to(
                    self.device
                ),
                # fx2onnx failed to process the buffer which is created by view
                # So pass position_ids explicitly
                "position_ids": torch.arange(seq_len).expand((1, -1)).to(self.device),
            },
            storage.path(out_dir + "/text_encoder"),
            export_kwargs={"use_fx2onnx": True},
            cache_options=CacheOptions(out_dir + "/encoder_cache"),
            num_compiler_threads=num_compiler_threads,
        )

    def compile_unet(  # noqa: CFQ002
        self,
        context,
        batch_size: int,
        height: int,
        width: int,
        guidance_scale: float,
        out_dir: str,
        num_compiler_threads: Optional[int] = None,
    ):
        seq_len = self.tokenizer.model_max_length
        do_classifier_free_guidance = guidance_scale > 1.0

        def unet_fn(inp):
            latents = inp["latents"]
            timesteps = inp["timesteps"]
            text_embeddings = inp["text_embeddings"]
            noise_pred = self.unet(
                latents, timesteps, encoder_hidden_states=text_embeddings
            ).sample
            return {"sample": noise_pred}

        context.registry.register("unet", self.unet)

        return context.compile(
            unet_fn,
            {
                "latents": torch.zeros(
                    (
                        batch_size * 2 if do_classifier_free_guidance else 1,
                        self.unet.in_channels,
                        height // 8,
                        width // 8,
                    )
                ).to(self.device),
                "timesteps": torch.tensor([0], dtype=torch.long),
                "text_embeddings": torch.zeros(
                    (
                        batch_size * 2 if do_classifier_free_guidance else 1,
                        seq_len,
                        self.text_encoder.config.hidden_size,
                    )
                ),
            },
            storage.path(out_dir + "/unet"),
            export_kwargs={"use_fx2onnx": True},
            cache_options=CacheOptions(out_dir + "/unet_cache"),
            num_compiler_threads=num_compiler_threads,
        )

    def compile_vae_decoder(  # noqa: CFQ002
        self,
        context,
        batch_size: int,
        height: int,
        width: int,
        out_dir: str,
        num_compiler_threads: Optional[int] = None,
    ):
        def vae_decoder_fn(inp):
            z = inp["z"]
            image = self.vae.decode(z).sample
            return {"image": image}

        context.registry.register("vae_post_quant_conv", self.vae.post_quant_conv)
        context.registry.register("vae_decoder", self.vae.decoder)

        return context.compile(
            vae_decoder_fn,
            {
                "z": torch.zeros(
                    (batch_size, self.unet.in_channels, height // 8, width // 8),
                ).to(self.device),
            },
            storage.path(out_dir + "/vae_decoder"),
            export_kwargs={"use_fx2onnx": True},
            cache_options=CacheOptions(out_dir + "/vae_decoder_cache"),
            num_compiler_threads=num_compiler_threads,
        )

    def compile(  # noqa: CFQ002
        self,
        *,
        batch_size: int,
        device: str,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        out_dir: str = "/tmp/mlsdk_stable_diffusion_out",
        skip_text_encoder_compilation: bool = False,
        skip_unet_compilation: bool = False,
        skip_vae_decoder_compilation: bool = False,
        num_compiler_threads: Optional[int] = None,
    ):
        device = MNDevice(device)
        context = Context(device)
        Context.switch_context(context)

        if not skip_text_encoder_compilation:
            self.compiled_text_encoder = self.compile_encoder(
                context,
                batch_size,
                out_dir=out_dir,
                num_compiler_threads=num_compiler_threads,
            )
        if not skip_unet_compilation:
            self.compiled_unet = self.compile_unet(
                context,
                batch_size,
                height,
                width,
                guidance_scale,
                out_dir=out_dir,
                num_compiler_threads=num_compiler_threads,
            )
        if not skip_vae_decoder_compilation:
            self.compiled_vae_decoder = self.compile_vae_decoder(
                context,
                batch_size,
                height,
                width,
                out_dir=out_dir,
                num_compiler_threads=num_compiler_threads,
            )

    def infer_text_encoder(self, input_ids, position_ids=None):
        if self.compiled_text_encoder is not None:
            if position_ids is None:
                position_ids = (
                    torch.arange(self.tokenizer.model_max_length)
                    .expand((1, -1))
                    .to(self.device)
                )
            return self.compiled_text_encoder(
                {"input_ids": input_ids, "position_ids": position_ids}
            )["embeddings"]
        else:
            return self.text_encoder(input_ids)[0]

    def infer_unet(self, latent_model_input, t, text_embeddings):
        if self.compiled_unet is not None:
            return self.compiled_unet(
                {
                    "latents": latent_model_input,
                    "timesteps": torch.tensor([t], dtype=torch.long),
                    "text_embeddings": text_embeddings,
                }
            )["sample"]
        else:
            return self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

    def infer_vae_decode(self, z):
        if self.compiled_vae_decoder is not None:
            return self.compiled_vae_decoder({"z": z})["image"]
        else:
            return self.vae.decode(z).sample

    # Ref https://github.com/huggingface/diffusers/blob/v0.2.4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py # noqa: B950
    # Ref https://github.com/huggingface/diffusers/blob/v0.8.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py # noqa: B950
    @torch.no_grad()
    def __call__(  # noqa: CFQ002,CFQ001
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        **kwargs,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = self.infer_text_encoder(text_input.input_ids.to(self.device))

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.infer_text_encoder(
                uncond_input.input_ids.to(self.device)
            )
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the intial random noise
        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
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
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.infer_unet(latent_model_input, t, text_embeddings)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.infer_vae_decode(latents)
        image = (image.cpu() / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).numpy()

        # run safety checker
        image, has_nsfw_concept = self.run_safety_checker(
            image, self.device, text_embeddings.dtype
        )

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}


def main(args):
    pipe = StableDiffusionMNCorePipeline.from_pretrained(args.model)
    pipe.compile(
        batch_size=1,
        device=args.device,
        out_dir=args.outdir,
        skip_text_encoder_compilation=args.skip_text_encoder_compilation,
        skip_unet_compilation=args.skip_unet_compilation,
        skip_vae_decoder_compilation=args.skip_vae_decoder_compilation,
        num_compiler_threads=args.num_compiler_threads,
    )

    if ":cuda:" in args.device:
        pipe.to(args.device[args.device.find(":cuda:") + 1 :])

    prompt = args.prompt
    image = pipe(prompt)["sample"][0]

    image.save(f"{args.outdir}/output.png")
    print(f"Output image saved at {args.outdir}/output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/tmp/mlsdk_stable_diffusion_out",
        help="Path to store the outputs",
    )
    parser.add_argument("--device", default="mncore2:auto")
    parser.add_argument(
        "--prompt", type=str, default="a photo of an astronaut riding a horse on mars"
    )
    parser.add_argument(
        "--num_compiler_threads",
        type=int,
        default=-1,
        help="Number of threads to use for compilation",
    )
    parser.add_argument("--skip_text_encoder_compilation", action="store_true")
    parser.add_argument("--skip_unet_compilation", action="store_true")
    parser.add_argument("--skip_vae_decoder_compilation", action="store_true")

    args = parser.parse_args()
    main(args)
