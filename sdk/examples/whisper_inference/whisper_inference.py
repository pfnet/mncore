import argparse
import os
from collections.abc import Callable
from itertools import islice

import numpy as np
import torch
import whisper
from mlsdk import (
    CompiledFunction,
    Context,
    MNDevice,
    TensorLike,
    TensorProxy,
    trace_scope,
)
from static_whisper import (
    AudioEncoderHead,
    AudioEncoderTransformer,
    CrossKVPrecompute,
    TextDecoderEmbedding,
    TextDecoderTransformer,
)
from tqdm import tqdm
from utility import (
    DeviceSet,
    Timer,
    apply_toml_defaults,
    compile_fn,
    decide_outdir,
    output_result_times,
    register_model,
    set_deterministic_mode,
)
from whisper.decoding import DecodingOptions, DecodingTask
from whisper_utils import (
    LibriSpeech,
    calc_wer,
    kv_cache_from_dict,
    prepare_cache_sample,
)

DEVICES = DeviceSet(["pfvm:cpu", "mncore2:auto", "mncore2:[0-7]", "emu2", "cpu"])
MNCORE_DEVICES = DeviceSet(["mncore2:auto", "mncore2:[0-7]"])
PFVM_DEVICES = DeviceSet(["pfvm:cpu"])
EMU_DEVICES = DeviceSet(["emu2"])
MLSDK_DEVICES = MNCORE_DEVICES + PFVM_DEVICES + EMU_DEVICES


def decode_loop(
    text_decoder_embedding: TextDecoderEmbedding,
    decoder_fn: Callable | CompiledFunction,
    decoder_inputs: dict[str, TensorLike],
    decoding_task: DecodingTask,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_logprobs = torch.zeros(args.batch_size)
    mask = torch.full(
        (1, 1, 1, decoding_task.sample_len), -np.inf, device=tokens.device
    )
    max_cached_positions = mask.shape[-1]
    current_offset = 0
    model_dims = decoding_task.model.dims

    for _ in range(decoding_task.sample_len):
        current_seq_len = tokens.shape[1]
        logits = None

        # iterate over 2nd dim to handle variable len for the tokens tensor
        # e.g. (batch, n) -> n x (batch, 1)
        while current_offset < current_seq_len:
            if current_offset >= max_cached_positions:
                return tokens, sum_logprobs

            # treat only one token in the forward pass per a iteration
            input_tokens = tokens[:, current_offset : current_offset + 1]
            # update mask for scaled_dot_product_attention in self_attn
            mask[0, 0, 0, current_offset] = 0.0

            # TextDecoder forward pass
            current_offset_tensor = torch.tensor([current_offset])
            embedded_tokens = text_decoder_embedding(
                input_tokens, current_offset_tensor
            )

            if isinstance(decoder_inputs["embedded_tokens"], TensorProxy):
                decoder_inputs["embedded_tokens"].load_from(
                    embedded_tokens, clone=False
                )
                # mask is mutated at every offset; snapshot it for asynchronous H2D.
                decoder_inputs["mask"].load_from(mask, clone=True)
                decoder_inputs["offset"].load_from(current_offset_tensor, clone=False)
            else:
                # CPU/CUDA execution calls decoder_forward directly, so provide
                # tensors instead of MLSDK's device-resident input proxies.
                decoder_inputs["embedded_tokens"] = embedded_tokens
                decoder_inputs["mask"] = mask
                decoder_inputs["offset"] = current_offset_tensor

            outputs = decoder_fn(decoder_inputs)
            logits = outputs["logits"]

            current_offset += 1

        # transfer logits on device to host
        logits = logits.cpu()
        # consider the logits at the last token only
        logits = logits[:, -1]
        # apply logit filters
        for logit_filter in decoding_task.logit_filters:
            logit_filter.apply(logits, tokens)

        # expand the tokens tensor with the selected next tokens
        # and judge if all sequences has reached the EOT.
        tokens, completed = decoding_task.decoder.update(tokens, logits, sum_logprobs)

        if completed or tokens.shape[-1] > model_dims.n_text_ctx:
            break

    return tokens, sum_logprobs


def decode_hypotheses_texts(
    tokens: torch.Tensor,
    sum_logprobs: torch.Tensor,
    decoding_task: DecodingTask,
) -> list[str]:
    tokens = tokens.reshape(args.batch_size, decoding_task.n_group, -1)
    sum_logprobs = sum_logprobs.reshape(args.batch_size, decoding_task.n_group)

    # get the final candidates for each group, and slice between the first sampled token and EOT
    tokens, sum_logprobs = decoding_task.decoder.finalize(tokens, sum_logprobs)

    def slice_until_eot(t: torch.Tensor) -> torch.Tensor:
        eot_positions = (t == decoding_task.tokenizer.eot).nonzero()
        end = int(eot_positions[0, 0]) if eot_positions.numel() else t.shape[0]
        return t[decoding_task.sample_begin : end]

    tokens = [[slice_until_eot(t) for t in s] for s in tokens]

    # select the top-ranked sample in each group
    selected = decoding_task.sequence_ranker.rank(tokens, sum_logprobs)
    tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
    texts = [decoding_task.tokenizer.decode(t).strip() for t in tokens]

    return texts


def transcribe_mels(  # noqa: CFQ002
    args: argparse.Namespace,
    audio_encoder_head: AudioEncoderHead,
    encoder_fn: Callable | CompiledFunction,
    text_decoder_embedding: TextDecoderEmbedding,
    text_decoder_transformer: TextDecoderTransformer,
    decoder_fn: Callable | CompiledFunction,
    decoder_inputs: dict[str, TensorLike],
    dataloader: torch.utils.data.DataLoader,
    encoder_sample: dict[str, torch.Tensor],
    decoding_task: DecodingTask,
    context: Context | None,
) -> tuple[float, list[float]]:

    if context is not None:
        assert isinstance(decoder_inputs["embedded_tokens"], TensorProxy)

    hypotheses = []
    references = []

    with Timer() as t:

        total_iterations = len(dataloader)
        if args.max_iterations != -1:
            total_iterations = min(total_iterations, args.max_iterations)

        for mels, ref_texts in tqdm(
            islice(dataloader, total_iterations), total=total_iterations
        ):

            if mels.ndim == 2:
                mels = mels.unsqueeze(0)

            decoding_task.decoder.reset()
            text_decoder_transformer.reset_self_kv_cache(context)

            # whisper.encoder.forward pass
            embeddings = audio_encoder_head(mels)
            encoder_sample.update(embeddings=embeddings)
            cross_kv_cache = encoder_fn(encoder_sample)

            # Compiled encoder outputs are TensorProxy objects and can be passed
            # directly to the compiled decoder without a device-to-host-to-device
            # round trip. Native torch backends use the same mapping with tensors.
            decoder_inputs.update(cross_kv_cache)

            tokens = torch.tensor([decoding_task.initial_tokens]).repeat(
                args.batch_size, 1
            )
            tokens = tokens.repeat_interleave(decoding_task.n_group, dim=0)

            # decode tokens and cumulate
            tokens, sum_logprobs = decode_loop(
                text_decoder_embedding,
                decoder_fn,
                decoder_inputs,
                decoding_task,
                tokens,
            )

            # decode texts from the inferred results
            texts = decode_hypotheses_texts(tokens, sum_logprobs, decoding_task)
            hypotheses.extend(texts)
            references.extend(ref_texts)

    # Calc Word Error Rate (WER) from the inferred result and the reference data
    wer = calc_wer(hypotheses, references)

    return wer, [t.time]


def run_infer(  # noqa: CFQ001, CFQ002
    args: argparse.Namespace,
    audio_encoder_head: AudioEncoderHead,
    audio_encoder_transformer: AudioEncoderTransformer,
    text_decoder_embedding: TextDecoderEmbedding,
    text_decoder_transformer: TextDecoderTransformer,
    decoding_task: DecodingTask,
    dataloader: torch.utils.data.DataLoader,
    context: Context | None,
    outdir: str,
) -> tuple[float, list[float]]:

    model_dims = decoding_task.model.dims
    cross_kv_precompute = CrossKVPrecompute(text_decoder_transformer)

    def encoder_forward(sample_d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            audio_features = audio_encoder_transformer(sample_d["embeddings"])
            cross_kv_cache = cross_kv_precompute(audio_features)
        return cross_kv_cache

    def decoder_forward(sample_d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cross_kv_cache = kv_cache_from_dict(sample_d, model_dims.n_text_layer, "cross")

        with torch.no_grad():
            logits = text_decoder_transformer(
                sample_d["embedded_tokens"],
                sample_d["offset"],
                sample_d["mask"],
                cross_kv_cache,
            )

        return {"logits": logits}

    # prepare sample inputs for compilation of encoder_fn and decoder_fn
    # The decoder processes one token per invocation in decode_loop().
    decode_token_length = 1
    encoder_sample = {
        "embeddings": torch.randn(
            args.batch_size, model_dims.n_audio_ctx, model_dims.n_audio_state
        )
    }
    cross_k_cache_dict, cross_v_cache_dict = prepare_cache_sample(args, model_dims)

    decoder_sample = (
        {
            "embedded_tokens": torch.randn(
                args.batch_size,
                decode_token_length,
                model_dims.n_text_state,
                dtype=torch.float32,
            ),
            "offset": torch.tensor([1]),
            "mask": torch.zeros(
                1, 1, 1, model_dims.n_text_ctx // 2
            ),  # size of dim=3 is hard-coded
        }
        | cross_k_cache_dict
        | cross_v_cache_dict
    )

    encoder_fn = encoder_forward
    decoder_fn = decoder_forward
    decoder_inputs: dict[str, TensorLike] = dict(decoder_sample)
    if context is not None:  # for mlsdk
        # CrossKVPrecompute shares key/value projection parameters with the
        # decoder. Register the owning decoder module first to avoid registering
        # the shared tensors twice while compiling the encoder.
        register_model(context, "whisper_decoder_transformer", text_decoder_transformer)

        # Compile the consumer first so its preferred cross-attention cache
        # layouts become available to the encoder compilation.
        print("\n########## compile decoder_fn ###########")
        decoder_fn = compile_fn(
            context,
            decoder_fn,
            text_decoder_transformer,
            decoder_sample,
            outdir=outdir + "_decoder_transformer",
            model_name="whisper_decoder_transformer",
            is_train=False,
            optimize_option=args.optimize_option,
            preset_options_dir=args.preset_options_dir,
            enable_cache=not args.disable_compile_cache,
        )

        decoder_inputs = decoder_fn.allocate_input_proxy()
        cross_cache_names = set(cross_k_cache_dict) | set(cross_v_cache_dict)
        for name, tensor in decoder_sample.items():
            if name not in cross_cache_names:
                decoder_inputs[name].load_from(tensor, clone=False)

        # Constrain the encoder outputs to the decoder input IOSpecs. This passes
        # only layout/type metadata to compilation; the encoder and decoder keep
        # independent allocations, and runtime relocation connects them directly.
        cross_cache_constraints = {
            name: decoder_fn.input_specs[name] for name in cross_cache_names
        }

        print("\n########## compile encoder_fn ##########")
        encoder_fn = compile_fn(
            context,
            encoder_fn,
            {
                "whisper_encoder_transformer": audio_encoder_transformer,
                "whisper_cross_kv_precompute": cross_kv_precompute,
            },
            encoder_sample,
            outdir=outdir + "_encoder_transformer",
            model_name="whisper_encoder_transformer",
            is_train=False,
            optimize_option=args.optimize_option,
            preset_options_dir=args.preset_options_dir,
            enable_cache=not args.disable_compile_cache,
            io_spec_constraints=cross_cache_constraints,
        )

    return transcribe_mels(
        args,
        audio_encoder_head,
        encoder_fn,
        text_decoder_embedding,
        text_decoder_transformer,
        decoder_fn,
        decoder_inputs,
        dataloader,
        encoder_sample,
        decoding_task,
        context,
    )


def main(args: argparse.Namespace) -> None:
    if args.max_iterations == 0 or args.max_iterations < -1:
        raise ValueError("max_iterations must be a positive integer or -1")
    if args.wer_criteria < 0.0 or args.wer_criteria > 1.0:
        raise ValueError("wer_criteria must be ranged in [0, 1]")

    # Fix seed values for reproducibility
    set_deterministic_mode(seed=args.seed)

    # Create model obj
    model = whisper.load_model(args.model, download_root=args.model_dir)
    model.alignment_heads = model.alignment_heads.to_dense()
    model.eval()
    audio_encoder_head = AudioEncoderHead(model.encoder)
    audio_encoder_transformer = AudioEncoderTransformer(model.encoder)
    text_decoder_embedding = TextDecoderEmbedding(model.decoder)
    text_decoder_transformer = TextDecoderTransformer(model.decoder, args.batch_size)

    # Create decoding_task obj to use DecodingTask params in the inference
    decoding_options = DecodingOptions(language="en", without_timestamps=True)
    decoding_task = DecodingTask(model, decoding_options)

    # Create Dataset/DataLoader obj
    infer_dataset = LibriSpeech(
        dataset_dir=args.dataset_dir,
        data_name=args.data_name,
        chunk_length=args.chunk_length,
        n_mels=model.dims.n_mels,
        device="cpu",
    )
    infer_dataloader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=args.batch_size, drop_last=True
    )
    infer_iterations = len(infer_dataloader)
    if args.max_iterations != -1:
        infer_iterations = min(infer_iterations, args.max_iterations)

    # Decide device and outdir from given command line args/options
    sample_name = "whisper_inference"
    outdir = decide_outdir(
        args.backend, example_name=sample_name, basedir=args.out_basedir
    )

    # Pass device info to the Context obj
    context = None
    if args.backend in MLSDK_DEVICES:
        device = MNDevice(args.backend)
        context = Context(device)
        Context.switch_context(context)
    else:
        device = args.backend

    with trace_scope(args.trace_filename):
        # Run inference
        wer, infer_times = run_infer(
            args,
            audio_encoder_head,
            audio_encoder_transformer,
            text_decoder_embedding,
            text_decoder_transformer,
            decoding_task,
            infer_dataloader,
            context,
            outdir,
        )

    # Output the inference time
    output_result_times(
        eval_times=infer_times,
        eval_iter=infer_iterations,
        eval_batch_size=args.batch_size,
        backend_name=args.backend,
        sample_name=sample_name,
        optimize_option=args.optimize_option,
    )

    print(f"\nResult: WordErrorRate(WER) = {wer * 100:.2f} %")
    assert wer <= args.wer_criteria, (
        f"{wer * 100:.2f} % exceeded the "
        f"specified criteria ({args.wer_criteria * 100:.2f} %)."
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=f"{script_dir}/librispeech_dataset",
        help="path to save dataset",
    )
    parser.add_argument(
        "--out_basedir",
        type=str,
        default="/tmp",
        help="basedir to save output artifacts",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/tmp/whisper_models",
        help="directory in which to download and load Whisper models",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=-1,
        help="maximum number of DataLoader iterations; -1 processes all samples",
    )
    parser.add_argument(
        "--wer_criteria",
        type=float,
        default=0.05,
        help="acceptable WordErrorRate (WER) criteria",
    )

    # mlsdk options
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=DEVICES,
        default="cpu",
        help="mlsdk options: specify a device as the backend (default: %(default)s)",
    )
    parser.add_argument(
        "--optimize_option",
        type=str,
        default="O1",
        choices=["debug", "O0", "O1", "O2", "O3", "O4"],
        help="mlsdk options: optimize option for Context.compile() (default %(default)s)",
    )
    parser.add_argument(
        "--preset_options_dir",
        type=str,
        default="/opt/pfn/pfcomp/codegen/preset_options",
        help="mlsdk options: path to preset_options/",
    )
    parser.add_argument(
        "--disable_compile_cache",
        action="store_true",
        help="mlsdk options: disable reusing compiled artifacts",
    )
    parser.add_argument(
        "--trace_filename",
        type=str,
        default=None,
        help="mlsdk options: specify a valid filename to enable tracing",
    )

    apply_toml_defaults(f"{script_dir}/configs.toml", parser)

    args = parser.parse_args()

    main(args)
