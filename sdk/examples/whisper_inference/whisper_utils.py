import argparse
import os

import jiwer
import torch
import torchaudio
import whisper
from whisper.model import ModelDimensions
from whisper.normalizers import EnglishTextNormalizer


# Wrapper class for torchaudio.datasets.LIBRISPEECH class
# class definition is originally from
# [whisper example](https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/notebooks/LibriSpeech.ipynb#L56)  # noqa: B950
class LibriSpeech(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        data_name: str = "test-clean",
        chunk_length: int = 30,
        n_mels: int = 80,
        device: str | torch.device = "cpu",
    ) -> None:
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=dataset_dir,
            url=data_name,
            download=True,
        )
        self.device = device
        self.chunk_length = chunk_length
        self.n_mels = n_mels
        self.SAMPLE_RATE = 16000  # Fixed param

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, str]:
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == self.SAMPLE_RATE
        audio = whisper.pad_or_trim(
            audio.flatten(), length=self.SAMPLE_RATE * self.chunk_length
        ).to(self.device)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)

        return (mel, text)


def prepare_cache_sample(
    args: argparse.Namespace,
    model_dims: ModelDimensions,
    dtype: torch.dtype = torch.float32,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:

    def create_cache_sample(
        size: list[int],
        n_text_layer: int,
        cache_prefix: str,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        return {
            f"{cache_prefix}_cache_{i}": torch.zeros(*size, dtype=dtype)
            for i in range(n_text_layer)
        }

    # cache sizes
    n_head = model_dims.n_text_head
    n_state_per_head = model_dims.n_text_state // model_dims.n_text_head
    n_layer = model_dims.n_text_layer
    cross_kv_cache_size = [
        args.batch_size,
        n_head,
        model_dims.n_audio_ctx,
        n_state_per_head,
    ]

    cross_k_cache_dict = create_cache_sample(
        cross_kv_cache_size, n_layer, "cross_k", dtype
    )
    cross_v_cache_dict = create_cache_sample(
        cross_kv_cache_size, n_layer, "cross_v", dtype
    )

    return cross_k_cache_dict, cross_v_cache_dict


def kv_cache_from_dict(
    sample_d: dict[str, torch.Tensor],
    n_text_layer: int,
    cache_prefix: str,  # self or cross
) -> list[tuple[torch.Tensor, torch.Tensor]]:

    kv_cache = [
        (
            sample_d[f"{cache_prefix}_k_cache_{i}"],
            sample_d[f"{cache_prefix}_v_cache_{i}"],
        )
        for i in range(n_text_layer)
    ]

    return kv_cache


def calc_wer(
    hypotheses: list[str], references: list[str], print_texts: bool = True
) -> float:
    normalizer = EnglishTextNormalizer()
    hypotheses = [normalizer(text) for text in hypotheses]
    references = [normalizer(text) for text in references]

    if print_texts:
        print("\n############ Outputs #############")
        print("\nNormalized references and hypotheses (first 3 samples):")
        for index, (reference, hypothesis) in enumerate(
            zip(references[:3], hypotheses[:3])
        ):
            print(
                f"[{index}]\n"
                f"  Reference : {reference}\n"
                f"  Hypothesis: {hypothesis}"
            )
        print("\n##################################")

    return jiwer.wer(references, hypotheses)
