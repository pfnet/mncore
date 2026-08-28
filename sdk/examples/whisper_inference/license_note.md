# whisper_inference 内で用いられる各構成要素の license について

## model について

本推論処理で用いる model は [MIT license](https://github.com/openai/whisper?tab=MIT-1-ov-file#) で配布されている [openai/whisper](https://github.com/openai/whisper) にて配布されている実装、学習済みの重みを用いている。本成果物には当該 model を install、load する処理が含まれている

## ソースコードについて

本ソースコード内には、[MIT license](https://github.com/openai/whisper?tab=MIT-1-ov-file#) で配布されている [openai/whisper](https://github.com/openai/whisper) にあるものを参考に作成した処理が含まれている。参考にした箇所は以下の通り

- [whisper/notebooks/LibriSpeech.ipynb](https://github.com/openai/whisper/blob/main/notebooks/LibriSpeech.ipynb)
  - torchauido.dataset.LIBRISPEECH dataset class の簡易 wrapper、推論処理、および model 出力を用いた Word Error Rate の算出方法
- [whisper/model.py](https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/whisper/model.py)
  - Whisper、AudioEncoder、TextDecoder、ResidualAttentionBlcok、MultiHeadAttention class の実装
- [whisper/decoding.py](https://github.com/openai/whisper/blob/c0d2f624c09dc18e709e37c2ad90c039a4eb72a2/whisper/decoding.py)
  - DecodingTask、PyTorchInference class 及び decode() 関数の実装
