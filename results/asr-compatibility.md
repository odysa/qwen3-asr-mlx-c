# Local ASR compatibility smoke tests

The generic adapter has run one short clip each for the checkpoints below on an Apple M1 Max, Python 3.14.3, MLX 0.32.4 and MLX Audio 0.6.3. The wav2vec2 CTC path is a dedicated adapter in this repository. Revisions are the recorded download targets. For the first seven rows, model files were not hashed at measurement time, so those receipts do not bind their bytes to the revisions. For the two additional rows below, the local model files, inputs and source were hashed before and after the runs and were unchanged. A successful text result indicates that inference returned; it does not prove transcription accuracy, a full language capability, realtime performance, or compatibility of other revisions. The example clips lack independently pinned reference transcripts.

| Model | Exact revision | Adapter | Example |
|---|---|---|---|
| [Whisper tiny MLX](https://huggingface.co/mlx-community/whisper-tiny-asr-fp16) | `77fa3f52b482ec80d086df55f893065a9baab172` | `mlx-audio` | English |
| [Parakeet TDT v3 MLX](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) | `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | `mlx-audio` | English |
| [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | `mlx-audio` | Mandarin |
| [Moonshine tiny](https://huggingface.co/UsefulSensors/moonshine-tiny) | `390624ed33d594443aa4aa221f5b9f283b545b5a` | `mlx-audio` | English |
| [SenseVoiceSmall MLX](https://huggingface.co/mlx-community/SenseVoiceSmall) | `8ddd966bd96243cff196422f81f0c5d955814792` | `mlx-audio` | Mandarin |
| [Fun-ASR Nano MLX](https://huggingface.co/mlx-community/Fun-ASR-Nano-2512) | `a7bc96fceaafce39ed6748e0c0fa9a9508b67f86` | `mlx-audio` | Mandarin |
| [wav2vec2 base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | `wav2vec2-ctc` | English |
| [Nemotron 3.5 streaming 0.6B 8-bit MLX](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit) | `7279359e4481b5e9e185a318bd618e429c6d86cd` | `mlx-audio` | English, whole-file inference |
| [Canary 1B v2 q8 MLX](https://huggingface.co/Mediform/canary-1b-v2-mlx-q8) | `0b6b32ee10f30c89e3ead7249bb636445e3019ee` | `mlx-audio` | English, whole-file inference |

For wav2vec2, text output matched the Hugging Face PyTorch `Wav2Vec2ForCTC` plus processor on the official English example and two English FLEURS clips. The reference loader warned that `wav2vec2.masked_spec_embed` was absent from the checkpoint; this parameter is relevant to training-time masking and the comparison ran in evaluation mode. Three matched outputs are a narrow parity check, not an accuracy evaluation. The two FLEURS samples have their own attribution requirements (see [attribution](quality/ATTRIBUTION.md)).

Checkpoint terms differ: for example, the original [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) uses a [custom model license](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE); do not assume it is Apache-2.0. Refer to each exact original and converted model card and dependency license. Other models listed by `list-asr` are upstream modules, not verified checkpoints.

The Nemotron and Canary tests used one English file each, Python 3.14.3, MLX Audio 0.6.3, and whole-file inference via `generate` (no live streaming or timing measured). Their separate local receipt seal SHA-256 is `11f59bd78c178de6d797356f696999b3e676783d0e3d1c1068d226b137727018`. The saved inputs, outputs and options were independently reviewed; no independent inference was run. The [Nemotron MLX conversion card](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit) declares NVIDIA Open Model License, whereas the current [original NVIDIA model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) identifies OpenMDW-1.1. Check the applicable terms for the exact original and conversion rather than assuming all revisions share one license. The Canary checkpoint card declares CC BY 4.0.
