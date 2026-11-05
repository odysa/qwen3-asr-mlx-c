# Third-party notices

The audio front-end and tokenizer contain adapted code from
[antirez/qwen-asr](https://github.com/antirez/qwen-asr), pinned in the source
headers to commit `924694251d9e0f18e5d86bbd06aa3ab5f870002d`. The adapted files
retain the upstream MIT license notice. The full MIT text is also present in
`src/audio.c`, `src/tokenizer.c`, and `include/tokenizer.h`.

The MLX C dependency is built from the pinned commit recorded in
`deps/MLX_C_PIN.json` and retains its upstream license at `deps/mlx-c/LICENSE`.

## Model graph and model repository sources

The model graph reference used for development is MLX Audio at commit
`7a0986280fe295f4cf40f64164bb23abed416a22`, from Prince Canuma and contributors,
under the MIT License:
https://github.com/Blaizzy/mlx-audio/tree/7a0986280fe295f4cf40f64164bb23abed416a22
The corresponding license text is retained at `licenses/MLX-AUDIO-MIT.txt`.

The official Qwen3-ASR source reference is commit
`7c6daf77a2421100f5fb066495372c00129d39ff`, from Qwen/Alibaba/Hugging Face,
under Apache License 2.0:
https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff
The corresponding license text is retained at
`licenses/QWEN3-ASR-APACHE-2.0.txt`.
