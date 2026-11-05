"""Local wav2vec2 CTC inference using MLX Audio's encoder implementation."""
import json
from pathlib import Path


def decode_ctc(values, id_to_token, blank):
    tokens, previous = [], None
    for value in values:
        if value != previous and value != blank:
            if value not in id_to_token:
                raise ValueError("CTC output is outside vocabulary")
            tokens.append(id_to_token[value])
        previous = value
    special = {"<pad>", "<s>", "</s>"}
    return "".join(t for t in tokens if t not in special).replace("|", " ").strip()


def validate_config(config, processor, tokenizer, vocab, weight_shape, bias_shape):
    if config.get("model_type") != "wav2vec2":
        raise ValueError("wav2vec2-ctc requires a wav2vec2 model directory")
    if processor.get("sampling_rate") != 16000 or processor.get("do_normalize") is not True:
        raise ValueError("unsupported wav2vec2 preprocessing configuration")
    if tokenizer.get("do_lower_case", False) is not False:
        raise ValueError("unsupported tokenizer case conversion")
    if tokenizer.get("word_delimiter_token", "|") != "|" or "|" not in vocab:
        raise ValueError("unsupported CTC word delimiter")
    if any(tokenizer.get(key, default) != default for key, default in
           (("pad_token", "<pad>"), ("bos_token", "<s>"), ("eos_token", "</s>"))):
        raise ValueError("unsupported tokenizer special tokens")
    size, hidden, blank = config.get("vocab_size"), config.get("hidden_size"), config.get("pad_token_id")
    if type(size) is not int or size <= 0 or type(hidden) is not int or hidden <= 0 or type(blank) is not int:
        raise ValueError("invalid CTC dimensions or blank token")
    if not isinstance(vocab, dict) or any(not isinstance(k, str) or type(v) is not int for k, v in vocab.items()):
        raise ValueError("invalid CTC vocabulary")
    if set(vocab.values()) != set(range(size)) or vocab.get("<pad>") != blank:
        raise ValueError("invalid CTC vocabulary or blank token")
    if tuple(weight_shape) != (size, hidden) or tuple(bias_shape) != (size,):
        raise ValueError("CTC head dimensions do not match configuration")
    return {value: key for key, value in vocab.items()}

def transcribe(model_dir: Path, audio_path: Path) -> str:
    import mlx.core as mx
    from mlx_audio.stt.models.wav2vec.wav2vec import ModelConfig, Wav2Vec2Model
    from mlx_audio.stt.utils import load_audio

    config = json.loads((model_dir / "config.json").read_text())
    processor = json.loads((model_dir / "preprocessor_config.json").read_text())
    tokenizer = json.loads((model_dir / "tokenizer_config.json").read_text())
    vocab = json.loads((model_dir / "vocab.json").read_text())
    weights = mx.load(str(model_dir / "model.safetensors"))
    if "lm_head.weight" not in weights or "lm_head.bias" not in weights:
        raise ValueError("wav2vec2-ctc needs fine-tuned CTC head weights")
    id_to_token = validate_config(config, processor, tokenizer, vocab,
                                  weights["lm_head.weight"].shape, weights["lm_head.bias"].shape)
    model = Wav2Vec2Model(ModelConfig.from_dict(config))
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    audio = load_audio(str(audio_path), sr=16000)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("audio decoder returned empty or non-mono data")
    audio = (audio - mx.mean(audio)) / mx.sqrt(mx.var(audio) + 1e-7)
    hidden = model(audio[None, :]).last_hidden_state
    logits = hidden @ weights["lm_head.weight"].T + weights["lm_head.bias"]
    ids = mx.argmax(logits, axis=-1)
    mx.eval(ids)
    return decode_ctc(ids.tolist()[0], id_to_token, config["pad_token_id"])
