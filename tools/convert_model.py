#!/usr/bin/env python3
"""Build an experimental checkpoint locally; no weights are distributed here."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import mlx.core as mx


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='BF16 model directory, or q8 directory for head8')
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--kind', choices=['q8', 'q4', 'head8'], required=True)
    args = parser.parse_args()
    src = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error('output directory already exists; refusing to overwrite it')
    original = mx.load(str(src / 'model.safetensors'))
    weights = dict(original)
    modules = []
    if args.kind in ('q8', 'q4'):
        bits = 8 if args.kind == 'q8' else 4
        parts = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
        for layer in range(28):
            for part in parts:
                key = f'thinker.model.layers.{layer}.{part}'
                value = original[key + '.weight']
                if value.ndim != 2 or value.dtype != mx.bfloat16 or value.shape[1] % 64:
                    raise ValueError(f'unexpected tensor format: {key}')
                packed, scales, biases = mx.quantize(value, group_size=64, bits=bits, mode='affine')
                mx.eval(packed, scales, biases)
                weights[key + '.weight'] = packed
                weights[key + '.scales'] = scales
                weights[key + '.biases'] = biases
                modules.append(key)
        assert len(modules) == 196
    else:
        value = original['thinker.model.embed_tokens.weight']
        if value.shape != (151936, 1024) or value.dtype != mx.bfloat16:
            raise ValueError('unexpected input embedding shape or dtype')
        packed, scales, biases = mx.quantize(value, group_size=64, bits=8, mode='affine')
        mx.eval(packed, scales, biases)
        for suffix, array in zip(('weight', 'scales', 'biases'), (packed, scales, biases)):
            weights['experimental.output_head.' + suffix] = array
        modules.append('experimental.output_head')
    output.mkdir(parents=True)
    mx.save_safetensors(str(output / 'model.safetensors'), weights)
    for name in ('config.json', 'vocab.json', 'merges.txt', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json', 'generation_config.json', 'chat_template.json'):
        file = src / name
        if file.is_file():
            shutil.copy2(file, output / name)
    receipt = {'kind': args.kind, 'source_sha256': digest(src / 'model.safetensors'), 'output_sha256': digest(output / 'model.safetensors'), 'modules': modules, 'note': 'experimental; not an accuracy claim'}
    (output / 'conversion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'modules'}, indent=2))


if __name__ == '__main__':
    main()
