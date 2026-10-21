

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, MODEL_ID  # noqa: E402


def verify_model(path):
    path = Path(path)
    required = ('config.json', 'tokenizer_config.json',
                'model.safetensors.index.json')
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f'Model download is incomplete; missing={missing}')
    import json
    import struct
    index=json.loads((path/'model.safetensors.index.json').read_text(encoding='utf-8'))
    names=sorted(set(index.get('weight_map',{}).values()))
    if not names:raise RuntimeError('The weight index contains no valid shards.')
    shards=[]
    for name in names:
        shard=path/name
        if not shard.resolve().is_relative_to(path.resolve()) or not shard.is_file():
            raise RuntimeError(f'The model is missing a weight shard: {name}')
        size=shard.stat().st_size
        try:
            with shard.open('rb') as handle:
                header_size=struct.unpack('<Q',handle.read(8))[0]
                if not 0<header_size<=min(size-8,100_000_000):raise ValueError('Invalid shard header length')
                header=json.loads(handle.read(header_size))
            tensors={key:value for key,value in header.items() if key!='__metadata__'}
            offsets=[value['data_offsets'] for value in tensors.values()]
            if not offsets or any(len(pair)!=2 or pair[0]<0 or pair[1]<pair[0] for pair in offsets):raise ValueError('Invalid shard data offsets')
            if 8+header_size+max(pair[1] for pair in offsets)!=size:raise ValueError('Shard length does not match the data offsets')
            expected={key for key,value in index['weight_map'].items() if value==name}
            if not expected.issubset(tensors):raise ValueError('The shard lacks tensors listed in the index')
        except (ValueError,KeyError,TypeError,struct.error) as exc:
            raise RuntimeError(f'The weight shard is damaged or incomplete: {name}; {exc}') from exc
        shards.append(shard)
    size_gb = sum(file.stat().st_size for file in path.rglob('*') if file.is_file()) / 1e9
    print(f'Model ready: {path}')
    print(f'Approximate total file size: {size_gb:.2f} GB; weight shards: {len(shards)}')


def download_from_modelscope(model_id, local_dir):
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError('modelscope is missing; run pip install -r requirements.txt first') from exc
    snapshot_download(model_id, local_dir=str(local_dir))


def download_from_huggingface(model_id, local_dir, endpoint=None):
    if endpoint:
        os.environ['HF_ENDPOINT'] = endpoint
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('huggingface_hub is missing; run pip install -r requirements.txt first') from exc
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
    )


def main():
    parser = argparse.ArgumentParser(description='Download the Qwen3.5-4B model (approximately 9.34GB)')
    parser.add_argument('--provider', choices=('modelscope', 'huggingface'),
                        default='modelscope')
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--output-dir', type=Path, default=MODEL_DIR)
    parser.add_argument('--hf-endpoint', default=None,
                        help='For example, https://hf-mirror.com')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.provider == 'modelscope':
        download_from_modelscope(args.model_id, args.output_dir)
    else:
        download_from_huggingface(
            args.model_id, args.output_dir, endpoint=args.hf_endpoint
        )
    verify_model(args.output_dir)


if __name__ == '__main__':
    main()
