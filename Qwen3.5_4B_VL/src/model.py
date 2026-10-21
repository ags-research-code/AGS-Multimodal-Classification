

import re
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import STAGE_DESCRIPTIONS  # noqa: E402


LABEL_ALIASES = {
    'growth': ('growth', 'growth stage', 'growth'),
    'stabilization': ('stabilization', 'stabilization stage', 'stabilization'),
    'disintegration': ('disintegration', 'disintegration stage', 'disintegration'),
}


class QwenFewShotClassifier:
    def __init__(self, model_path, min_pixels, max_pixels, offload_dir):
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                'A newer transformers version with Qwen3.5 support is missing. Run: '
                'pip install -r requirements.txt'
            ) from exc

        model_path = Path(model_path)
        if not (model_path / 'config.json').is_file():
            raise FileNotFoundError(
                f'The local model is incomplete: {model_path}. Run src/download_model.py first.'
            )
        offload_dir = Path(offload_dir)
        offload_dir.mkdir(parents=True, exist_ok=True)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            size={
                'shortest_edge': min_pixels,
                'longest_edge': max_pixels,
            },
            local_files_only=True,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype='auto',
            device_map='auto',
            low_cpu_mem_usage=True,
            local_files_only=True,
            offload_folder=offload_dir,
            attn_implementation='sdpa',
        )
        self.model.eval()

    @property
    def input_device(self):
        return self.model.device

    def classify(self, query_image, examples, max_new_tokens=24):
        messages = [{
            'role': 'system',
            'content': 'You are an AGS microscopic image stage classifier. Select only one of the following labels: growth, stabilization, or disintegration. Do not explain.\nMorphological criteria:\ngrowth: Sludge aggregates gradually develop initial granular morphology. Outlines are relatively irregular, edge undulations are pronounced, some regions remain relatively translucent, and the overall morphology is not fully mature.\nstabilization: Granules have relatively regular outlines, clear and intact boundaries, relatively compact overall structures, lower light transmission, and few free fragments, indicating more mature granular morphology.\ndisintegration: Granule outlines become irregular, with loose or locally damaged edges. Floc-like or filamentous structures and surrounding free fragments are visible, consistent with granule disintegration.\nAssess visible features including granule outlines, boundary integrity, light transmission, filamentous structures, and surrounding fragments. A single microscopic image cannot directly demonstrate temporal changes in particle size; use the example images to guide stage classification.',
        }]
        for example_image, example_label in examples:
            messages.append({
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': example_image},
                    {'type': 'text', 'text': 'This is a labeled example. Identify its class.'},
                ],
            })
            messages.append({'role': 'assistant', 'content': example_label})

        messages.append({
            'role': 'user',
            'content': [
                {'type': 'image', 'image': query_image},
                {
                    'type': 'text',
                    'text': 'Classify this query AGS microscopic image. Output only one label: growth, stabilization, or disintegration.',
                },
            ],
        })

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            enable_thinking=False,
        ).to(self.input_device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        new_tokens = generated[:, inputs['input_ids'].shape[1]:]
        raw_output = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return parse_label(raw_output), raw_output


def parse_label(text):
    
    normalized = str(text).strip().lower()
    matches = []
    for label, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(re.escape(alias.lower()), normalized):
                matches.append((match.start(), label))
    if not matches:
        return 'invalid'
    labels = {label for _, label in matches}
    if len(labels) == 1:
        return next(iter(labels))
    return 'invalid'
