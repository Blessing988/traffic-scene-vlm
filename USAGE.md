# Detailed Usage Guide

This document provides comprehensive usage instructions for the AI City Track 2 project.

---

## Table of Contents

1. [Dataset Preparation](#dataset-preparation)
2. [Training with LLaMA Factory](#training-with-llama-factory)
3. [Inference Examples](#inference-examples)
4. [Output Formats](#output-formats)
5. [Advanced Configuration](#advanced-configuration)

---

## Dataset Preparation

### WTS Dataset Structure

The WTS (What-is-This-Scene) dataset contains traffic surveillance videos with annotations for pedestrian and vehicle behavior.

#### Input Structure
```
updated_wts_dataset/
├── annotations/
│   ├── caption/
│   │   ├── train/
│   │   │   ├── normal_trimmed/      # Normal behavior videos
│   │   │   └── [scenario_folders]/  # Anomaly videos
│   │   └── val/
│   │       ├── normal_trimmed/
│   │       └── [scenario_folders]/
│   └── bbox_annotated/
│       ├── pedestrian/
│       └── vehicle/
└── videos/
    ├── train/
    └── val/
```

### Processing Steps

The `scripts/data/prepare_wts_dataset.py` script performs three main operations:

#### 1. Video Segment Extraction
Extracts video clips based on event timestamps:
```bash
python scripts/data/prepare_wts_dataset.py /path/to/base_dir --split val
```

#### 2. Frame Extraction for Bounding Boxes
Extracts specific frames containing annotated bounding boxes:
```bash
python scripts/data/prepare_wts_dataset.py /path/to/base_dir --split val \
    --skip_video_processing
```

#### 3. Image Grounding Annotations
Creates annotations with normalized bounding box coordinates:
```bash
python scripts/data/prepare_wts_dataset.py /path/to/base_dir --split val \
    --skip_video_processing \
    --skip_frame_extraction
```

### Output Format (JSONL)

The processed annotations are saved in JSONL format:

**Video Annotations:**
```json
{"video": ["videos/scene/overhead_view/video_seg_0.mp4"], "conversations": [{"from": "human", "value": "<video>\nDescribe the pedestrian and vehicle behavior in this segment."}, {"from": "gpt", "value": "The pedestrian is crossing...\nThe vehicle is approaching..."}]}
```

**Image Annotations:**
```json
{"image": ["images/pedestrian/frame.jpg"], "conversations": [{"from": "human", "value": "<image>\nDescribe the pedestrian [0, 100, 200, 300] and vehicle [100, 200, 150, 100] behavior in this image"}, {"from": "gpt", "value": "Caption text here"}]}
```

---

## Training with LLaMA Factory

### Prerequisites

1. Install LLaMA Factory:
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

2. Register your dataset in `data/dataset_info.json`:
```json
{
  "aicity_track2": {
    "file_name": "/path/to/annotations_video_overhead_val_anomaly.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "videos": "video"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  },
  "aicity_track2_vqa_external": {
    "file_name": "/path/to/vqa_annotations.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "videos": "video"
    }
  }
}
```

### Training Commands

#### Qwen2.5-VL (Captioning)
```bash
# 10 epochs, LoRA rank 8
llamafactory-cli train configs/qwen2_5vl_lora_sft.yaml
```

#### MiniCPM-V (VQA)
```bash
# 5 epochs, LoRA rank 8
llamafactory-cli train configs/minicpm_V_lora_sft.yaml
```

#### InternVL3 (VQA)
```bash
# 5 epochs, with flash attention
llamafactory-cli train configs/intern_vl_lora_sft.yaml
```

#### LLaVA-NeXT-Video (VQA)
```bash
llamafactory-cli train configs/llava_video_next_lora_sft.yaml
```

### Merging LoRA Weights

After training, merge LoRA adapters with base model:
```bash
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter_name_or_path saves/qwen2_5vl-7b/lora/sft_10_epochs \
    --export_dir output/qwen2_5vl_merged \
    --export_legacy_format false
```

---

## Inference Examples

### Subtask 1: Video Captioning

#### Qwen2.5-VL

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# Load model with LoRA adapter
base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
adapter_path = "saves/qwen2_5vl-7b/lora/sft_10_epochs"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

processor = AutoProcessor.from_pretrained(base_model_path)

# Prepare input
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": frames},  # List of PIL images
            {"type": "text", "text": "Describe the pedestrian and vehicle behavior in this segment."}
        ]
    }
]

# Generate
text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_prompt], videos=[frames], return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

#### MiniCPM-V

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "output/minicpm_V_lora_sft"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# frames: list of PIL images (resized to 448x448)
msgs = [{'role': 'user', 'content': frames + ["Describe the pedestrian and vehicle behavior."]}]

response = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    max_slice_nums=2
)
```

### Subtask 2: Visual Question Answering

```python
# Build multiple-choice prompt
question = (
    "Answer the multiple choice question about the pedestrian in the scene.\n"
    f"Question: {conv['question']}\n"
    "a) Option A\n"
    "b) Option B\n"
    "c) Option C"
)

# Run inference (same as captioning)
response = model.generate(...)  # Returns "a", "b", or "c"
```

---

## Output Formats

### Subtask 1: Captioning Output

```json
{
  "scenario_001": [
    {
      "labels": ["0"],
      "caption_pedestrian": "A pedestrian is walking on the sidewalk, approaching the crosswalk. They stop at the curb and look both ways before crossing.",
      "caption_vehicle": "A silver sedan is traveling eastbound on the main road. The vehicle slows down as it approaches the intersection.",
      "start_time": "2.5",
      "end_time": "8.0"
    }
  ],
  "scenario_002": [...]
}
```

### Subtask 2: VQA Output

```json
[
  {"id": "q_001", "correct": "a"},
  {"id": "q_002", "correct": "c"},
  {"id": "q_003", "correct": "b"}
]
```

---

## Advanced Configuration

### Memory Optimization

For limited GPU memory, use 4-bit quantization:

```python
from transformers import BitsAndBytesConfig

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_cfg,
    device_map="auto"
)
```

### Frame Extraction Settings

Adjust frame sampling in inference scripts:

```python
def extract_video_frames_simple(video_path, start_time, end_time, max_frames=8):
    # max_frames: Number of frames to sample uniformly
    # Increase for longer videos, decrease for memory savings
```

### Batch Processing

For faster inference on multiple videos:

```python
# Process in batches
batch_size = 4
for i in range(0, len(videos), batch_size):
    batch = videos[i:i+batch_size]
    # Process batch...
```

---

## Troubleshooting

### Video Codec Issues

If OpenCV fails to read videos:
```bash
# Install additional codecs
pip install opencv-python-headless
# or
sudo apt-get install ffmpeg libavcodec-extra
```

### CUDA Out of Memory

1. Reduce batch size in YAML config
2. Use gradient checkpointing:
   ```yaml
   gradient_checkpointing: true
   ```
3. Enable 4-bit quantization during inference

### Slow Inference

1. Use `torch.compile`:
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```
2. Enable flash attention:
   ```yaml
   flash_attn: auto
   ```
