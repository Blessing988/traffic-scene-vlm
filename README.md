# 🚗 Traffic Scene Understanding with Vision-Language Models

> **Multimodal Vision-Language Models for Video Captioning and Visual Question Answering in Traffic Scenes**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![LLaMA Factory](https://img.shields.io/badge/LLaMA%20Factory-LoRA-green.svg)](https://github.com/hiyouga/LLaMA-Factory)

---

## 📋 Overview

This repository provides a framework for **traffic scene understanding** using fine-tuned Vision-Language Models (VLMs). The project focuses on analyzing pedestrian and vehicle behavior in traffic scenes through:

- **Video Captioning**: Generate dual captions describing pedestrian and vehicle behavior in traffic video segments
- **Visual Question Answering (VQA)**: Answer multiple-choice questions about pedestrian behavior in traffic scenes

We fine-tune multiple state-of-the-art Vision-Language Models using **LoRA adapters** via [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) on the **WTS (What-is-This-Scene) Dataset**.

---

## ✨ Features

### Supported Models

| Model | Base Model | Captioning | VQA | Parameters |
|-------|------------|:----------:|:---:|------------|
| **Qwen2.5-VL** | `Qwen/Qwen2.5-VL-7B-Instruct` | ✅ | ✅ | 7B |
| **MiniCPM-V** | `openbmb/MiniCPM-V-2_6` | ✅ | ✅ | 2.4B |
| **InternVL3** | `OpenGVLab/InternVL3-2B-hf` | ✅ | ✅ | 2B |
| **LLaVA-NeXT-Video** | `llava-hf/LLaVA-NeXT-Video-7B-hf` | ✅ | ✅ | 7B |
| **Video-LLaVA** | `LanguageBind/Video-LLaVA-7B-hf` | ✅ | ✅ | 7B |

### Key Capabilities

- 🎬 **Video Understanding**: Process traffic surveillance videos from overhead and vehicle views
- 🚶 **Pedestrian Analysis**: Describe pedestrian behavior, actions, and interactions
- 🚗 **Vehicle Analysis**: Capture vehicle movements, turning patterns, and traffic flow
- ❓ **Visual QA**: Answer multiple-choice questions about traffic scenes
- ⚡ **Efficient Training**: LoRA fine-tuning for memory-efficient adaptation

---

## 📁 Project Structure

```
AI-CITY-Track2/
├── 📄 README.md                              # This file
├── 📄 requirements.txt                       # Python dependencies
├── 📄 USAGE.md                               # Detailed usage guide
├── 📄 LICENSE                                # MIT License
├── 📄 .gitignore                             # Git ignore rules
│
├── 📁 scripts/
│   ├── 📁 data/                              # Dataset preparation
│   │   └── prepare_wts_dataset.py            # WTS dataset processing
│   │
│   └── 📁 inference/                         # Model inference scripts
│       ├── prepare_subtask1.py               # Qwen2.5-VL captioning
│       ├── prepare_subtask1_internvl3.py     # InternVL3 captioning
│       ├── prepare_subtask1_minicpm_v.py     # MiniCPM-V captioning
│       ├── prepare_subtask1_llava_video_next.py  # LLaVA-NeXT-Video captioning
│       ├── prepare_subtask2.py               # Qwen2.5-VL VQA
│       ├── prepare_subtask2_minicpm_v.py     # MiniCPM-V VQA
│       └── prepare_subtask2_llava_video_next.py  # LLaVA-NeXT-Video VQA
│
├── 📁 configs/                               # LLaMA Factory training configs
│   ├── qwen2_5vl_lora_sft.yaml               # Qwen2.5-VL config
│   ├── intern_vl_lora_sft.yaml               # InternVL3 config
│   ├── minicpm_V_lora_sft.yaml               # MiniCPM-V config
│   ├── llava_video_next_lora_sft.yaml        # LLaVA-NeXT-Video config
│   └── video_llava_lora_sft.yaml             # Video-LLaVA config
│
└── 📁 outputs/                               # Generated predictions
    └── test_data_predictions_*.json          # Model predictions
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ VRAM recommended (for 7B models)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/traffic-scene-vlm.git
cd traffic-scene-vlm
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install LLaMA Factory** (for training)
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
```

---

## 📦 Dataset Preparation

The **WTS (Woven Traffic Safety) Dataset** must be prepared before training or inference.
The dataset can be downloaded from [here](https://woven-visionai.github.io/wts-dataset-homepage/).
### Dataset Structure

```
updated_wts_dataset/
├── annotations/
│   ├── caption/
│   │   ├── train/
│   │   └── val/
│   └── bbox_annotated/
│       ├── pedestrian/
│       └── vehicle/
└── videos/
    ├── train/
    └── val/
```

### Prepare Dataset

```bash
python scripts/data/prepare_wts_dataset.py /path/to/base_project_dir \
    --split val \
    --skip_frame_extraction  # Optional: skip if not needed
```

**Options:**
- `--split`: Dataset split to process (`train` or `val`)
- `--skip_video_processing`: Skip video segment extraction
- `--skip_frame_extraction`: Skip frame extraction for bounding boxes
- `--skip_image_grounding`: Skip image grounding annotation

**Output:** Processed dataset saved to `Foundation_models_in_Transportation_v2/VideoLLaMA3/wts_dataset_v2/`

---

## 🎓 Training

We use [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for efficient LoRA fine-tuning.

### Training Configuration

All models use similar hyperparameters:

| Parameter | Value |
|-----------|-------|
| Fine-tuning Type | LoRA |
| LoRA Rank | 8 |
| LoRA Target | all |
| Learning Rate | 1e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.1 |
| Batch Size | 2 |
| Gradient Accumulation | 8 |
| Epochs | 5-10 |
| Precision | bf16 |

### Train a Model

```bash
# Navigate to LLaMA Factory directory
cd LLaMA-Factory

# Train Qwen2.5-VL for captioning
llamafactory-cli train ../configs/qwen2_5vl_lora_sft.yaml

# Train MiniCPM-V for VQA
llamafactory-cli train ../configs/minicpm_V_lora_sft.yaml

# Train InternVL3
llamafactory-cli train ../configs/intern_vl_lora_sft.yaml

# Train LLaVA-NeXT-Video
llamafactory-cli train ../configs/llava_video_next_lora_sft.yaml
```

### Custom Dataset Registration

Add your dataset to LLaMA Factory's `data/dataset_info.json`:

```json
{
  "wts_traffic": {
    "file_name": "path/to/your/annotations.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "videos": "video"
    }
  }
}
```

---

## 🔮 Inference

### Video Captioning

Generate pedestrian and vehicle behavior descriptions from traffic videos.

```bash
# Using Qwen2.5-VL
python scripts/inference/prepare_subtask1.py

# Using MiniCPM-V
python scripts/inference/prepare_subtask1_minicpm_v.py

# Using LLaVA-NeXT-Video
python scripts/inference/prepare_subtask1_llava_video_next.py
```

**Output Format:**
```json
{
  "scene_id": [
    {
      "labels": ["0"],
      "caption_pedestrian": "A pedestrian is crossing...",
      "caption_vehicle": "A vehicle is approaching...",
      "start_time": "0.0",
      "end_time": "5.0"
    }
  ]
}
```

### Visual Question Answering

Answer multiple-choice questions about pedestrian behavior.

```bash
# Using Qwen2.5-VL
python scripts/inference/prepare_subtask2.py

# Using MiniCPM-V
python scripts/inference/prepare_subtask2_minicpm_v.py

# Using LLaVA-NeXT-Video
python scripts/inference/prepare_subtask2_llava_video_next.py
```

**Output Format:**
```json
[
  {"id": "question_001", "correct": "a"},
  {"id": "question_002", "correct": "b"}
]
```

---

## ⚙️ Configuration

### Model Paths

Update these paths in the inference scripts:

```python
# Base model path
base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

# LoRA adapter path (trained weights)
adapter_path = "saves/qwen2_5vl-7b/lora/sft_10_epochs"
```

### Dataset Paths

```python
CAPTION_DIR = '/path/to/wts_dataset/SubTask1-Caption/'
VQA_DIR = '/path/to/wts_dataset/SubTask2-VQA'
```

---

## 📊 Model Outputs

Predictions are saved with timestamps:

| Task | Output Pattern |
|------|----------------|
| Captioning | `test_data_predictions_subtask1_*_<model>_<timestamp>.json` |
| VQA | `test_data_predictions_subtask2_*_<model>_<timestamp>.json` |

---

## 🔧 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```python
# Use 4-bit quantization
from transformers import BitsAndBytesConfig
quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
```

**Video Processing Errors**
- Ensure videos are in MP4 format
- Check video codecs are compatible with OpenCV
- Verify video paths exist

**LoRA Adapter Not Loading**
- Verify adapter path matches training output directory
- Ensure PEFT library is installed: `pip install peft`

---

## 📚 References

- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) - Training framework
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - Vision-Language Model
- [InternVL](https://github.com/OpenGVLab/InternVL) - Vision-Language Model
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) - Efficient VLM
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) - Video understanding

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- WTS Dataset creators
- Open-source VLM communities
- LLaMA Factory team

---

<p align="center">
  <b>Traffic Scene Understanding with Vision-Language Models</b><br>
  Video Captioning & Visual Question Answering
</p>
