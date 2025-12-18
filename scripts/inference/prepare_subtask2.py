from collections import Counter
from PIL import Image
import sys
import os
import random
import json
import cv2
import tempfile
import torch
import numpy as np
import time
import warnings
import re
from tqdm.auto import tqdm
from pathlib import Path


# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("🔧 Loading Qwen2.5-VL model for VQA...")

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    
    base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_path = "saves/qwen2_5vl-7b/lora/sft_vqa_external"
    
    print("Loading base model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)


# Dataset paths - same as your VideoLLaMA3 script
CAPTION_DIR = '/mmfs1/projects/armstrong.aboah/updated_wts_dataset/test_data/SubTask1-Caption/'
VQA_DIR = '/mmfs1/projects/armstrong.aboah/updated_wts_dataset/test_data/SubTask2-VQA'

caption_annotations = f'{CAPTION_DIR}/WTS_DATASET_PUBLIC_TEST/annotations/caption/test/public_challenge'
caption_annotations_normal = f'{caption_annotations}/normal_trimmed'
caption_annotations_external = f'{CAPTION_DIR}/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/annotations/caption/test/public_challenge'

VQA_json = f'{VQA_DIR}/WTS_VQA_PUBLIC_TEST.json'

VIEW = "overhead_view"

video_dir = f'{CAPTION_DIR}/WTS_DATASET_PUBLIC_TEST/videos/test/public'
video_dir_normal = f'{video_dir}/normal_trimmed'
video_dir_external = f'{CAPTION_DIR}/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public'

# Performance tracking
total_questions = 0
successful_answers = 0
failed_answers = 0


def prefix_first_four_segments(path: str) -> str:
    """
    Return the first four underscore-separated fields of the filename
    (without its extension or directory).
    Same function as your VideoLLaMA3 script.
    """
    fname = os.path.splitext(os.path.basename(path))[0]  # strip dir + .mp4
    return "_".join(fname.split("_")[:4])


def extract_video_frames_simple(video_path,
                                start_time=None,
                                end_time=None,
                                max_frames=8):
    """
    Return a list of ≤ max_frames PIL images sampled uniformly.
    If start/end times are None, the whole clip is used.
    Returns None if the file cannot be opened or no frames could be read.
    """
    # ─── 1. open video ────────────────────────────────────────────────────
    if not os.path.isfile(video_path):
        print(f"[error] file not found → {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] cannot open video → {video_path}")
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25            # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames == 0:
        print(f"[warn] OpenCV sees 0 frames → {video_path}")
        cap.release()
        return None

    # ─── 2. determine span in frame indices ──────────────────────────────
    if start_time is None or end_time is None:
        start_idx, end_idx = 0, total_frames - 1
    else:
        start_idx = max(0, int(start_time * fps))
        end_idx   = min(total_frames - 1, int(end_time * fps))
        if start_idx >= end_idx:                          # bad window → full clip
            start_idx, end_idx = 0, total_frames - 1

    # ─── 3. choose indices uniformly (handles short clips) ───────────────
    span = end_idx - start_idx + 1
    num  = min(max_frames, span)
    indices = np.linspace(start_idx, end_idx, num=num, dtype=int)

    # ─── 4. read frames ──────────────────────────────────────────────────
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames if frames else None


def vqa_inference(frames, question):
    """VQA inference with Qwen2.5-VL"""
    try:
        if not frames or len(frames) == 0:
            return None
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        text_prompt = processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            videos=[frames],
            padding=True,
            return_tensors="pt"
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Clean inputs
        clean_inputs = {}
        for k, v in inputs.items():
            if k in ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']:
                clean_inputs[k] = v
        
        with torch.no_grad():
            output_ids = model.generate(
                **clean_inputs,
                max_new_tokens=100,  # Shorter for VQA answers
                do_sample=True,
                temperature=0.7,  # Lower temperature for more focused answers
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        generated_ids = output_ids[0][len(clean_inputs["input_ids"][0]):]
        response = processor.tokenizer.decode(
            generated_ids, 
            skip_special_tokens=True
        ).strip()
        
        return response
        
    except Exception as e:
        print(f"VQA inference error: {e}")
        return None


def process_vqa(video_path, full_question, start_time=None, end_time=None):
    """
    Process VQA question - same interface as your VideoLLaMA3 script
    """
    global total_questions, successful_answers, failed_answers
    
    total_questions += 1
    
    frames = extract_video_frames_simple(video_path, start_time, end_time, max_frames=8)
    
    if not frames:
        print('Video path', video_path)
        print(f"[warning] no frames extracted from {video_path}; skipping.")
        failed_answers += 1
        return random.choice(["a", "b"])  # Fallback same as original
    
    response = vqa_inference(frames, full_question)
    if response:
        successful_answers += 1
        return response
    else:
        failed_answers += 1
        return random.choice(["a", "b"])  # Fallback same as original


# ----------------------------------------------------------------------
# helper ─ choose one clip to answer the question
# ----------------------------------------------------------------------
def build_video_path(video_name: str) -> str:
    """
    Map a raw file-name from the JSON to an **absolute** path on disk.
    Adjust the three directory constants to match your system.
    """
    if video_name.startswith("video"):               # external set
        return f"{video_dir_external}/{video_name}"
    if "vehicle_view" in video_name:                 # dash-cam
        scene = video_name.split("_vehicle_view.mp4")[0]
        return f"{video_dir}/{scene}/vehicle_view/{video_name}"
    if "normal" in video_name:                       # overhead (normal_trimmed)
        scene = video_name[:-4]                      # strip .mp4
        return f"{video_dir_normal}/{scene}/overhead_view/{video_name}"
    # fallback → treat as overhead, using first four tokens of the name
    scene = "_".join(video_name.split("_")[:4])
    return f"{video_dir}/{scene}/overhead_view/{video_name}"

# # Main VQA processing - same structure as your VideoLLaMA3 script

# Performance summary

# ----------------------------------------------------------------------
# main inference loop
# ----------------------------------------------------------------------
vqa_pred_list = []
seen_ids      = set()          # ⚠ guarantees no duplicate predictions

print("🎯 Processing VQA data...")
with open(VQA_json, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📊 Found {len(data):,} records in file")

for rec in tqdm(data, desc="records"):
    # any video in the list is valid for answering the questions.
    # prefer overhead if present, else first one.
    overheads = [v for v in rec["videos"] if "overhead" in v or "normal" in v]
    video_name = overheads[0] if overheads else rec["videos"][0]
    video_path = build_video_path(video_name)

    # -- iterate through *all* phases (or fall back to top-level) ----------
    phases = rec.get("event_phase", [rec])
    for phase in phases:
        start_time = float(phase.get("start_time", 0))
        end_time   = float(phase.get("end_time",   0))

        for conv in phase.get("conversations", []):
            cid = conv["id"]
            if cid in seen_ids:                      # should never happen
                continue
            seen_ids.add(cid)

            # -----------------------------------------------------------------
            # build the multiple-choice prompt
            # -----------------------------------------------------------------
            letters = ["a", "b", "c", "d", "e"]
            options = [f"{l}) {conv[l]}" for l in letters
                       if l in conv and conv[l]]

            # 🔧  never skip an ID — fabricate a dummy option list if needed
            if len(options) < 2:
                # keep at least the real answers that exist
                if not options and "a" in conv:
                    options.append(f"a) {conv['a']}")
                # pad with “N/A” so the prompt is syntactically valid
                while len(options) < 2:
                    fake_letter = letters[len(options)]
                    options.append(f"{fake_letter}) N/A")

            question = (
                "Answer the multiple choice question about the pedestrian "
                "in the scene.\n"
                f"Question: {conv['question']}\n" +
                "\n".join(options)
            )

            # -----------------------------------------------------------------
            # run your model
            # -----------------------------------------------------------------
            answer_text = process_vqa(
                video_path,
                question,
                start_time=start_time,
                end_time=end_time,
            ).strip()

            # map the model's textual answer back to a, b, c…
            option_letter = next(
                (l for l in letters if conv.get(l, "").strip() == answer_text),
                None
            ) or random.choice(letters[:3])          # fallback guess
            
            
            print('Full question', question)
            print('Answer', answer_text)
            print('Option', option_letter)

            vqa_pred_list.append({"id": cid, "correct": option_letter})

# ----------------------------------------------------------------------
# sanity check – should be exactly 19 624 unique IDs
# ----------------------------------------------------------------------
total        = len(vqa_pred_list)
unique_total = len(set(d["id"] for d in vqa_pred_list))
print(f"\n✅ predictions written  : {total:,}")
print(f"✅ distinct IDs covered : {unique_total:,}")

total_time = time.time()
success_rate = successful_answers / max(1, total_questions) * 100

print(f"\n📊 VQA Processing Summary:")
print(f"   Total questions: {total_questions}")
print(f"   Successful answers: {successful_answers}")
print(f"   Failed answers: {failed_answers}")
print(f"   Success rate: {success_rate:.1f}%")
print(f"   Total predictions: {len(vqa_pred_list)}")

# Save results - same format as your VideoLLaMA3 script
timestamp = time.strftime("%Y%m%d_%H%M%S")
out_file = Path(f"test_data_predictions_subtask2_vqa_qwen25vl_{timestamp}_fixed_vqa_rerun.json")

with out_file.open("w", encoding="utf-8") as f:
    json.dump(vqa_pred_list, f, ensure_ascii=False, indent=2)

print(f"💾 Saved to {out_file.resolve()}")

# Sample output for verification
if vqa_pred_list:
    print(f"\n📝 Sample predictions:")
    for i, pred in enumerate(vqa_pred_list[:3]):
        print(f"   {i+1}. ID: {pred['id']}, Answer: {pred['correct']}")

assert unique_total == 19_624,   "Some IDs are still missing!"
assert total == unique_total,    "You produced duplicate IDs!"

# Cleanup
try:
    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("🧹 Memory cleanup completed!")
except:
    pass
