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
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import av

from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

MODEL_DIR = "output/minicpm_V_lora_sft"       # <- your merged dir
PROC_SIZE = 448
MAX_NUM_FRAMES=4

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",        # or torch.bfloat16/float16
    device_map="auto",         # puts GPU-fit shards on GPU
    trust_remote_code=True,    # MiniCPM uses custom code
    attn_implementation="sdpa"
)
model.eval()                  # inference mode

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR,
                                          trust_remote_code=True)

model = torch.compile(model, mode="reduce-overhead")

# Dataset paths for ALL views
CAPTION_DIR = '/mmfs1/projects/armstrong.aboah/updated_wts_dataset/test_data/SubTask1-Caption/'
VQA_DIR = '/mmfs1/projects/armstrong.aboah/updated_wts_dataset/test_data/SubTask2-VQA'

# Overhead paths
caption_annotations = f'{CAPTION_DIR}/WTS_DATASET_PUBLIC_TEST/annotations/caption/test/public_challenge'
video_dir = f'{CAPTION_DIR}/WTS_DATASET_PUBLIC_TEST/videos/test/public'

# Normal paths
caption_annotations_normal = f'{caption_annotations}/normal_trimmed'
video_dir_normal = f'{video_dir}/normal_trimmed'

# External paths
caption_annotations_external = f'{CAPTION_DIR}/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/annotations/caption/test/public_challenge'
video_dir_external = f'{CAPTION_DIR}/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public'

VIEW = "overhead_view"

# Results storage
overhead_json = {}
normal_json = {}
external_json = {}

# Performance tracking
total_processed = 0
total_failed = 0
start_time = time.time()


def extract_video_frames_simple(video_path,
                                start_time=None,
                                end_time=None,
                                num_frames=8):
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
    num  = min(num_frames, span)
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
    frames = [img.resize((PROC_SIZE, PROC_SIZE)) for img in frames]
    return frames if frames else None


def optimized_inference(frames, prompt):
    msgs   = [{'role': 'user', 'content': frames + [prompt]}]

    out = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        max_slice_nums=2,     # keep if you need slicing
        use_image_id=False,
    )
    
    return out
    

def process_captions(video_path, start_time, end_time):
    """Unified caption processing"""
    global total_processed, total_failed
    
    frames = extract_video_frames_simple(video_path, start_time, end_time, num_frames=4)
    if not frames:
        print(f"[warning] no frames extracted from {video_path}; skipping.")
        total_failed += 1
        return None, None
    
    prompt = "Describe the pedestrian and vehicle behavior in this segment."
    response = optimized_inference(frames, prompt)
    
    if not response:
        total_failed += 1
        return None, None
    
    # Parse response - same logic as VideoLLaMA3
    res = response.split('\n')
    
    if len(res) == 1:
        pedestrian_caption = res[0]
        vehicle_caption = ""
    else:
        pedestrian_caption = res[0]
        vehicle_caption = res[1]
    
    total_processed += 1
    return pedestrian_caption, vehicle_caption


def process_overhead_videos():
    """Process overhead videos (original working script logic)"""
    print("\n🎯 Processing OVERHEAD videos...")
    
    scene_list = [s for s in os.listdir(caption_annotations) if not s.startswith('normal')]
    
    for scene in tqdm(scene_list, desc="Overhead scenes"):
        
        overhead_json[scene] = []
        
        annot_path = os.path.join(caption_annotations, scene, VIEW, f'{scene}_caption.json')
        
        if not os.path.exists(annot_path):
            continue
            
        with open(annot_path, 'r') as f:
            data = json.load(f)
            
        videos = data['overhead_videos']
        events = data['event_phase']
        
        for event in events:
            
            label = event['labels'][0]
            start_time = float(event['start_time'])
            end_time = float(event['end_time'])

            if end_time <= start_time:
                print(f"[skip] {video}  bad times: {start_time} ≥ {end_time}")
                continue       # or swap them, or set end_time = start_time + 1

            # Select video
            video = [video for video in videos if video.split('_')[-1][0] == label]
            if video:
                target_video = random.choice(video)
            else:
                target_video = random.choice(videos)
            
            video_path = os.path.join(video_dir, scene, VIEW, target_video)
            
            if not os.path.exists(video_path):
                continue
            
            pedestrian_caption, vehicle_caption = process_captions(video_path, start_time, end_time)
            
            if pedestrian_caption is None or vehicle_caption is None:
                continue
            
            results_dict = {
                "labels": [label],
                "caption_pedestrian": pedestrian_caption,
                "caption_vehicle": vehicle_caption,
                "start_time": str(start_time),
                "end_time": str(end_time)
            }
            
            overhead_json[scene].append(results_dict)


def process_normal_videos():
    """Process normal trimmed videos"""
    print("\n🎯 Processing NORMAL videos...")
    
    if not os.path.exists(caption_annotations_normal):
        print(f"⚠️  Normal annotations directory not found: {caption_annotations_normal}")
        return
    
    for scene in tqdm(os.listdir(caption_annotations_normal), desc="Normal scenes"):
        
        normal_json[scene] = []
        
        annot_path = os.path.join(caption_annotations_normal, scene, VIEW, f'{scene}_caption.json')
        
        if not os.path.exists(annot_path):
            continue
            
        with open(annot_path, 'r') as f:
            data = json.load(f)
            
        videos = data['overhead_videos']
        events = data['event_phase']
        
        for event in events:
            
            label = event['labels'][0]
            start_time = float(event['start_time'])
            end_time = float(event['end_time'])
            
            if end_time <= start_time:
                print(f"[skip] {video}  bad times: {start_time} ≥ {end_time}")
                continue       # or swap them, or set end_time = start_time + 1

            video = [video for video in videos if video.split('_')[-1][0] == label]
            if video:
                target_video = random.choice(video)
            else:
                target_video = random.choice(videos)
            
            video_path = os.path.join(video_dir_normal, scene, VIEW, target_video)
            
            if not os.path.exists(video_path):
                continue
            
            pedestrian_caption, vehicle_caption = process_captions(video_path, start_time, end_time)
            
            if pedestrian_caption is None or vehicle_caption is None:
                continue
            
            results_dict = {
                "labels": [label],
                "caption_pedestrian": pedestrian_caption,
                "caption_vehicle": vehicle_caption,
                "start_time": str(start_time),
                "end_time": str(end_time)
            }
            
            normal_json[scene].append(results_dict)


def process_external_videos():
    """Process external BDD_PC_5K videos"""
    print("\n🎯 Processing EXTERNAL videos...")
    
    if not os.path.exists(caption_annotations_external):
        print(f"⚠️  External annotations directory not found: {caption_annotations_external}")
        return
    
    for caption in tqdm(os.listdir(caption_annotations_external), desc="External videos"):
        scene = caption.split('_caption.json')[0]
        external_json[scene] = []
        
        annot_path = os.path.join(caption_annotations_external, caption)
        with open(annot_path, 'r') as f:
            data = json.load(f)
        
        video = data['video_name']
        events = data['event_phase']
        
        for event in events:
            
            label = event['labels'][0]
            start_time = float(event['start_time'])
            end_time = float(event['end_time'])

            if end_time <= start_time:
                print(f"[skip] {video}  bad times: {start_time} ≥ {end_time}")
                continue       # or swap them, or set end_time = start_time + 1

            video_path = os.path.join(video_dir_external, video)
            
            if not os.path.exists(video_path):
                continue
            
            pedestrian_caption, vehicle_caption = process_captions(video_path, start_time, end_time)
            
            if pedestrian_caption is None or vehicle_caption is None:
                continue
            
            results_dict = {
                "labels": [label],
                "caption_pedestrian": pedestrian_caption,
                "caption_vehicle": vehicle_caption,
                "start_time": str(start_time),
                "end_time": str(end_time)
            }
            
            external_json[scene].append(results_dict)


def save_results():
    """Save all results and create combined JSON"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save individual JSONs
    overhead_file = Path(f"test_data_predictions_subtask1_overhead_views_minicpm_v_v2_{timestamp}_10_epochs.json")
    normal_file = Path(f"test_data_predictions_subtask1_normal_views_minicpm_v_v2_{timestamp}_10_epochs.json")
    external_file = Path(f"test_data_predictions_subtask1_external_views_minicpm_v_v2_{timestamp}_10_epochs.json")
    
    if overhead_json:
        with overhead_file.open("w", encoding="utf-8") as f:
            json.dump(overhead_json, f, ensure_ascii=False, indent=2)
        print(f"💾 Overhead results saved: {overhead_file}")
    
    if normal_json:
        with normal_file.open("w", encoding="utf-8") as f:
            json.dump(normal_json, f, ensure_ascii=False, indent=2)
        print(f"💾 Normal results saved: {normal_file}")
    
    if external_json:
        with external_file.open("w", encoding="utf-8") as f:
            json.dump(external_json, f, ensure_ascii=False, indent=2)
        print(f"💾 External results saved: {external_file}")
    
    # Create combined JSON
    combined_json = {}
    combined_json.update(overhead_json)
    combined_json.update(normal_json)
    combined_json.update(external_json)
    
    if combined_json:
        combined_file = Path(f"test_data_predictions_subtask1_ALL_VIEWS_minicpm_v_v2_{timestamp}_10_epochs.json")
        with combined_file.open("w", encoding="utf-8") as f:
            json.dump(combined_json, f, ensure_ascii=False, indent=2)
        print(f"🎉 COMBINED results saved: {combined_file}")
        
        return combined_file
    
    return None


# Main execution
def main():
    print("🚀 Starting comprehensive processing of ALL video types...")
    
    # Process all three types
    process_overhead_videos()
    process_normal_videos() 
    process_external_videos()
    
    # Performance summary
    total_time = time.time() - start_time
    print(f"\n📊 Processing Summary:")
    print(f"   Total segments processed: {total_processed}")
    print(f"   Total segments failed: {total_failed}")
    print(f"   Success rate: {total_processed/(total_processed+total_failed)*100:.1f}%")
    print(f"   Total processing time: {total_time/60:.1f} minutes")
    print(f"   Average time per segment: {total_time/max(1,total_processed):.1f}s")
    
    print(f"\n📁 Results breakdown:")
    print(f"   Overhead scenes: {len(overhead_json)}")
    print(f"   Normal scenes: {len(normal_json)}")
    print(f"   External scenes: {len(external_json)}")
    
    # Save all results
    combined_file = save_results()
    
    if combined_file:
        total_segments = sum(len(scenes) for scenes in [overhead_json, normal_json, external_json])
        print(f"\n🎯 Final combined file contains {total_segments} scenes from all views!")
        print(f"📄 Ready for submission: {combined_file}")
    
    # Cleanup
    try:
        del model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🧹 Memory cleanup completed!")
    except:
        pass


if __name__ == "__main__":
    main()
