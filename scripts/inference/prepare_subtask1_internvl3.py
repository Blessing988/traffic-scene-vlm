from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

LOCAL_MODEL_DIR = "saves/internvl3/lora/sft/"   # whatever you passed to `save_dir`
quant_cfg       = BitsAndBytesConfig(load_in_4bit=True)

processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
model     = AutoModelForImageTextToText.from_pretrained(
                LOCAL_MODEL_DIR,
                quantization_config=quant_cfg,
                trust_remote_code=True           # keep this for InternVL 3
           ).eval()
           

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video",
             "url": "/mmfs1/home/blessing.agyeikyem/Foundation_models_transportation/20230707_12_SN17_T1_Camera1_0.mp4"},
            {"type": "text",
             "text": "Describe the pedestrian and vehicle behavior in this segment."}
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,         # ← DON’T forget this
    return_tensors="pt",
    return_dict=True,
).to(model.device, dtype=model.dtype)

out_ids = model.generate(**inputs, max_new_tokens=1024)

print(
    processor.decode(
        out_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
)
