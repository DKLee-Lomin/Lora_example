import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# 경로 설정
base_model_name = "Qwen/Qwen3-VL-4B-Instruct"
lora_adapter_path = "./qwen3_qlora_final"  # QLoRA adapter가 저장된 경로
merged_model_path = "./qwen3_vl_merged"

print("="*60)
print("Qwen3-VL-4B-Instruct + QLoRA Adapter Merge")
print("="*60)

# 1. 4-bit 양자화 설정
print(f"\n1. Setting up 4-bit quantization config...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print(f"   ✓ Quantization config created")

# 2. Base VL 모델 로드 (4-bit 양자화)
print(f"\n2. Loading base model with 4-bit quantization: {base_model_name}")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
print(f"   ✓ Base model loaded (4-bit quantized)")

# 3. Processor 로드
print(f"\n3. Loading processor...")
processor = AutoProcessor.from_pretrained(base_model_name)
print(f"   ✓ Processor loaded")

# 4. QLoRA adapter 로드
print(f"\n4. Loading QLoRA adapter from: {lora_adapter_path}")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
print(f"   ✓ QLoRA adapter loaded")

# 5. 모델 merge (dequantize 후 merge)
print("\n5. Merging QLoRA weights into base model...")
print("   ⚠️  Note: Merging will dequantize the model to full precision")
merged_model = model.merge_and_unload()
print(f"   ✓ Merge completed")

# 6. Merged 모델 저장
print(f"\n6. Saving merged model to: {merged_model_path}")
merged_model.save_pretrained(merged_model_path)
processor.save_pretrained(merged_model_path)
print(f"   ✓ Model saved (full precision)")

print("\n" + "="*60)
print("Merge 완료!")
print("="*60)

# 7. Merged 모델 테스트 (이미지 설명)
print("\n" + "="*60)
print("Merged Model 테스트")
print("="*60)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
print("\nPreparing inputs...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(merged_model.device)

# Inference
print("Generating response...")
generated_ids = merged_model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n" + "-"*60)
print("Response:")
print("-"*60)
print(output_text[0])
print("-"*60)

# 8. 모델 정보 출력
print("\n" + "="*60)
print("모델 정보")
print("="*60)
print(f"Base model: {base_model_name}")
print(f"QLoRA adapter: {lora_adapter_path}")
print(f"Merged model saved to: {merged_model_path}")
print(f"Model dtype: {merged_model.dtype}")
print(f"Total parameters: {merged_model.num_parameters():,}")
print("="*60)

print("\n✓ 이제 merged 모델을 다음과 같이 사용할 수 있습니다:")
print(f"  model = Qwen3VLForConditionalGeneration.from_pretrained('{merged_model_path}')")
#print(f"  processor = AutoProcessor.from_pretrained('{merged_model_path}')")
##print("\n💡 참고: Merged 모델은 full precision입니다.")
##print("   메모리를 절약하려면 다시 양자화하여 로드할 수 있습니다:")
#print(f"  model = Qwen3VLForConditionalGeneration.from_pretrained('{merged_model_path}', quantization_config=bnb_config)")
