# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

print("=" * 50)
print("Kanana-nano-2.1B 한국어 모델 (LoRA) 다운로드 및 로딩 시작...")
print("=" * 50)

# 모델과 토크나이저 불러오기 (한국어 특화)
model_name = "kakaocorp/kanana-nano-2.1b-instruct"  # 카카오 한국어 경량 모델 (2.1B)

print(f"\n[1/3] 토크나이저 다운로드 중: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("[OK] 토크나이저 로딩 완료")

print(f"\n[2/3] 베이스 모델 다운로드 중: {model_name} (약 2.1GB)")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("[OK] 베이스 모델 로딩 완료")

print(f"\n[3/3] LoRA 설정 적용 중...")
# LoRA 설정 (파라미터 일부만 학습)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # LoRA rank (낮을수록 빠름)
    lora_alpha=16,          # LoRA scaling
    lora_dropout=0.1,       # Dropout
    target_modules=["q_proj", "v_proj"],  # 어텐션 레이어만 학습
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 학습 가능한 파라미터 출력
print("[OK] LoRA 적용 완료\n")

from datasets import load_dataset

print("=" * 50)
print("데이터셋 로딩 중...")
print("=" * 50)
# 예시 데이터 로딩 (실제 데이터셋 경로로 변경 필요)
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # 예시
dataset = load_dataset("json", data_files={"train": "train.json", "test": "test.json"})
print(f"[OK] Train: {len(dataset['train'])}개, Test: {len(dataset['test'])}개 샘플 로딩 완료\n")

# 데이터 토큰화
print("데이터 토큰화 중...")
def tokenize_function(examples):
    # 입력을 토큰화하고 labels도 동일하게 설정 (Causal LM 학습용)
    model_inputs = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("[OK] 토큰화 완료\n")


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # 결과를 저장할 디렉토리
    num_train_epochs=5,              # LoRA는 빠르므로 5 에폭
    per_device_train_batch_size=4,   # LoRA는 메모리 적게 사용 -> 배치 증가
    per_device_eval_batch_size=4,    # 평가 배치 사이즈
    learning_rate=1e-4,              # LoRA는 높은 LR 사용
    warmup_steps=10,                 # 러닝 레이트 웜업
    weight_decay=0.01,               # 가중치 감소
    logging_dir='./logs',            # 로그 저장 디렉토리
    logging_steps=5,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    fp16=False,                      # Windows 호환성 위해 비활성화
    bf16=False,                      # BFloat16도 비활성화
    save_total_limit=1,              # 최근 1개 체크포인트만 저장
    gradient_accumulation_steps=1,
)

trainer = Trainer(
    model=model,                         # fine-tuning할 모델
    args=training_args,                  # 학습 인자
    train_dataset=tokenized_datasets['train'],  # 학습 데이터셋
    eval_dataset=tokenized_datasets['test'],    # 평가 데이터셋
)

# 학습 시작
print("=" * 50)
print("파인튜닝 시작...")
print("=" * 50)
trainer.train()

# LoRA adapter 저장 (전체 모델이 아닌 adapter만 저장 - 매우 작음!)
print("\n" + "=" * 50)
print("LoRA adapter 저장 중...")
print("=" * 50)
model.save_pretrained("./fine_tuned_kanana_lora")
tokenizer.save_pretrained("./fine_tuned_kanana_lora")
print("[OK] LoRA adapter 저장 완료: ./fine_tuned_kanana_lora")
print("    (adapter만 저장되어 크기가 매우 작습니다 - 약 10~50MB)\n")

# 평가하기
print("=" * 50)
print("모델 평가 중...")
print("=" * 50)
eval_results = trainer.evaluate()
print(f"[OK] 평가 완료: {eval_results}\n")


# 추론을 위한 예시 코드
print("=" * 50)
print("테스트 추론 중...")
print("=" * 50)
test_prompt = "인공지능이란"
print(f"입력: {test_prompt}")
inputs = tokenizer(test_prompt, return_tensors="pt")
# 입력을 모델과 같은 디바이스로 이동
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model.generate(inputs['input_ids'], max_length=100)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"출력: {generated_text}")
print("\n[OK] 모든 작업 완료!")

