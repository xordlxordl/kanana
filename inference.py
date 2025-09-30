# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("=" * 50)
print("파인튜닝된 Kanana 한국어 모델 (LoRA) 로딩 중...")
print("=" * 50)

# 베이스 모델과 LoRA adapter 경로
base_model_name = "kakaocorp/kanana-nano-2.1b-instruct"
lora_adapter_path = "./fine_tuned_kanana_lora"

# 베이스 모델 로드
print(f"\n[1/3] 베이스 모델 로딩: {base_model_name}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("[OK] 베이스 모델 로딩 완료")

# LoRA adapter 로드
print(f"\n[2/3] LoRA adapter 로딩: {lora_adapter_path}")
model = PeftModel.from_pretrained(model, lora_adapter_path)
print("[OK] LoRA adapter 로딩 완료")

# 토크나이저 로드
print(f"\n[3/3] 토크나이저 로딩...")
tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
print(f"[OK] 모든 컴포넌트 로딩 완료 (디바이스: {model.device})\n")

def generate_answer(question, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """
    질문에 대한 답변 생성

    Args:
        question: 입력 질문
        max_new_tokens: 최대 생성 토큰 수 (입력 제외)
        temperature: 샘플링 온도 (낮을수록 일관된 답변)
        top_p: nucleus sampling 파라미터

    Returns:
        생성된 답변 텍스트
    """
    # 질문을 설명문 시작 형태로 변환
    if question.endswith('?'):
        prompt = question[:-1].replace('이란', '은').replace('란', '는') + ' '
    elif question.endswith('란'):
        prompt = question[:-1] + '은 '
    elif '설명' in question or '알려' in question or '무엇' in question:
        topic = question.split()[0]
        prompt = f"{topic}은 "
    else:
        prompt = question + ' '

    # 입력 토큰화
    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 답변 생성
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # 입력 부분 제거하고 생성된 부분만 디코딩
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 자연스러운 문장 추출 - 완성된 문장만 반환
    import re
    # 마침표 뒤 공백 또는 끝으로 문장 분리
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

    # 처음 2-3개 완성된 문장만 추출
    complete_sentences = []
    for sent in sentences[:3]:
        if sent and (sent.endswith('.') or sent.endswith('!') or sent.endswith('?')):
            complete_sentences.append(sent)
        elif sent and len(complete_sentences) == 0:  # 첫 문장이 완성 안되면 추가
            if not sent.endswith(('.', '!', '?')):
                sent += '.'
            complete_sentences.append(sent)

    if complete_sentences:
        return ' '.join(complete_sentences[:2])  # 최대 2문장

    return answer.strip()

# 대화형 QA 루프
print("=" * 50)
print("QA 시스템 시작 (종료: 'exit' 또는 'quit')")
print("=" * 50)

while True:
    try:
        # 사용자 입력
        question = input("\n질문: ").strip()

        # 종료 조건
        if question.lower() in ['exit', 'quit', '종료']:
            print("\n[OK] QA 시스템 종료")
            break

        # 빈 입력 처리
        if not question:
            continue

        # 답변 생성
        print("\n답변 생성 중...")
        answer = generate_answer(question)
        print(f"\n답변: {answer}")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\n\n[OK] 사용자에 의해 종료됨")
        break
    except Exception as e:
        print(f"\n[에러] {str(e)}")
        continue