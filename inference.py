from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer, BitsAndBytesConfig 
import torch
from typing import Tuple
import os
import json 

def load_hf_model_mps_optimized(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    PaliGemma 모델을 로드하고, MacBook 환경(CPU/MPS)에서 안정적으로 실행되도록 최적화합니다.
    
    1. 8비트 양자화를 시도합니다 (메모리 최적화 시도).
    2. 8비트 로드 실패 시, BFloat16/Float32로 폴백하여 실행 가능성을 높입니다.
    """
    print(f"Loading model from {model_path}")
    print(f"Target device: {device}")
    
    # ----------------------------------------------------
    # 1. Config 로드
    # ----------------------------------------------------
    try:
        config = PaliGemmaConfig.from_pretrained(model_path)
    except Exception:
        # 커스텀 config 클래스가 from_pretrained를 지원하지 않을 경우 수동 로드
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            model_config_file = json.load(f)
            config = PaliGemmaConfig(**model_config_file)


    # ----------------------------------------------------
    # 2. 8비트 양자화 시도 (BitsAndBytes 호환성 문제 회피를 위해 HF 로직 활용)
    # ----------------------------------------------------
    model = None # 초기화

    try:
        print("Attempting 8-bit quantization load...")
        
        nf8_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # device_map="auto"와 torch_dtype=bfloat16 설정을 사용하여 MPS 환경에 최적화
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            quantization_config=nf8_config, # 8비트 설정 전달
            device_map="auto",              # 자동으로 CPU/MPS에 할당
            torch_dtype=torch.bfloat16,     # M2/M3에서 가장 효율적인 정밀도
            trust_remote_code=True,
        )
        print("8-bit quantization loading successful (or successfully substituted by HF logic).")
    
    except Exception as e:
        # 8비트 로드 실패 시 (MacBook에서 자주 발생)
        print(f"Error during 8-bit loading: {e}")
        print("Falling back to standard BFloat16/Float32 loading...")
        
        # ----------------------------------------------------
        # 3. 폴백: 양자화 없이 BFloat16으로 로드 (MacBook 최적화)
        # ----------------------------------------------------
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            device_map="cpu", # 양자화 실패 시 명시적으로 CPU로 로드
            torch_dtype=torch.bfloat16, # BFloat16으로 메모리 절감 시도
            trust_remote_code=True,
        )
        print("Fallback to BFloat16/CPU loading successful. No quantization applied.")


    # ----------------------------------------------------
    # 4. 토크나이저 로드 및 마무리
    # ----------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        
        # 모델의 최종 디바이스 설정 및 eval 모드
        # device_map="auto"를 사용했으므로, .to(device)는 모델이 할당된 장치로 이동을 시도합니다.
        # inference.py의 device 변수가 최종적으로 "mps" 또는 "cpu"를 가리키므로, 해당 장치로 설정합니다.
        model.to(device).eval() 

        print(f"Model loading complete. Final device: {model.device}")
        
    except Exception as e:
        print(f"Error loading tokenizer or setting device: {e}")
        raise

    return (model, tokenizer)