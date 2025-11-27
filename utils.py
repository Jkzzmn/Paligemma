from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer # BitsAndBytesConfig는 수동 로드로 인해 필요 없으므로 제거
import torch
from typing import Tuple
import os
import json 
import glob # safetensors 파일 목록 탐색을 위해 필요
from safetensors import safe_open # safetensors 파일 읽기를 위해 필요

# NOTE: 이 파일은 Custom Model 클래스의 from_pretrained 속성 부재 문제를 해결하기 위해
# 가중치를 수동으로 로드(load_state_dict)하는 방식으로 변경되었습니다.

def load_hf_model_mps_optimized(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    PaliGemma 모델을 수동으로 생성하고, safetensors 파일에서 가중치를 로드하여 
    MacBook 환경(CPU/MPS)에서 안정적으로 실행되도록 최적화합니다.
    """
    print(f"Loading model from {model_path}")
    print(f"Target device: {device}")
    
    # ----------------------------------------------------
    # 1. Config 로드 (모델 구조 파악)
    # ----------------------------------------------------
    try:
        # Hugging Face 표준 방식으로 Config 로드
        config = PaliGemmaConfig.from_pretrained(model_path)
    except Exception:
        # 수동 Config 로드 (폴백)
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            model_config_file = json.load(f)
            config = PaliGemmaConfig(**model_config_file)

    # ----------------------------------------------------
    # 2. 모델 생성 및 가중치 수동 로드 (AttributeError 해결)
    # ----------------------------------------------------
    tensors = {}
    try:
        print("Creating model instance and loading state dict manually...")
        
        # 2-1. safetensors 파일 탐색 및 가중치 딕셔너리로 통합
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # 2-2. 모델 생성 (BFloat16으로 메모리 절약 시도)
        # MPS/CPU 환경 최적화를 위해 BFloat16으로 dtype 지정
        model = PaliGemmaForConditionalGeneration(config).to(torch.bfloat16)
        
        # 2-3. 가중치 로드
        model.load_state_dict(tensors, strict=False)
        print("Model state dict loaded successfully.")

    except Exception as e:
        print(f"FATAL ERROR during model creation or state dict application: {e}")
        raise

    # ----------------------------------------------------
    # 3. 토크나이저 로드 및 마무리
    # ----------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        
        # 모델을 최종 디바이스로 이동 및 eval 모드 설정
        # utils.py 파일의 해당 부분을 찾아서 아래처럼 수정하세요.

    # 4. 토크나이저 로드 및 마무리
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
            
            # 모델의 최종 디바이스 설정 및 eval 모드
            model.to(device).eval() 

            # -------------------- [수정된 부분] --------------------
            # model.device 대신 model.parameters()에서 디바이스 정보를 안전하게 가져옴
            final_device = next(model.parameters()).device 
            print(f"Model loading complete. Final device: {final_device}")
            # -------------------- [END 수정된 부분] --------------------
            
        except Exception as e:
            print(f"Error loading tokenizer or setting device: {e}")
            raise

        return (model, tokenizer)
        
    except Exception as e:
        print(f"Error loading tokenizer or setting device: {e}")
        raise

    return (model, tokenizer)