from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
# AutoTokenizer 외에, 4비트 양자화 설정을 위한 BitsAndBytesConfig와 torch를 임포트합니다.
from transformers import AutoTokenizer, BitsAndBytesConfig 
import torch
from typing import Tuple
import os

# 이전 코드에서 사용되던 glob, json, safetensors 관련 임포트는 
# from_pretrained를 사용하면서 필요 없어졌으므로 제거했습니다.


def load_hf_quantized_model(model_path: str, device: str = "cuda") -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Hugging Face 모델 저장소에서 PaliGemma 모델을 4-bit 양자화하여 로드합니다.
    
    Args:
        model_path (str): 모델 가중치가 있는 로컬 경로 또는 Hugging Face 모델 이름.
        device (str): 모델을 로드할 대상 디바이스 ("cuda", "cpu").
        
    Returns:
        Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]: 로드된 모델과 토크나이저.
    """
    print(f"Loading quantized model from {model_path}")
    print(f"Target device: {device}")
    
    # ----------------------------------------------------
    # 1. 양자화 설정 (BitsAndBytes 4-bit NF4)
    # ----------------------------------------------------
    # bnb_4bit_compute_dtype=torch.bfloat16 설정은 4비트 가중치로 연산할 때 
    # BFloat16 정밀도를 사용하여 성능을 유지하도록 권장됩니다. (GPU 지원 시)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",       # NormalFloat 4-bit quantization 사용
        bnb_4bit_use_double_quant=True,  # 2차 양자화 (메모리 사용량 추가 절약)
        bnb_4bit_compute_dtype=torch.bfloat16 
    )
    print("4-bit NF4 Quantization configuration created.")
    
    # ----------------------------------------------------
    # 2. 모델 및 토크나이저 로드 (Hugging Face 표준 방식 사용)
    # ----------------------------------------------------
    try:
        # PaliGemmaForConditionalGeneration.from_pretrained()를 사용하여
        # config 로드, safetensors 로드, state_dict 적용, weights tying을 한 번에 처리합니다.
        # 이 때 quantization_config를 전달하여 모델을 로드와 동시에 양자화합니다.
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            config=PaliGemmaConfig.from_pretrained(model_path), # 커스텀 Config 로드
            quantization_config=nf4_config,
            device_map=device,           
            torch_dtype=torch.bfloat16,  
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        
        # 4비트 양자화 적용 시 LLM의 Embedding Layer는 양자화되지 않지만,
        # 메모리 효율을 위해 전체 모델을 지정된 장치로 옮깁니다.
        if device.startswith("cuda") or device == "mps":
            model.to(device)

        print(f"Quantized Model and Tokenizer loaded successfully on {device}.")
        
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        raise

    return (model, tokenizer)


# NOTE: 기존의 load_hf_model 함수는 위 load_hf_quantized_model 함수로 대체되었습니다.
# 기존의 수동 로직(glob, safe_open 등)은 Hugging Face의 from_pretrained가 더 안정적으로 처리합니다.