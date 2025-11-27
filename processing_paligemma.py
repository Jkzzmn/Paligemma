from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# NOTE: 순환 참조를 피하기 위해 프로젝트 내부 모듈(modeling_gemma, utils) 임포트 제거됨.

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt: str, bos_token: str, image_seq_len: int, image_token: str) -> str:
    # ... (함수 내용은 그대로 유지)
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale_np(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    return (image * scale).astype(dtype)


def normalize_np(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    return (image - mean) / std


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: Image.Resampling,
    rescale_factor: float,
    image_mean: Optional[Union[float, List[float]]],
    image_std: Optional[Union[float, List[float]]],
) -> List[torch.Tensor]:
    # ... (함수 내용은 그대로 유지)
    height, width = size

    processed_images = []
    for image in images:
        resized_image = image.resize(
            (width, height), resample=resample
        )
        
        image_np = np.array(resized_image)
        image_np = rescale_np(image_np, scale=rescale_factor)
        image_np = normalize_np(image_np, mean=image_mean, std=image_std)
        image_np = image_np.transpose(2, 0, 1)
        
        processed_images.append(torch.tensor(image_np))
        
    return processed_images


class PaliGemmaProcessor:
    """
    PaliGemma 모델을 위한 이미지와 텍스트 전처리 파이프라인을 통합합니다.
    """
    IMAGE_TOKEN = "<image>"
    DEFAULT_RESAMPLING = Image.Resampling.BICUBIC 

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        self.tokenizer = tokenizer

        # ... (tokenizer 설정 코드는 동일)
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False


    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        # ... (call 함수 코드는 동일)
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=self.DEFAULT_RESAMPLING,
            rescale_factor=1 / 255.0, 
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        pixel_values = torch.stack(pixel_values, dim=0) 

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data