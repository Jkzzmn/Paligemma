Google DeepMind의 30억 매개변수 **Vision-Language Model (VLM)**인 PaliGemma-3B의 추론 파이프라인을 구축하고, 비표준 하드웨어(예: Apple Silicon / CPU 환경)에서 발생하는 호환성 및 메모리 문제를 해결하여 안정적인 실행 환경을 최적화하는 것을 목표로 합니다.

## Key Achievements

VLM 추론 파이프라인 구축: Siglip 비전 인코더와 Gemma 언어 모델을 결합하는 멀티모달 추론 구조를 PyTorch 기반으로 커스텀 구현했습니다.

4-bit 및 8-bit 양자화 문제 해결: MacBook (MPS/CPU) 환경에서 BitsAndBytes 라이브러리 비호환성으로 인한 오류를 진단하고, Hugging Face 표준 로딩 방식을 활용하여 문제를 우회했습니다.

메모리 최적화 로직 구현: $\text{AttributeError}$와 $\text{MPS}$ 메모리 부족(`RuntimeError: MPS backend out of memory`) 문제를 해결하기 위해, 가중치 로드를 **$\text{BFloat16}$ 정밀도** 및 safetensors 수동 로드 방식으로 전환하여 메모리 효율을 극대화했습니다.

안정적인 로더 구현: 모델 추론 클래스의 from_pretrained 누락 문제를 해결하고, 가중치를 직접 로드하는 안정적인 utils 함수를 구현했습니다.

## 기술 스택

언어: Python 3.11

프레임워크: PyTorch, Hugging Face Transformers, BitsAndBytes (진단용)

컴퓨팅: Apple Silicon (MPS), CPU

데이터 형식: safetensors, BFloat16

## 실행 방법
bash launch_inference.sh

or 

test.ipynb

## 환경 설정:

Bash
### pyenv를 사용하여 3.11.2 환경 생성 및 활성화
pyenv local my_paligemma_env
### 라이브러리 설치
pip install -r requirements.txt
가중치 다운로드: (Hugging Face 로그인 및 모델 접근 승인 필요)

Bash
python download_weights.py
추론 실행 (Jupyter Notebook 권장):

--only_cpu=True 옵션이 설정된 경우, CPU에서 BFloat16으로 폴백되어 실행됩니다.