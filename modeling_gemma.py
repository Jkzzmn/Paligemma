import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

# -------------------- [양자화 관련 추가] --------------------
# 4비트 양자화를 위해 bitsandbytes 라이브러리 임포트
# 모든 nn.Linear 레이어를 bnb.Linear4bit으로 대체하여 메모리를 크게 절감합니다.
# -------------------- [END] --------------------

class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():
    """Gemma 언어 모델의 설정 정보를 담는 클래스."""
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        # ... (설정값 초기화는 동일)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    """
    PaliGemma의 전체 설정을 담는 클래스. 
    Siglip Vision Config와 Gemma Text Config를 통합합니다.
    """
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        # Siglip 및 Gemma 설정을 내부적으로 초기화
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # 이미지 토큰 개수 계산 및 설정
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
    """Gemma 모델에 사용되는 Root Mean Square Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # weight는 bias 없이 스케일링을 수행하기 위해 사용
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        # x * 1 / sqrt(mean(x^2) + epsilon)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 정규화는 float32로 수행하고 최종적으로 원래 타입으로 복원 (bfloat16 등)
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) 구현."""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 회전 주파수 (theta) 계산: inv_freq = base^(-2i/dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: theta * position_ids
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # sin/cos 계산을 위한 임베딩 행렬
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """벡터의 절반을 회전시키는 헬퍼 함수."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Query와 Key에 회전 임베딩을 적용."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # RoPE 적용 공식: (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # 8비트/4비트 양자화 제거 후 nn.Linear로 복원
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Grouped Query Attention (GQA)를 위해 Key/Value 헤드를 복제하는 헬퍼 함수."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Key/Value 상태를 Num_Query_Heads에 맞게 복제
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    """
    Gemma의 Multi-Head Attention 모듈.
    Rotary Embedding (RoPE)과 Grouped Query Attention (GQA)을 사용합니다.
    """

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        # Q 헤드 수 / KV 헤드 수 (GQA 그룹 수)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0            

        # -------------------- [MODIFIED: 4-bit 양자화 적용] --------------------
        # Q, K, V, O 프로젝션에 bnb.Linear4bit 적용
        self.q_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_heads, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_key_value_heads, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim * self.num_key_value_heads, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.head_dim * self.num_heads, config.hidden_size, bias=False)
        # -------------------- [END MODIFIED] --------------------
        
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Q, K, V 계산 및 헤드/차원 조정
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE 적용
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV Cache 업데이트
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # GQA를 위해 K, V 헤드 복제
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # 어텐션 가중치 계산: Q * K^T / sqrt(d_k)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 마스킹 적용 (Causal Mask)
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Softmax 및 Dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # 출력 계산: Attn_Weights * V
        attn_output = torch.matmul(attn_weights, value_states)

        # 최종 출력 조정 및 O_proj 적용
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    """Gemma 모델의 단일 트랜스포머 디코더 레이어."""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Layer 1: Attention + 잔차 연결
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Layer 2: MLP + 잔차 연결
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module):
    """Gemma 모델의 핵심 (임베딩 + 디코더 스택)."""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 토큰 임베딩 (Embedding Layer는 일반적으로 양자화에서 제외)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        hidden_states = inputs_embeds
        
        # 임베딩 스케일링 (Gemma 특유의 처리)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # 디코더 레이어 순차적 실행
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # 최종 정규화
        hidden_states = self.norm(hidden_states)

        return hidden_states

class GemmaForCausalLM(nn.Module):
    """Gemma 언어 모델 (최종 출력 로짓 생성)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        
        # -------------------- [MODIFIED: 4-bit 양자화 적용] --------------------
        # 최종 로짓 헤드에 bnb.Linear4bit 적용
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # -------------------- [END MODIFIED] --------------------

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        """임베딩과 LM 헤드 가중치를 공유 (가중치 묶기)."""
        # Linear4bit 가중치 객체가 약간 다르므로, weight.data를 공유합니다.
        self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # GemmaModel을 통한 순전파
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        # 최종 로짓 계산
        logits = self.lm_head(hidden_states)
        logits = logits.float() # 로짓은 float32로 유지

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        # 8비트/4비트 양자화 제거 후 nn.Linear로 복원
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    """
    PaliGemma 모델의 최종 클래스.
    Siglip 비전 타워와 Gemma 언어 모델을 결합하고, 입력 병합 로직을 수행합니다.
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        # 비전 인코더 (Siglip)
        self.vision_tower = SiglipVisionModel(config.vision_config)
        # 멀티모달 투영기
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        # 언어 모델 (Gemma)
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        """이미지 특징을 텍스트 임베딩 내의 <image> 토큰 위치에 삽입하고 어텐션 마스크를 준비합니다."""
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        
        # 이미지 특징 스케일링
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
    
        # 최종 임베딩 행렬 초기화
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        
        # 마스크 생성 (텍스트, 이미지, 패딩)
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # 텍스트 임베딩 추가
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # 이미지 임베딩 삽입 (masked_scatter 사용)
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # 패딩 토큰 0 처리
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### 어텐션 마스크 및 Position ID 생성 ####

        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill 단계 (KV 캐시가 비어있음): Causal Mask 생성 (패딩이 없다고 가정)
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Decoding 단계 (KV 캐시 사용): 쿼리 길이가 1이어야 함
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # 캐시 전체를 바라보는 마스크 (패딩이 없다고 가정)
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # 헤드 차원 추가
        causal_mask = causal_mask.unsqueeze(1)

        # Position ID 계산
        if kv_cache is not None and kv_cache.num_items() > 0:
            # 디코딩 시에는 마지막 토큰의 위치만 사용
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Prefill 시에는 누적합으로 Position ID 생성
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # 입력 임베딩 추출
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 1. 이미지 특징 추출 및 투영
        # 비전 타워를 통과시켜 이미지 특징 추출
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # 투영기를 통과시켜 언어 모델 차원으로 변환
        image_features = self.multi_modal_projector(selected_image_feature)

        # 2. 텍스트 임베딩과 이미지 특징 결합
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
        
        # 3. 언어 모델 순전파
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs