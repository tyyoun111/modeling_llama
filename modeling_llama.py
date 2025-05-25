# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Callable, Optional, Tuple, Union

#####
#ty add
import os
import numpy as np
import serial #add for uart
#####

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg
from .configuration_llama import LlamaConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types(e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
# Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along
which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted
to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the
shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim],
then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to
the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set
unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors
rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1,repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch,num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

###########################################################################
## modified eager_attention_forward : eager 함수 내부에서 양자화 →  matmul -> dequant
############################################################################
#def eager_attention_forward(
#    module: nn.Module,
#    query: torch.Tensor,
#    key: torch.Tensor,
#    value: torch.Tensor,
#    attention_mask: Optional[torch.Tensor],
#    scaling: float,
#    dropout: float = 0.0,
#    **kwargs,
#):
#    import os
#    import numpy as np
#
#    # 디렉토리 및 카운터 초기화
#    dequant_query_dir = "/home/seokjin/Desktop/Taeyoung/dequant_query_state"
#    dequant_key_dir   = "/home/seokjin/Desktop/Taeyoung/dequant_key_state"
#    os.makedirs(dequant_query_dir, exist_ok=True)
#    os.makedirs(dequant_key_dir, exist_ok=True)
#    if not hasattr(module, "_dequant_layer_counter"):
#        module._dequant_layer_counter = {"query": {}, "key": {}}
#    dequant_layer_counter = module._dequant_layer_counter
#
#    # 원본 RoPE 적용 후 q,k 복사 (fp32)
#    key_states   = repeat_kv(key, module.num_key_value_groups)
#    value_states = repeat_kv(value, module.num_key_value_groups)
#    orig_query      = query.clone().detach()
#    orig_key_states = key_states.clone().detach()
#
#    # -------------- int8 양자화만 수행하는 헬퍼 --------------
#    def quantize_only(x, bits=4):
#        B, H, S, D = x.shape
#        qmax = 2 ** (bits - 1) - 1
#        qmin = -qmax
#
#        q_int8 = torch.zeros_like(x, dtype=torch.int8)
#        scales = []
#        for h in range(H):
#            head = x[:, h]  # (B, S, D)
#            abs_max = torch.quantile(head.abs().float(), 0.995).clamp(min=1e-6)
#            scale = abs_max / qmax
#            q = torch.clamp((head / scale).round(), qmin, qmax).to(torch.int8)
#            q_int8[:, h] = q
#            scales.append(scale)
#        # 헤드별 스케일을 (1, H, 1, 1) 텐서로
#        scale_tensor = torch.stack(scales).to(x.device).view(1, -1, 1, 1)
#        return q_int8, scale_tensor, 0  # zero_point=0 고정
#
#    # 1) 원본 FP32 어텐션 계산
#    A_fp32 = torch.matmul(orig_query, orig_key_states.transpose(2, 3)) * scaling
#
#    # 2) 진짜 quantize→int8 matmul→dequantize 방식
#    q_int8, q_scale, _ = quantize_only(orig_query,      bits=4)
#    k_int8, k_scale, _ = quantize_only(orig_key_states, bits=4)
#
#    # int8끼리 matmul → int32 accumulator
#    int32_res = torch.matmul(q_int8.int(), k_int8.int().transpose(2, 3))
#
#    # dequantize matmul 결과
#    combined_scale = q_scale * k_scale
#    A_qdq = int32_res.float() * combined_scale * scaling
#
#    # 로그용 저장 (optional)
#    layer_idx = getattr(module, 'layer_idx', 'X')
#    counter_q = dequant_layer_counter["query"].setdefault(layer_idx, 0)
#    dequant_layer_counter["query"][layer_idx] = counter_q + 1
#    np.save(os.path.join(dequant_query_dir, f"q8_matmul_layer{layer_idx}_step{counter_q}.npy"),
#            A_qdq.cpu().numpy())
#    print(f"[SAVE QDQ] layer={layer_idx} step={counter_q}, shape={A_qdq.shape}")
#
#    # 3) SNR 계산
#    signal_power = (A_fp32 ** 2).sum()
#    noise_power  = ((A_fp32 - A_qdq) ** 2).sum()
#    snr_db = 10 * torch.log10(signal_power / noise_power)
#    print(f"[SNR dB] layer={layer_idx}  Q→MM→DQ SNR={snr_db:.2f} dB")
#
#    # 4) 이후 흐름: dequantized matmul 결과를 attention 가중치로 사용
#    attn_weights = A_qdq
#    if attention_mask is not None:
#        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#        attn_weights = attn_weights + causal_mask
#
#    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
#    attn_weights = attn_weights.to(orig_query.dtype)
#    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
#
#    # 5) value와 곱해서 최종 어텐션 아웃풋
#    attn_output = torch.matmul(attn_weights, value_states)
#    attn_output = attn_output.transpose(1, 2).contiguous()
#    attn_output = attn_output.to(value.dtype)
#
#    return attn_output, attn_weights

###########################################################################
## modified eager_attention_forward : fakequant version
###########################################################################
"""
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    import os
    import numpy as np
    import torch.nn.functional as F  # functional.softmax

    SAVE_INTERVAL = 10

    # 로그용 디렉토리/카운터
    ##dequant_query_dir = "/home/seokjin/Desktop/Taeyoung/dequant_query_state"
    ##os.makedirs(dequant_query_dir, exist_ok=True)
    if not hasattr(module, "_dequant_layer_counter"):
        module._dequant_layer_counter = {"fake": {}}
    fq_counter = module._dequant_layer_counter["fake"]

    # 1) RoPE 적용 후 Q, K 준비
    key_states   = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    orig_q = query.clone().detach()
    orig_k = key_states.clone().detach()

    # 2) fake‑quant → dequant 없이 fp32로 바로 리턴
    def fake_quant(x, bits=4):
        qmax = 2 ** (bits - 1) - 1
        B, H, S, D = x.shape
        x_fq = torch.empty_like(x)
        for h in range(H):
            head = x[:, h, :, :].float()
            abs_max = torch.quantile(head.abs(), 0.995).clamp(min=1e-6)
            scale = abs_max / qmax
            q = (head / scale).round().clamp(-qmax, qmax)
            x_fq[:, h, :, :] = (q * scale).to(x.dtype)
        return x_fq

    # 3) 원본 FP32 어텐션 스코어
    A_fp32 = torch.matmul(orig_q, orig_k.transpose(2, 3)) * scaling

    # 4) fake‑quant(fp32) → 바로 matmulr1
    q_fq = fake_quant(orig_q, bits=4)
    k_fq = fake_quant(orig_k, bits=4)
    A_fq = torch.matmul(q_fq, k_fq.transpose(2, 3)) * scaling

    # 5) Pre‑MatMul SNR 계산
    layer_idx = getattr(module, "layer_idx", "X")
    step = fq_counter.setdefault(layer_idx, 0)
    # Q 텐서 SNR
    signal_q = (orig_q.float()**2).sum()
    noise_q  = ((orig_q.float() - q_fq.float())**2).sum()
    snr_q_db = 10 * torch.log10(signal_q / (noise_q + 1e-12))
    # K 텐서 SNR
    signal_k = (orig_k.float()**2).sum()
    noise_k  = ((orig_k.float() - k_fq.float())**2).sum()
    snr_k_db = 10 * torch.log10(signal_k / (noise_k + 1e-12))
    print(f"[SNR FakeQ PreMatMul] layer={layer_idx} step={step}
Q_SNR={snr_q_db:.2f} dB, K_SNR={snr_k_db:.2f} dB")

    # 6) Post‑MatMul SNR 계산
    signal = (A_fp32 ** 2).sum()
    noise  = ((A_fp32 - A_fq) ** 2).sum()
    snr_db = 10 * torch.log10(signal / (noise + 1e-12))
    print(f"[SNR FakeQ PostMatMul] layer={layer_idx} step={step}
SNR={snr_db:.2f} dB")



    ## storing at a 10 step period

    if step % SAVE_INTERVAL == 0:
        print (f"[SAVE] step {step} (interval {SAVE_INTERVAL}) -> tensor saved")
        save_tensor(q_fq, "query", layer_idx)
        save_tensor(k_fq, "key", layer_idx)
        save_quantized_tensor(orig_q, "query", layer_idx)
        save_quantized_tensor(orig_k, "key", layer_idx)

    fq_counter[layer_idx] += 1

    # 8) softmax → value 곱셈
    attn_weights = A_fq
    if attention_mask is not None:
        cm = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + cm

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().to(value.dtype)

    return attn_output, attn_weights

"""

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    import os
    import numpy as np

    SAVE_INTERVAL = 10

    if not hasattr(module, "_dequant_layer_counter"):
        module._dequant_layer_counter = {"int8": {}}
    q_counter = module._dequant_layer_counter["int8"]

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    orig_q = query.clone().detach()
    orig_k = key_states.clone().detach()

    def quantize_only(x, bits=4):
        B, H, S, D = x.shape
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax

        q_int8 = torch.zeros_like(x, dtype=torch.int8)
        scales = []
        for h in range(H):
            head = x[:, h]  # (B, S, D)
            abs_max = torch.quantile(head.abs().float(), 0.995).clamp(min=1e-6)
            scale = abs_max / qmax
            q = torch.clamp((head / scale).round(), qmin, qmax).to(torch.int8)
            q_int8[:, h] = q
            scales.append(scale)
        scale_tensor = torch.stack(scales).to(x.device).view(1, -1, 1, 1)
        return q_int8, scale_tensor, 0  # zero_point = 0

    ############### defs send 8bit to FPGA using uart ################
    ### tx ###
    # pack 4bit to 8bit
    def pack_4bit(tensor_int8):
        assert tensor_int8.min() >= -8 and tensor_int8.max() <= 7
        data = tensor_int8.view(-1).tolist()
        packed = [((a&0x0F)<<4)|(b&0x0F) for a,b in zip(data[::2],data[1::2])]
        return bytes(packed)
    # send Q
    def send_tensor_via_uart(tensor_int8, uart, tag=b'Q'):
        packed = pack_4bit(tensor_int8)
        uart.write(tag + packed)
    # send K
    def send_tensor_k_with_length(tensor_int8, uart, tag=b'K'):
        packed = pack_4bit(tensor_int8)
        S = tensor_int8.shape[2]
        assert S<=128, "S too large to fit in 1byte"
        uart.write(tag + bytes([S])+packed)
    ### rx ###
    # receive qkt
    def receive_qkt_result(uart, length=64):
        raw = uart.read(length)
        data = np.frombuffer(raw, dtype=np.uint8)
        return data.astype(np.float32)
    # unpack 8bit to 4bit
    def unpack_4bit_signed(data_bytes):
        result = []
        for byte in data_bytes:
            high = (byte >> 4) & 0x0F
            low = byte&0x0F
            for val in (high,low):
                if val >= 8:
                    val -= 16
                result.append(val)
            result.extend([high,low])
        return np.array(result, dtype= np.float32)
    # uart '/dev/ttyTHS1' or '/dev/ttyTHS2', baudrate 115200
    uart = serial.Serial('/dev/ttyTHS1', 115200)
    #########################################################################

    # 양자화
    q_int8, q_scale, _ = quantize_only(orig_q, bits=4)
    k_int8, k_scale, _ = quantize_only(orig_k, bits=4)

    ##### send uart #####
    send_tensor_via_uart(q_int8, uart, tag=b'Q')
    send_tensor_k_with_length(k_int8, uart, tag=b'K')
    #####################

    layer_idx = getattr(module, "layer_idx", "X")
    step = q_counter.setdefault(layer_idx, 0)

    # 출력
    print(f"\n[Quantized] layer={layer_idx} step={step}")
    print(f"  query int8 shape: {q_int8.shape}, dtype: {q_int8.dtype}")
    print(f"  key int8 shape  : {k_int8.shape}, dtype: {k_int8.dtype}")
    print(f"  query scale per head: {q_scale.view(-1).cpu().numpy()}")
    print(f"  key scale per head  : {k_scale.view(-1).cpu().numpy()}")

    head_idx = 0  # 디버깅할 head index

    # ──────────────── QUERY 디버깅 ────────────────
    print("\n[DEBUG] Query Head 0 정보")
    print("→ query int8:", q_int8[0, head_idx, 0].cpu().numpy())
    print("→ query scale:", q_scale[0, head_idx, 0, 0].item())

    dequant_q = q_int8[0, head_idx, 0].float() * q_scale[0, head_idx, 0, 0]
    print("→ dequantized q:", dequant_q.cpu().numpy())
    print("→ original q    :", orig_q[0, head_idx, 0].float().cpu().numpy())
    print("→ abs error     :", (orig_q[0, head_idx, 0].float() -dequant_q).abs().cpu().numpy())

    # ──────────────── KEY 디버깅 ────────────────
    print("\n[DEBUG] Key Head 0 정보")
    print("→ key int8:", k_int8[0, head_idx, 0].cpu().numpy())
    print("→ key scale:", k_scale[0, head_idx, 0, 0].item())

    dequant_k = k_int8[0, head_idx, 0].float() * k_scale[0, head_idx, 0, 0]
    print("→ dequantized k:", dequant_k.cpu().numpy())
    print("→ original k    :", orig_k[0, head_idx, 0].float().cpu().numpy())
    print("→ abs error     :", (orig_k[0, head_idx, 0].float() -dequant_k).abs().cpu().numpy())

    # 저장
    if step % SAVE_INTERVAL == 0:
        os.makedirs("./debug_quant", exist_ok=True)
        np.save(f"./debug_quant/query_int8_layer{layer_idx}_step{step}.npy",q_int8.cpu().numpy())
        np.save(f"./debug_quant/query_scale_layer{layer_idx}_step{step}.npy",q_scale.cpu().numpy())
        np.save(f"./debug_quant/key_int8_layer{layer_idx}_step{step}.npy",k_int8.cpu().numpy())
        np.save(f"./debug_quant/key_scale_layer{layer_idx}_step{step}.npy",k_scale.cpu().numpy())
        print(f"[SAVE] step={step} tensors saved to ./debug_quant/")

    q_counter[layer_idx] += 1

    ########## receive qkt result & run softmax using result ##########
    #if connect fpga, remove#
    #fpga_result = receive_qkt_result(uart,length=32)
    #fpga_result_float = unpack_4bit_signed(fpga_result)
    #attn_weight = torch.tensor(fpga_result_float,device=orig_q.device,dtype=torch.float32)
    #attn_weight = attn_weights.view(1, module.num_heads,orig_q.size(2), orig_k.size(2)
    #if attention_mask is not None:
    #    cm = attention_mask[:, :, :, : key_states.shape[-2]]
    #    attn_weights = attn_weights + cm
    #attn_weight = torchc.nn.functional.sofmax(attn_weight, dim = -1)
    #attn_weight = torch.nn.functional.dropout(attn_weight, p = dropout, training=module.training)


    # 나머지 흐름은 FP32 그대로
    attn_weights = torch.matmul(orig_q, orig_k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        cm = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + cm

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().to(value.dtype)

    return attn_output, attn_weights




###########################################################################
## to save each layer's qk tensor output
#query_dir = "/home/seokjin/Desktop/Taeyoung/query_state"
#key_dir = "/home/seokjin/Desktop/Taeyoung/key_state"
#os.makedirs(query_dir, exist_ok=True)
#os.makedirs(key_dir, exist_ok=True)
#layer_counter = {"query":{}, "key":{}}
#def save_tensor(tensor, name, layer_idx):
#    save_path = query_dir if name == "query" else key_dir
#    if layer_idx not in layer_counter[name]:
#        layer_counter[name][layer_idx] = 0
#    else:
#        layer_counter[name][layer_idx] += 1
#    step = layer_counter[name][layer_idx]
#    filename = f"{name}_layer{layer_idx}_step{step}.npy"
#    filepath = os.path.join(save_path, filename)
#    print(f"[SAVE] {filename}, shape: {tensor.shape}")
#    np.save(filepath, tensor.to(torch.float32).cpu().numpy())
#def quantize_tensor(tensor, bits=4):
#    """
#    Head-wise quantization with outlier clipping (per-head scale)
#    Args:
#        tensor: shape (1, H, S, D)
#        bits: bit width, e.g., 4-bit → [-7, 7]
#    Returns:
#        q_tensor: quantized tensor (int8)
#        scale_list: list of scale per head
#        zero_point: fixed as 0
#    """
#    assert tensor.ndim == 4, "Expected shape (1, H, S, D)"
#    B, H, S, D = tensor.shape
#    qmax = 2 ** (bits - 1) - 1
#    qmin = -qmax
#    # 저장할 quantized 텐서 및 scale 리스트
#    q_tensor = torch.zeros_like(tensor, dtype=torch.int8)
#    scale_list = []
#    for h in range(H):
#        head_tensor = tensor[:, h, :, :]  # shape: (1, S, D)
#        abs_max = torch.quantile(head_tensor.abs().float(), 0.995)  #99.5% 구간만 고려
#        if abs_max == 0:
#            abs_max = 1e-6
#        scale = abs_max / qmax
#        # 양자화
#        q = torch.clamp(torch.round(head_tensor / scale), qmin, qmax)
#        q_tensor[:, h, :, :] = q.to(torch.int8)
#        scale_list.append(scale.item())
#    return q_tensor, scale_list, 0


#quant_query_dir = "/home/seokjin/Desktop/Taeyoung/quant_query_state"
#quant_key_dir = "/home/seokjin/Desktop/Taeyoung/quant_key_state"
#os.makedirs(quant_query_dir, exist_ok=True)
#os.makedirs(quant_key_dir, exist_ok=True)
#quant_layer_counter = {"query":{}, "key":{}}
#def save_quantized_tensor(tensor, name, layer_idx, bits=4):
#    save_dir = quant_query_dir if name == "query" else quant_key_dir
#    if layer_idx not in quant_layer_counter[name]:
#        quant_layer_counter[name][layer_idx] = 0
#    else:
#        quant_layer_counter[name][layer_idx] += 1
#    step = quant_layer_counter[name][layer_idx]
#    quant_tensor, scale, zp = quantize_tensor(tensor, bits)
#    tensor_file = os.path.join(save_dir,f"{name}_layer{layer_idx}_step{step}_tensor.npy")
#    meta_file = os.path.join(save_dir,f"{name}_layer{layer_idx}_step{step}_meta.npy")
#   print(f"[SAVE QUANT] {tensor_file}, shape: {quant_tensor.shape},scale={scale:.4f}, zp={zp}")
#   np.save(tensor_file, quant_tensor.cpu().numpy().astype(np.int8))
#   np.save(meta_file, {"scale": float(scale), "zero_point": int(zp)},allow_pickle=True)
#    print(f"[SAVE QUANT] {tensor_file}, shape: {quant_tensor.shape}")
#
#    if isinstance(scale, list):
#        #for h, s in enumerate(scale):
#        #   print(f"  └ Head {h:2d}: scale = {s:.6f}")
#        np.save(meta_file, {"scale": scale, "zero_point": int(zp)},allow_pickle=True)
#    else:
#        #print(f"  scale={scale:.6f}, zp={zp}")
#        np.save(meta_file, {"scale": float(scale), "zero_point":int(zp)}, allow_pickle=True)
#
#    np.save(tensor_file, quant_tensor.cpu().numpy().astype(np.int8))
################################################################################

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size// config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads //config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True



        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads *self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads *self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads *self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim,config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states =self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states =self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states =self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)


        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states,key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_positionneeded for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position":cache_position}
            key_states, value_states =past_key_value.update(key_states, value_states, self.layer_idx,cache_kwargs)

       # print(f"\n[DEBUG] LlamaAttention layer {self.layer_idx} -_attn_implementation = {self.config._attn_implementation}")

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once("`torch.nn.functional.scaled_dot_product_attention` does not support`output_attentions=True`. Falling back to " 'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.')
            else:
                attention_interface =ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm =LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor,torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass
documentation for the generic methods the
    library implements for all its model (such as downloading or
saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch
[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch
documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the
model. Initializing with a config file does not
            load the weights associated with the model, only the
configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the
model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without anyspecific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See
[`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size,
sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token
indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See
[`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last
`input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read
[`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the
paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size,
sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the
position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the
self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding.
This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding,
when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details,
see our [kv cache
guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally
input only the last `input_ids` (those that don't
            have their past key value states given to this model) of
shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size,
sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose
to directly pass an embedded representation. This
            is useful if you want more control over how to convert
`input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are
returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all
attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead
of a plain tuple.
        cache_position (`torch.LongTensor` of shape
`(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence
tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to
update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers*layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange( past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input` attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.   
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size,
config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50",
new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast,
config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for
extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute
`span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]