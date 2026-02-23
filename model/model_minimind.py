# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = True,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


# # 外推长度 = factor * original_max_position_embeddings = 32768
# self.rope_scaling = {
#     "beta_fast": 32,  高频边界
#     "beta_slow": 1,   低频边界
#     "factor": 16,     外推因子
#     "original_max_position_embeddings": 2048,   训练时最大长度
#     "attention_factor": 1.0,   注意力缩放因子
#     "type": "yarn"      使用的RoPE补丁
# } if self.inference_rope_scaling else None



def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    
    ## 算θ值，以及attn_factor 注意力缩放因子
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    ## 拿到YaRN所有的参数
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )

        ## 代表推理预测的长度 要大于训练时的最大长度，这时候开始启动YaRN
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp

            ## 这俩步是解个方程，算出在对应高，低频边界的dim索引位置，即知道 哪些dim是高频 可以直接外推，哪些dim是低频 直接内插
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)  ## low是高频边界 dim索引  high是低频边界 dim索引

            
            # 将dim位置索引 映射到 [0,1]区间上，作为ramp值 i为dim即索引
            # i ≤ low	0  高频区
            # low < i < high	(0,1)
            # i ≥ high	1  低频区
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)

            ## 高频就按原公式算 低频就按 factor 倍缩小
            freqs = freqs * (1 - ramp + ramp / factor)  ## shape是 (dim/2, )

    t = torch.arange(end, device=freqs.device) ## shape是 (end, )
     ## 位置编码的具体值  θ*position
    freqs = torch.outer(t, freqs).float()  ## 表示 位置*θ  shape是 (end, dim/2)
     ## 拼接cos和sin矩阵
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  ## freq的形状是 (end, dim/2)！！！，但现在seq形状是 (end, dim)；所以要拼一起，让俩维度能用同样的cos/sin矩阵
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin  ## 最后shape都是 (end, dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):  ## 这里用unsqueeze_dim是为了适配多头注意力的维度
    def rotate_half(x):  
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)  ## 把第二半部分负号翻到前面、前半部分放到后面，方便和 sin/cos 做旋转 （规定只要固定俩维度就行,不一定是相邻）

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim)) ## x2i′​=x2i​cos(θ)−x2i+1​sin(θ)  x2i+1′​=x2i​sin(θ)+x2i+1​cos(θ)  这里主要是实现这个公式
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:  ## 做n_rep次重复  用于GQA机制
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

## 实现attention  包含kv cache，YaRN，Flash Attention
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()                          ## GQA主要作用是效果不降多少的情况下，省显存跑的快
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0  ## GQA 机制 确保能整除 Q为多头，KV少头  原来是头数一样，一组跟一组来，现在是每四组Q用一组KV
        self.n_local_heads = args.num_attention_heads  ## q有8个头
        self.n_local_kv_heads = self.num_key_value_heads  ## kv有2个头
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  ## 重复几次  复用kv头4次
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  ## KV头只有2个
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  ## KV头只有2个
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  ## 是否有kv缓存
                use_cache=False,  ## 是否使用缓存
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape  ## 【1，seq, hiddin_size] 代表一句话

        ## 以下是attention的实现
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  ## shape (bsz, seq_len, n_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim) ## shape (bsz, seq_len, n_kv_heads, head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim) ## shape (bsz, seq_len, n_kv_heads, head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])   ## q，k内部各自进行旋转矩阵相乘，做位置编码   

        # kv_cache实现  直接用原来的past_key_value与当前的key_value做拼接
        if past_key_value is not None:  ## 如果是推理阶段,xk, xv 只包含【当前 forward 这次输入的 token】
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
       ## 直接用原来的past_key_value与当前的key_value做拼接
        xq, xk, xv = ( 
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2), ## 用GQA机制扩展kv头
            repeat_kv(xv, self.n_rep).transpose(1, 2) ## 用GQA机制扩展kv头
        )

        ## scaled_dot_product_attention满足条件时才会用 Flash Attention
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)  ## flash attention 自带 causal mask，对于上三角矩阵不进行attention计算

        ## 正常计算 attention
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  ## shape (bsz, n_heads, seq_len, seq_len)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),  ### 这是生成上三角矩阵的mask
                diagonal=1                    
            ).unsqueeze(0).unsqueeze(0)  # scores+casual mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)   ## 原来形状 (bsz, seq_len) -> [B, 1, 1, seq_len] 这里是mask padding的token
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask    ## 屏蔽 padding token / 无效 token  valid 为1 invalid为0 可以直接mask掉 加一个很大的负数；但这里只mask掉了key，query应该后面有写

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))  ## shape还是 (bsz, seq_len, hidden_size)！！！
        return output, past_kv  
    

## FFN的网络结构  输入时shape (bsz, seq_len, hidden_size)  输出时shape还是 (bsz, seq_len, hidden_size)
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act] ##SILU，就是x*sigmoid(x)

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))  ## 最后shape还是 (bsz, seq_len, hidden_size)


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  ## 最终选择到底几个专家  2
        self.n_routed_experts = config.n_routed_experts ## 总的专家数量  4

        self.scoring_func = config.scoring_func  ## 评分函数  softmax
        self.alpha = config.aux_loss_alpha  ## 辅助损失的alpha参数 0.01 >0
        self.seq_aux = config.seq_aux   # 是否在序列级别上计算辅助损失 ## True

        self.norm_topk_prob = config.norm_topk_prob ## 是否标准化top-k概率
        self.gating_dim = config.hidden_size  ## 和hidden size一样 512
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  ## 专家数是4  shape是 (4, 512)  empty意思是先不初始化，随便赋值
        self.reset_parameters() ## 初始化weight参数

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  ##初始化  只是借用了 Kaiming 初始化的公式来算一个特定的初始化区间，但并不是严格的 Kaiming 初始化，这个是专门为relu系列激活函数设计的初始化方法

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)   ## (bsz*seq_len, 512) 变成二维

         ## 计算每个token对于每个专家的打分
        logits = F.linear(hidden_states, self.weight, None)   ## (bsz*seq_len, n_routed_experts)  这里相当于转置乘法
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  ## (bsz*seq_len, top_k) 取前2个专家的分数和索引  就是取每个token对应分数最高的2个专家

        if self.top_k > 1 and self.norm_topk_prob:   ## 标准版topk概率 True
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator  ## 再做个归一化

        ## 下面纯算辅助loss 需要搞懂！！！
        if self.training and self.alpha > 0.0:  ## 只有训练阶段并且 alpha>0，才计算辅助 loss，否则 aux_loss = 0 即训练阶段算loss
            scores_for_aux = scores
            aux_topk = self.top_k ## 2
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) ## reshape 成 (bsz, seq_len*2)  方便按 batch 维度计算辅助 loss，而不管 token 和 top-k 的原始二维展开
            if self.seq_aux:  # 在序列级别上计算辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) ## (bsz, seq_len, n_routed_experts)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)  ## (bsz, n_routed_experts) 零初始化
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)  ## 计算在batch维度内  平均每个专家被选择的次数  ce.shape（bth,n_routed_experts)
                ## 这句相当于 对每个 batch,每当某个 token 选中了专家 j, 就给 ce[batch, j] + 1  再做归一化

                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha   ## 计算辅助 loss
            else:  ## 在token级别上计算辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)  ## one-hot 显示每个token的每两个选择究竟是哪个  (bsz*seq_len*2，4)
                ce = mask_ce.float().mean(0) ## 算每个expert 平均被选多少次 （4) --token级别  /b*s*n_routed_experts
                Pi = scores_for_aux.mean(0)  ## 算每个batch expert 的平均分  (4)  --token级别  /b*s*n_routed_experts
                fi = ce * self.n_routed_experts ## 为什么要乘4 -- 把频率转成频次
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss  ## topk_idx (bsz*seq_len, top_k)  topk_weight (bsz*seq_len, top_k) 返回对于top2的专家索引和权重，以及辅助loss
         

## MoE全过程结构！！！
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        ## expert本身就是个ffn
        self.experts = nn.ModuleList([
            FeedForward(config) ## 做ffn
            for _ in range(config.n_routed_experts)  ## 有四个专家  每个专家最后输出shape还是 (bsz, seq_len, hidden_size)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config) 
                for _ in range(config.n_shared_experts) ##  有一个共享专家  每个专家最后输出shape还是 (bsz, seq_len, hidden_size)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 原始输入做门控
        topk_idx, topk_weight, aux_loss = self.gate(x)  ## topk_idx (bsz*seq_len, top_k)  topk_weight (bsz*seq_len, top_k)

        ## 以下用门控乘以专家网络输出
        x = x.view(-1, x.shape[-1])  ## (bsz*seq_len, hidden_size)

         ## 展平 topk_idx 形状为 (bsz*seq_len*2)
        flat_topk_idx = topk_idx.view(-1)
        if self.training:                                                   ## 原本x是 (bsz*seq_len, hidden_size)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) ## (bsz*seq_len*num_experts, hidden_size)，把每个token的输入复制 num_experts_per_tok 次 即2次 因为选的俩个专家 这样能和前面展平的topk_idx对应上
            y = torch.empty_like(x, dtype=x.dtype)  ## 提前分配一块和 x 一模一样形状的输出缓存   (bsz*seq_len*num_experts, hidden_size)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i]) ## 返回True,False的mask，取出分配给第i个专家的token输入，送入该专家网络得到输出
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)  ##  给每个专家处理对应的token输入，得到输出后放回对应位置
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())  ## 避免没有token分配到该专家时，expert_out为空张量，导致后续scatter_add_报错
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)  ## (bsz*seq_len, hidden_size)  把专家输出乘以对应权重后加和，把对于应专家的输出加权求和  unsqueeze()是增加一个维度
            y = y.view(*orig_shape)  ## 恢复原始形状 (bsz, seq_len, hidden_size)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)  ## 共享专家的输出直接加到最终输出上
        self.aux_loss = aux_loss ## 保存辅助loss
        return y

    ## 推理阶段的moe实现
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)  ## 零初始化输出缓存 (bsz*seq_len, hidden_size)
        idxs = flat_expert_indices.argsort()  ## 从小到大 按专家索引排序，返回对应下标！！！  为了了后续按专家分组处理token输入
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) ## 计算每个专家分配到的token数量的累积和  shape是 (n_routed_experts, )
        token_idxs = idxs // self.config.num_experts_per_tok ## 计算每个位置对应expert要处理的token索引  shape是 (bsz*seq_len*num_experts, )
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        ## 看上述解释！！！！！！
        
        #接下来就是按着上述说的分组处理token输入
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


### 以下都为模块拼装

## 这里一层的block为 先过一层attention 再过一层moe/ffn
## 细节是 x会先过rmsnorm -- attention+旋转位置编码 -- 残差连接 -- rmsnorm -- moe/ffn -- 残差连接
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads  ##  8个
        self.hidden_size = config.hidden_size  ## 512
        self.head_dim = config.hidden_size // config.num_attention_heads ## 64
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) 
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value ## 返回当前层的输出和kv缓存



## 整个MiniMind模型结构  包含多层 MiniMindBlock
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)  ## 预计算位置编码的cos和sin矩阵
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  ## 过去有多少token  用于位置编码的起始

        hidden_states = self.dropout(self.embed_tokens(input_ids)) ## 先拿embedding ！！
        position_embeddings = (                                ## 再拿theta cos和sin矩阵 为了做位置编码 ！！
            self.freqs_cos[start_pos:start_pos + seq_length],  ## 过去位置的编码当作起始+这次的length，这里这么设计主要为了推理，一个一个推理输出；不是推理的话就直接seq_length 因为start_pos=0
            self.freqs_sin[start_pos:start_pos + seq_length]
        )  

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):  ## 过好几层layer 好几层block ！！ 位置cos，sin矩阵放进去为了做位置编码
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)  ## 每次都把每层的kv缓存存起来 ！！

        hidden_states = self.norm(hidden_states) ## 最后做个norm ！！

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss  ## 返回最后的输出，所有层的kv缓存，moe的辅助loss ！！



##  MiniMind用于因果语言模型的封装
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache, ## 可以决定是推理阶段还是训练阶段
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep  ## 推理阶段只取最后一个token，训练阶段取所有token
        logits = self.lm_head(hidden_states[:, slice_indices, :]) ## 计算logits  线性层，把 hidden_state 投影到词表空间
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
