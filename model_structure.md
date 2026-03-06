# Qwen2_5_VLMoEForAction 模型结构

```mermaid
graph TD
    subgraph Qwen2_5_VLMoEForAction
        A[视觉编码器 visual]
        B[语言模型 model]
        C[输出层 lm_head]
        D[损失函数 loss_fct]
        E[动作处理器 action_preprocessor]
    end

    subgraph visual
        A1[补丁嵌入 patch_embed]
        A2[旋转位置编码 rotary_pos_emb]
        A3[视觉块 blocks]
        A4[补丁合并器 merger]
    end

    subgraph model
        B1[词嵌入 embed_tokens]
        B2[解码器层 layers]
        B3[层归一化 norms]
        B4[旋转位置编码 rotary_emb]
    end

    subgraph layers
        B2_1[Qwen2_5_VLDecoderLayer_with_MoE 1]
        B2_2[Qwen2_5_VLDecoderLayer_with_MoE 2]
        B2_3[...]
        B2_36[Qwen2_5_VLDecoderLayer_with_MoE 36]
    end

    subgraph decoder_layer[Qwen2_5_VLDecoderLayer_with_MoE]
        B2a[自注意力 self_attn]
        B2b[输入层归一化 input_layernorms]
        B2c[注意力后层归一化 post_attention_layernorms]
        B2d[路由器 router]
        B2e[混合专家 moe]
    end

    subgraph self_attn
        B2a1[查询投影 q_proj_experts]
        B2a2[键投影 k_proj_experts]
        B2a3[值投影 v_proj_experts]
        B2a4[输出投影 o_proj_experts]
        B2a5[旋转位置编码 rotary_emb]
    end

    subgraph moe
        B2e1[专家列表 experts]
    end

    subgraph experts
        B2e1_1[专家 0 BlockSparseMLP]
        B2e1_2[专家 1 BlockSparseMLP]
    end

    subgraph action_preprocessor
        E1[属性投影 propri_proj]
        E2[时间嵌入 time_embed]
        E3[线性层 w1]
        E4[线性层 w2]
        E5[线性层 w3]
        E6[激活函数 act_fn]
        E7[动作投影 action_proj_back]
        E8[MSE 损失 mse_loss]
    end

    A --> A1
    A --> A2
    A --> A3
    A --> A4

    B --> B1
    B --> B2
    B --> B3
    B --> B4

    B2 --> B2_1
    B2 --> B2_2
    B2 --> B2_3
    B2 --> B2_36

    B2_1 --> decoder_layer
    B2_2 --> decoder_layer
    B2_36 --> decoder_layer

    decoder_layer --> B2a
    decoder_layer --> B2b
    decoder_layer --> B2c
    decoder_layer --> B2d
    decoder_layer --> B2e

    B2a --> B2a1
    B2a --> B2a2
    B2a --> B2a3
    B2a --> B2a4
    B2a --> B2a5

    B2e --> B2e1
    B2e1 --> B2e1_1
    B2e1 --> B2e1_2

    E --> E1
    E --> E2
    E --> E3
    E --> E4
    E --> E5
    E --> E6
    E --> E7
    E --> E8
```

## 模型结构说明

### 1. 主模型组件
- **视觉编码器 (visual)**：处理输入图像，提取视觉特征
- **语言模型 (model)**：处理文本输入，生成文本输出
- **输出层 (lm_head)**：将语言模型的输出转换为词表概率分布
- **损失函数 (loss_fct)**：计算训练过程中的损失
- **动作处理器 (action_preprocessor)**：处理动作相关的输入和输出

### 2. 视觉编码器
- **补丁嵌入 (patch_embed)**：将图像分割为补丁并嵌入为向量
- **旋转位置编码 (rotary_pos_emb)**：为视觉特征添加位置信息
- **视觉块 (blocks)**：包含 32 个视觉注意力块，提取视觉特征
- **补丁合并器 (merger)**：合并视觉补丁特征

### 3. 语言模型
- **词嵌入 (embed_tokens)**：将输入词转换为向量表示
- **解码器层 (layers)**：包含 36 个解码器层，每个层包含自注意力和前馈网络
- **层归一化 (norms)**：对输入进行层归一化
- **旋转位置编码 (rotary_emb)**：为文本特征添加位置信息

### 4. 解码器层
- **自注意力 (self_attn)**：处理序列内的依赖关系
  - **查询/键/值/输出投影 (q/k/v/o_proj_experts)**：每个都包含 2 个专家
- **输入层归一化 (input_layernorms)**：对输入进行层归一化
- **注意力后层归一化 (post_attention_layernorms)**：对注意力输出进行层归一化
- **路由器 (router)**：根据输入令牌类型选择专家
- **混合专家 (moe)**：包含 2 个专家，根据路由器的输出加权组合

### 5. 动作处理器
- **属性投影 (propri_proj)**：处理动作属性输入
- **时间嵌入 (time_embed)**：为动作添加时间信息
- **线性层 (w1, w2, w3)**：处理动作特征
- **激活函数 (act_fn)**：引入非线性
- **动作投影 (action_proj_back)**：将特征转换为动作输出
- **MSE 损失 (mse_loss)**：计算动作预测的损失

## LoRA 微调建议

对于全量 LoRA 微调，建议包含以下目标模块：

```yaml
lora_target_modules: [
  "embed_tokens",
  "q_proj_experts",
  "k_proj_experts",
  "v_proj_experts",
  "o_proj_experts",
  "moe.experts",
  "lm_head",
  "action_preprocessor"
]
```

这样可以对模型的所有关键组件进行 LoRA 微调，同时保持计算效率。