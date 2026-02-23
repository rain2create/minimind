from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )  ## 先encode 成词表index ！！！

        ## 这里一条input可能有多轮对话，但是是pretrain没关系 都算进loss，做词语接龙做预测

        """""
        {
    'input_ids': tensor([[101, 7592, 2088, 102, 0, 0, ...]]),  # token id
    'attention_mask': tensor([[1, 1, 1, 1, 0, 0, ...]])      # 1表示有效token, 0表示pad
        }  这是encoding输出

        这里把输入x形式变成了   [B,seq], 一个seq是一句话多个token，这里规定了max_length，所以不足长度的要padding
        """
        ##  input_ids 这里是转换回的token id 总共代表一句话
        input_ids = encoding.input_ids.squeeze()  ## 原本形状是 (1,seq_len) 因为tokenizer输出是 (b,seq_len)，这里再压缩成一维 变成(seq_len)
        loss_mask = (input_ids != self.tokenizer.pad_token_id)  ## 返回True,False集合，告诉模型哪些是要算loss，哪些不算 哪些是paddiing的；只算有效的

        X = torch.tensor(input_ids[:-1], dtype=torch.long)  ## 作为模型输入，去掉最后一个token
        Y = torch.tensor(input_ids[1:], dtype=torch.long)   ## 作为模型输出 预测下一个目标
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  ## 对其目标Y 只计算有效token的loss
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids  ## assistan开始的标记
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    ## 这里的一条input是一个conversation，它里面有多段回答 通过打标识别出来后 模型再去学习
    ## 只计算回答的loss！！ 要让模型从一个词语接龙机器 真正变成 有效回答你问题的助手！！！

    """""
        JSON 数据
        │
        ▼
    {
    "conversations": [          ← 一条训练样本（一条input）
        {"role": "user", ...},    ← 第1轮问题（不学习）
        {"role": "assistant", ...}, ← 第1轮回答（✓ 学习）
        {"role": "user", ...},    ← 第2轮问题（不学习）
        {"role": "assistant", ...}, ← 第2轮回答（✓ 学习）
        {"role": "user", ...},    ← 第3轮问题（不学习）
        {"role": "assistant", ...}  ← 第3轮回答（✓ 学习）
    ]
    }
        │
        ▼
    create_chat_prompt()         ← 格式化成带标记的文本
        │
        ▼
    tokenizer() → input_ids       ← 转成数字序列
        │
        ▼
    generate_loss_mask()          ← 标记所有 assistant 位置
        │
        ▼
    loss_mask = [0,0,0, 1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1]
                ↑问题↑回答↑  ↑问题↑回答↑  ↑问题↑回答↑
                                
    模型训练：只在 mask=1 的位置计算 loss

    """

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids) # 初始全0（全都不学）
        i = 0
        while i < len(input_ids):
             # 找到 "assistant" 标记的位置 这里可以识别多个回答
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:  # assistant内容开始
                start = i + len(self.bos_id)  
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                 # 关键：只把 assistant 回答的部分设为 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1   # 这些位置才计算loss！
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self.generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask  ## 只对回答 assistant部分做训练


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        self.data = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }

if __name__ == "__main__":
    pass