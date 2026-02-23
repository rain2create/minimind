# 注：不建议再重复训练tokenizer（“词典”），MiniMind已自带，此脚本仅供学习和参考。基于不同词典训练的模型将导致输出完全不统一，降低社区的模型复用性
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

DATA_PATH = '../dataset/pretrain_hq.jsonl' 
TOKENIZER_DIR = '../model_learn_tokenizer/'
VOCAB_SIZE = 6400


## 只用第一行实验
def get_texts(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000: break # 实验性，可只用前10000行测试
            data = json.loads(line)
            yield data['text']



## tokenizer功能是主要是把文本转成索引，然后去查表得到emb；emb矩阵是根据vobca表来的，文本转为token后，一个token即一个索引也就只有一个emb
## tokenizer不会也不能按照文本的词或者字去切分来做索引，因为那样词表太大，有数不清的新词以及语义太碎
## 一个有效的tokenizer是能用 一个有限词表，尽可能短的seq，少的token去表示文本 -- 压缩率较高 -- 字符串数/token数 ！！！！
## BPE是常用的方法 
## 字符级别的BPE，单就是先统计文本词频，把出现次数多的相邻的合并在一起给一个index，不停迭代，靠这样来做tokenizer，因为要用少的seq 代表多的文本！！！
## Byte-level的bpe，算法一样，只是统计最小的byte词频（一个byte只有256种表示），但是一个字符串可能有多个byte(字节), 把出现次数多的相邻的byte合在一起 变成一个新index -256 以此类推 
## 词表大的tokenizer相当于表达能力也就越丰富!!! 一般来说

'''        （可能遇到新词）      （只有256个byte,所以初始索引就是这么多，再碰到新词它也能分成已知的byte然后用已知索引来代表）
"维度	     字符级 BPE	               Byte-level BPE
最小单位	Unicode 字符	             UTF-8 byte
支持语言	特定语言，字符集受限	所有语言、符号、emoji 都行
OOV 风险	    存在	                   不存在
token 数	   一般少	            token 数多一点，序列稍长 -- 一个字符串可能有多个字节
典型模型	BERT / SentencePiece	     GPT / LLaMA"
'''


def train_tokenizer(data_path, tokenizer_dir, vocab_size):

    tokenizer = Tokenizer(models.BPE())  ## 初始化tokenizer，表示用bpe算法--词频merge
    
    ## 定义pre_tokenizer 切分文本成字节
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  ## 把原始文本切成最小符号序列--字节，供 BPE 训练;  告诉 tokenizer：在训练 BPE 之前，先把文本切成「字节级 token」

    ## 定义trainer
    trainer = trainers.BpeTrainer(  ## tokenizer会在pre_tokenizers输出的序列上做训练 ！！！
        vocab_size=vocab_size, ## 可以规定 词表大小
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"], ## 特殊符号 加入初始索引，代表开始或者结束
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  ## 在 BPE merge 之前，先把「256 个 byte token」直接加入 vocab；定义初始字母索引表 表示最初就有256个索引 在这基础上新增
    )
    texts = get_texts(data_path)  ## 喂数据

    # 正式训练                                              ## 就是bpe训练流程
    tokenizer.train_from_iterator(texts, trainer=trainer)  ## 把原始文本一条条读进来 → 先按 pre_tokenizer 切成最小 token → 再用 trainer（BPE）统计频率、不断 merge → 最终学出 vocab 和 merges，写进 tokenizer。 
    tokenizer.decoder = decoders.ByteLevel()  ## 索引 -- decode --转为字符串 里面有俩步 1. 先把id转为byte  2. 再把byte映射回字符串

    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

## 上面是检查特殊字符是否安置好


    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")


def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '你来自哪里？'},
        {"role": "assistant", "content": '我来自地球'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-'*100)
    print(new_prompt)


    print('-'*100)
    print('tokenizer词表长度：', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('encoder长度：', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder一致性：', response == new_prompt, "\n")


    print('-'*100)
    print('流式解码（字节缓冲）测试：')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
