from llama_cpp import Llama
import os
import time

abs_path = os.path.dirname(os.path.abspath(__file__))

class Llama_Red_Spider:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            verbose=False
        )


    def chat(self, input_sentence):
        # prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input_sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        # prompt = f"<|start_header_id|>user<|end_header_id|>\n{input_sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        # 加入 system prompt 约束输出长度
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            "你是一名中文医疗助手，回答简洁，不超过50字。<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{input_sentence}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )


        t1 = time.time()
        output = self.llm(prompt, max_tokens=64,
                          stop=["<|eot_id|>", "<|start_header_id|>"],
                          temperature=0.7)
        t2 = time.time()

        text = output['choices'][0]['text'].strip()
        # return text
        # 推理速度统计
        usage = output.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        elapsed = t2 - t1
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        print(f"[Llama] 生成耗时: {elapsed:.3f}s | "
              f"prompt tokens: {prompt_tokens} | "
              f"completion tokens: {completion_tokens} | "
              f"速度: {tokens_per_sec:.1f} tokens/s")

        return text, {
            'elapsed': round(elapsed, 3),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'tokens_per_sec': round(tokens_per_sec, 1),
        }