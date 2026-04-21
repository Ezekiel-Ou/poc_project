# coding=utf-8
# Filename:    llm_model.py
# Author:      ZENGGUANRONG
# Date:        2023-12-17
# description: 大模型调用模块，这里默认用的chatglm2

# from transformers import AutoModel, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from loguru import logger
from src.utils.device import resolve_device

class QWen3Model:
    def __init__(self, model_path, config = {}, device="cuda"):
        self.device = resolve_device(device)

        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # decoder-only 模型必须左填充，否则 pad token 会出现在 prompt 中间导致生成错乱
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.model.eval()

        self.generate_config = self._read_config_(config)
        if not self.generate_config.get("do_sample", False):
            self.model.generation_config.temperature = 1.0
            self.model.generation_config.top_p = 1.0
            self.model.generation_config.top_k = 50
        logger.info("load LLM Model done")

    def _read_config_(self, config):
        tmp_config = {}
        tmp_config["num_beams"] = config.get("num_beams", 1)
        do_sample = config.get("do_sample", False)
        tmp_config["do_sample"] = do_sample
        if do_sample:
            tmp_config["top_k"] = config.get("top_k", 50)
            tmp_config["top_p"] = config.get("top_p", 1.0)
            tmp_config["temperature"] = config.get("temperature", 1.0)
        return tmp_config

    def _render_prompt(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def predict_batch(self, queries: List[str]) -> List[str]:
        if not queries:
            return []

        texts = [self._render_prompt(q) for q in queries]
        model_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=512,
            **self.generate_config,
        )

        input_len = model_inputs.input_ids.shape[1]
        trimmed = generated_ids[:, input_len:]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)

    def predict(self, query):
        return self.predict_batch([query])[0]

if __name__ == "__main__":
    from config.project_config import LLM_CONFIG, LLM_PATH
    print(LLM_CONFIG)
    llm_model = QWen3Model(LLM_PATH, config = LLM_CONFIG, device=None)
    print(llm_model.predict("如何做番茄炒蛋"))
