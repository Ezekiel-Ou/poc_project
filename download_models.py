#!/usr/bin/env python
# coding=utf-8
"""
下载所需的模型（使用 HuggingFace 镜像站加速）
- 向量模型：simcse-chinese-roberta-wwm-ext (~1GB)
"""

import os
import sys
from pathlib import Path

# 使用 HuggingFace 镜像站加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("使用 HF 镜像站: https://hf-mirror.com")

from transformers import AutoTokenizer, AutoModel


def ensure_models_dir():
    """确保 models 目录存在"""
    models_dir = Path(__file__).parent / "models"
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def download_vector_model(models_dir):
    """下载向量模型：simcse-chinese-roberta-wwm-ext"""
    model_name = "cyclone/simcse-chinese-roberta-wwm-ext"
    save_path = models_dir / "simcse-chinese-roberta-wwm-ext"
    
    
    print(f"\n{'='*60}")
    print(f"下载向量模型: {model_name}")
    print(f"保存路径: {save_path}")
    print(f"{'='*60}")
    
    try:
        os.makedirs(save_path, exist_ok=True)
        
        # 下载 tokenizer
        print("正在下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("✓ Tokenizer 下载完成")
        
        # 下载模型权重
        print("正在下载模型权重...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(save_path)
        print("✓ 模型权重下载完成")
        
        print(f"✓ 向量模型已保存到: {save_path}\n")
        return True
    except Exception as e:
        print(f"✗ 下载向量模型失败: {e}\n")
        return False

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("模型下载工具 - Qwen3 + SimCSE")
    print(f"{'='*60}\n")
    
    models_dir = ensure_models_dir()
    
    # 下载向量模型（必需）
    vec_ok = download_vector_model(models_dir)
    if not vec_ok:
        print("✗ 向量模型下载失败，请检查网络连接")
        sys.exit(1)
    
    # 总结
    print(f"{'='*60}")
    if vec_ok :
        print("模型下载完成！")
        print("\n下一步：")
        print("1. cd llm_classification")
        print("2. source ../default.env")
        print("3. python script/build_vec_index.py")
    else:
        print("✗ 下载失败，请重试")
        sys.exit(1)
    print(f"{'='*60}\n")
