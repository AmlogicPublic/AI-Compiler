"""BGE Micro V2 - 文本嵌入模型 Demo
用途: 计算文本语义相似度, 检索, 聚类
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F
from demo.shared import load_model


def mean_pooling(model_output, attention_mask):
    """平均池化 (masked)"""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode(model, tokenizer, texts):
    """编码文本为向量"""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def cosine_sim(a, b):
    """余弦相似度"""
    return (a @ b.T).item()


if __name__ == "__main__":
    model, config, tokenizer = load_model("bge-micro-v2")
    
    # 查询和候选文档
    query = "如何学习机器学习?"
    docs = [
        "机器学习入门需要掌握Python编程和线性代数基础",
        "今天天气真好，适合出去散步",
        "深度学习是机器学习的一个重要分支",
        "我最喜欢的食物是披萨",
        "神经网络训练需要大量的数据和计算资源",
    ]
    
    print(f"\n查询: {query}")
    print("-" * 50)
    
    query_emb = encode(model, tokenizer, [query])
    doc_embs = encode(model, tokenizer, docs)
    
    # 计算相似度并排序
    scores = [(cosine_sim(query_emb, doc_embs[i:i+1]), docs[i]) for i in range(len(docs))]
    scores.sort(reverse=True)
    
    print("相似度排序:")
    for score, doc in scores:
        print(f"  {score:.4f} | {doc}")
    
    # 交互模式
    print("\n" + "=" * 50)
    print("交互模式 (输入查询, 按 Ctrl+C 退出)")
    print("=" * 50)
    
    while True:
        try:
            q = input("\n查询> ").strip()
            if not q:
                continue
            q_emb = encode(model, tokenizer, [q])
            scores = [(cosine_sim(q_emb, doc_embs[i:i+1]), docs[i]) for i in range(len(docs))]
            scores.sort(reverse=True)
            for score, doc in scores:
                print(f"  {score:.4f} | {doc}")
        except KeyboardInterrupt:
            print("\n退出")
            break
