# MultiHop-RAG Dataset Processing

MultiHop-RAG 数据集处理脚本，用于对 corpus 进行分块、embedding 和索引构建。

## 数据集信息

- **Corpus**: 609 个文档 (title, author, source, published_at, category, url, body)
- **Queries**: 2,556 个多跳问答查询 (query, answer, question_type, evidence_list)
- **数据源**: HuggingFace `yixuantt/MultiHopRAG`

## 快速开始

```bash
# 基本用法
python eval_multihop_rag.py

# 使用脚本
chmod +x eval_multihop_rag.sh
./eval_multihop_rag.sh
```

## 配置参数

```bash
python eval_multihop_rag.py \
  --output-dir ./multihop_rag_output \
  --embedding-api-url http://localhost:30000/v1/embeddings \
  --embedding-model Alibaba-NLP/gte-Qwen2-7B-instruct \
  --chunk-size 1024 \
  --chunk-overlap 100 \
  --top-k 10
```

## 输出文件

- `chunks.jsonl`: 所有文档分块（metadata chunk + body chunks）
- `index.faiss` + `index.faiss.meta`: FAISS 向量索引
- `chunk_id_mapping.json`: 索引位置到 chunk_id 的映射
- `queries.jsonl`: 查询及其检索结果

## 处理流程

1. 从 HuggingFace 加载 corpus 和 queries 数据集
2. 对每个文档创建 metadata chunk（title + 元信息）
3. 对 body 字段进行分块（1024 tokens，100 overlap）
4. 使用 gte-Qwen2-7B-Instruct 生成 embeddings
5. 构建 FAISS 索引（cosine similarity）
6. 对所有 queries 进行检索（top-k）
7. 保存所有结果到输出目录

## 前置要求

- 已安装 basic_rag 包
- Embedding API 服务运行中（默认 `http://localhost:30000/v1/embeddings`）
- 安装 `datasets` 库: `pip install datasets`
