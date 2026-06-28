# RAG 企业知识库项目

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/Milvus-2.x-brightgreen.svg)](https://milvus.io/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple.svg)](https://platform.deepseek.com/)

基于 LangGraph + Milvus 的自纠正、自适应 RAG 系统，面向企业私域知识库问答场景。

---

## 项目特性

- **混合检索**：BM25 稀疏向量 + BGE 语义向量双路检索，RRF 融合排序
- **自纠正 RAG**：生成后经两道质量门禁——幻觉检测 + 答案质量评估，不合格自动重试
- **自适应 RAG**：入口路由自动判断走本地知识库还是网络搜索；检索失败超限自动降级 Tavily 网搜
- **查询改写**：检索效果差时 LLM 分析语义意图重写 Query，再次检索
- **有状态对话**：LangGraph `MemorySaver` checkpointing，通过 `thread_id` 隔离多会话

---

## 整体架构

```
用户问题
  │
  ├── [路由] → web_search → Tavily 搜索 ──────────────────┐
  │                                                        │
  └── [路由] → vectorstore                                 │
                   │                                       ▼
           [混合检索 Milvus]                          [生成 LLM]
           BM25 + BGE + RRF                               │
                   │                              [幻觉检测] → 重试
           [文档相关性过滤]                                │
                   │                              [答案质量] → 改写 Query
           有文档 → 生成                                   │
           无文档 → 改写 Query → 重新检索（最多2次）      END
```

三个计数器各自兜底，防止任意一条路陷入死循环：

| 计数器 | 上限 | 超限去向 |
|--------|------|---------|
| `transform_count` | 2次 | 降级 Tavily 网搜 |
| `generation_count` | 3次 | END，输出最后一次结果 |
| `not_useful_count` | 2次 | END，输出最后一次结果 |

---

## 项目结构

```
RAG_PROJECT/
├── llm_models/        # LLM & Embedding 模型统一配置
├── documents/         # 文档解析 & 向量库写入
│   ├── markdown_parser.py   # MD 解析 + 语义分块
│   ├── milvus_db.py         # Collection 建表 & 连接
│   └── write_milvus.py      # 双进程生产者-消费者写入
├── tools/             # 混合检索工具封装（LangChain Tool）
├── agent/             # 对话 Agent（Graph 1 带记忆版）
├── graph/             # Graph 1：基础 RAG + Agent Tool-Calling
├── graph2/            # Graph 2：完整自纠正 + 自适应 RAG
├── utils/             # 日志、环境变量工具
├── tests/             # pytest 单元测试
├── datas/md/          # 原始 Markdown 知识库文档
├── docker-compose.yml # Milvus 三容器编排
├── requirements.txt
└── .env.example
```

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 编排框架 | LangGraph | 有向状态图，支持循环和条件分支 |
| 向量库 | Milvus（Docker） | 原生混合检索，BM25 + HNSW |
| Embedding | BGE-small-zh-v1.5（本地） | 中文优化，离线可用，分块和检索统一同一模型 |
| LLM | DeepSeek（官方 API） | Temperature=0，结构化输出用 function_calling |
| Web 搜索 | Tavily | 知识库覆盖不足时的降级方案 |
| 文档解析 | Unstructured | 按元素模式解析 Markdown |
| 语义分块 | LangChain SemanticChunker | 以语义边界切割，而非固定字符数 |

---

## 快速开始

### 前置条件

- Docker Desktop（已启动）
- Python 3.10 + Anaconda
- DeepSeek API Key：[platform.deepseek.com](https://platform.deepseek.com)
- Tavily API Key：[tavily.com](https://tavily.com)

### 1. 创建环境 & 安装依赖

```bash
conda create -n rag_project python=3.10 -y
conda activate rag_project
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```
DEEPSEEK_API_KEY=sk-你的密钥
TAVILY_API_KEY=tvly-你的密钥
NO_PROXY=localhost,127.0.0.1
```

### 3. 启动 Milvus

```bash
docker compose up -d
```

等待约 90 秒，确认三个容器状态均为 `healthy`：

```bash
docker ps
```

### 4. 数据入库（只需跑一次）

将你自己的 `.md` 文档放入 `datas/md/` 目录，然后执行：

```bash
python documents/write_milvus.py
```

日志输出 `写入进程结束，总计写入 N 个文档` 即为成功。

> **注意**：放入新文档后，需同步修改 `graph2/query_route_chain.py` 中的 system prompt 以及 `tools/retriever_tools.py` 中的工具描述，将其中对知识库内容的描述改为你自己文档的领域，否则路由会失效（所有问题都走网络搜索）。

### 5. 验证入库结果

```bash
python _check_milvus.py
```

### 6. 启动问答

```bash
python graph2/graph_2.py
```

进入交互循环，输入问题即可，输入 `q` 退出。

---

## 核心设计说明

### 为什么用混合检索

纯语义检索对型号、专有名词等精确词汇不敏感；纯 BM25 不理解同义词和近义表达。BM25 负责精确词命中，BGE 负责语义泛化，RRF 融合取最优。

### 为什么分块和检索都用 BGE，不用 OpenAI Embedding

两个模型的向量空间不对齐：用 A 模型切出的语义边界，B 模型检索时理解的"相似"是另一套度量。统一用 BGE 消除这个不一致性，同时 BGE 对中文语料更准，且完全离线无外部依赖。

### 为什么用 LangGraph 而不是普通 Chain

普通 Chain 是线性管道，无法表达循环重试和条件跳转。LangGraph 用 StateGraph 建模，天然支持有向图中的循环和条件边。

### 为什么评分节点用结构化输出

评分节点需要机器可读的 `yes/no` 结果来控制流程分支。用 Pydantic 模型约束输出（`method="function_calling"`），防止 LLM 自由发挥导致分支判断失效。

---

## 常见问题

**Q：Milvus 连接失败 `Connection refused`**

等待约 90 秒让容器完全启动，确认三个容器状态均为 `healthy`：

```bash
docker ps | grep milvus
```

若仍失败，重启 Milvus：

```bash
docker compose restart
```

---

**Q：所有问题都走网络搜索，不走本地知识库**

检查 `graph2/query_route_chain.py` 的 system prompt 描述是否与实际入库的内容领域一致。

验证知识库是否有数据：

```bash
python _check_milvus.py
```

---

**Q：`with_structured_output` 报 `BadRequestError`**

确认所有 `with_structured_output()` 调用均加了 `method="function_calling"`，DeepSeek 不支持 OpenAI 默认的 Structured Outputs 接口。

涉及文件：`graph2/query_route_chain.py`、`graph2/grader_chain.py`、`graph2/grade_hallucinations_chain.py`、`graph2/grade_answer_chain.py`

---

**Q：BGE 模型下载失败或速度慢**

首次运行会自动从 HuggingFace 下载 `BAAI/bge-small-zh-v1.5`（约 90MB）。国内网络可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com  # Linux/macOS
$env:HF_ENDPOINT="https://hf-mirror.com"  # Windows PowerShell
```

---

**Q：入库进程卡住不退出**

多进程写入时若解析进程异常退出，写入进程可能挂死。直接 `Ctrl+C` 终止后重新运行即可，Milvus 数据已持久化，不会丢失已写入的内容。
