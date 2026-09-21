# RAG 企业知识库项目

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/Milvus-2.x-brightgreen.svg)](https://milvus.io/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple.svg)](https://platform.deepseek.com/)

基于 LangGraph + Milvus 的自纠正、自适应 RAG 系统，面向企业私域知识库问答场景。示例知识库用的是 Kubernetes 官方文档（concepts + tasks），文档数据本身未随仓库提供（见下方「知识库数据」），可自行下载或替换成你自己的文档。

---

## 项目特性

- **混合检索 + Reranker 精排**：BM25 稀疏向量 + BGE 语义向量双路检索（各自召回 50 条），RRF（k=60）融合截断 20 条后经 BGE CrossEncoder 精排取 top-4；`grade_documents` 节点直接复用精排分数做阈值过滤，不再二次调 LLM
- **Small-to-big 父子分块**：检索单元是 200~300 字的小块（对齐 embedding/精排模型的最大输入长度），命中后按 `parent_id` 取回所在章节的父块（≤2000 字，超长退化为"命中小块+相邻小块"）供 LLM 生成，兼顾检索精度和生成上下文完整性
- **自纠正 RAG**：生成后经两道独立门禁节点——幻觉检测（`check_hallucination`）+ 答案质量评估（`check_answer`），不合格自动重试；两者均可通过 `config` 开关单独启停，关闭时改为响应后异步补跑（见「FastAPI 服务」）
- **Corrective RAG**：不做检索前路由，问题一律先检索，再用同一个 CrossEncoder 分数逐条过滤（`grade_documents`）——全部被过滤即判定知识库未覆盖，改写 1 次仍无结果自动降级 Tavily 网搜，网搜结果同样过一遍阈值过滤（`grade_web_results`）才进生成；网搜降级本身可通过部署级开关 `ENABLE_WEB_SEARCH` 整体关闭（纯内部专有知识库场景避免用网上通用做法冒充内部规范作答，关闭后直接判定"未找到"，见「常见问题」）
- **查询改写**：检索效果差时 LLM 分析语义意图重写 Query，再次检索（上限 1 次）
- **有状态对话**：LangGraph `PostgresSaver` checkpointing，通过 `thread_id` 隔离多会话，进程重启可续接
- **检索层权限过滤**：按 `dept_id` 做 Milvus pre-filter，越权文档在向量检索前即被排除，两路（dense + sparse）各自注入 expr
- **链路追踪**：每次问答写入 SQLite（`tracing.db`），记录各节点耗时、token 用量、检索结果、reranker 得分，支持 badcase 定位
- **文档生命周期管理**：支持单篇文档的新增、更新（幂等，内容未变自动跳过）、删除，CLI 工具 `manage_docs.py` 一行命令操作
- **FastAPI 服务层**：`POST /chat` + `GET /health`，同步图跑在线程池里不阻塞事件循环；默认走「生产快速路径」（跳过阻塞式质量门禁，按采样率在响应后台异步补跑质检，结果写入 `tracing.db`，不阻塞响应也不丢失可查询性）

---

## 整体架构

```
用户问题
  │
  ▼
[混合检索 Milvus + 权限过滤 dept_id pre-filter]
  BM25 + BGE 各召回50 → RRF(k=60)融合截断20 → CrossEncoder精排取top-4
  │
  ▼
[grade_documents：逐条按 reranker_score 阈值过滤]
  │
  ├─ 有文档 ──► [assemble_context：小块→父块拼装] ──► [生成 LLM]
  │
  └─ 无文档
       ├─ 还没改写过 → [transform_query 改写] → 回到最上面重新检索
       └─ 已改写过仍无结果 → 降级 web_search
                                 │
                          [Tavily检索5条 → 切分 → 复用同一个CrossEncoder精排]
                                 │
                          [grade_web_results：同款阈值过滤]
                                 │
                          ├─ 有内容过阈值 ──► [生成 LLM]
                          └─ 全部被过滤 ──► END（未找到资料）

[生成 LLM] ──► [check_hallucination] → 有幻觉则重试
        │
        ▼
   [check_answer] → 未解决则改写 Query 重新检索
        │
        ▼
       END
```

三个计数器各自兜底，防止任意一条路陷入死循环：

| 计数器 | 上限 | 超限去向 |
|--------|------|---------|
| `transform_count` | 1次 | 降级 Tavily 网搜（若 `ENABLE_WEB_SEARCH=false` 则直接判定未找到） |
| `generation_count` | 2次（即最多重试1次）| `hallucination_fallback`，不再输出模型答案，改为输出检索到的原始资料 |
| `not_useful_count` | 1次（即最多重试1次）| `not_useful_fallback`，输出"未找到相关资料"提示 |

`check_hallucination`、`check_answer` 是两个独立节点（各自独立 span，token 可分别统计），而非合并在一个条件边函数里；两者均可通过 `config["configurable"]["enable_hallucination_check"/"enable_answer_check"]` 关闭，跳过 LLM 调用直接判定通过（生产快速路径用）。

---

## 项目结构

```
RAG_PROJECT/
├── api/                 # FastAPI 服务层（POST /chat + GET /health）
├── llm_models/          # LLM & Embedding 模型统一配置
├── documents/           # 文档解析 & 向量库写入 & 生命周期管理
│   ├── markdown_parser.py   # MD 解析 + 语义分块
│   ├── milvus_db.py         # Collection 建表、连接、按 doc_id 增删
│   ├── write_milvus.py      # 双进程生产者-消费者批量入库，DIR_TO_DEPT 目录级权限映射
│   └── lifecycle.py         # 单篇文档 upsert / delete（幂等 + 回滚）
├── tools/               # 混合检索工具封装（含权限感知 auth_hybrid_search）
├── agent/               # 对话 Agent（Graph 1 带记忆版）
├── graph/               # Graph 1：基础 RAG + Agent Tool-Calling
├── graph2/              # Graph 2：完整自纠正 + Corrective RAG
│   ├── grade_documents_node.py  # 文档相关性过滤（reranker_score 阈值，全部被过滤即判定知识库未覆盖）
│   ├── grade_web_results_node.py  # web 检索结果的同款阈值过滤
│   ├── check_hallucination_node.py  # 幻觉检测（独立节点，可 config 开关）
│   └── check_answer_node.py         # 答案质量评估（独立节点，可 config 开关）
├── tracing/             # 链路追踪
│   ├── db.py                # SQLite 建表（spans + document_registry）& CRUD
│   ├── context.py           # TraceContext：trace_id、execution_index
│   └── handler.py           # TraceHandler：LangChain callback 埋点
├── utils/               # 日志、环境变量工具
├── tests/               # pytest 单元测试 + 批量评估脚本
├── datas/md/            # 原始 Markdown 知识库文档（K8s 示例数据未随仓库提供，见下文）
├── manage_docs.py       # 文档管理 CLI（list / status / update / delete）
├── tracing.db           # SQLite 数据库（运行时生成，链路追踪 + 文档注册表）
├── docker-compose.yml   # Milvus + Postgres 四容器编排
├── requirements.txt
└── .env.example
```

---

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 编排框架 | LangGraph | 有向状态图，支持循环和条件分支 |
| 向量库 | Milvus（Docker） | 原生混合检索，BM25 + HNSW，pre-filter 权限控制 |
| 会话持久化 | PostgreSQL（Docker）+ `PostgresSaver` | LangGraph checkpointing，进程重启可续接 |
| API 服务 | FastAPI + Uvicorn | `POST /chat` + `GET /health`，同步图跑线程池不阻塞事件循环 |
| Embedding | BGE-small-zh-v1.5（本地） | 中文优化，离线可用，分块和检索统一同一模型 |
| Reranker | BGE-reranker-base（本地） | CrossEncoder 精排，对 RRF 粗排 top-20 重排后取 top-4；分数复用做文档过滤阈值 |
| LLM | DeepSeek（官方 API） | Temperature=0，结构化输出用 function_calling |
| Web 搜索 | Tavily | 知识库覆盖不足时的降级方案 |
| 文档解析 | Unstructured | 按元素模式解析 Markdown |
| 语义分块 | LangChain SemanticChunker | 以语义边界切割，而非固定字符数 |
| 链路追踪 | SQLite + LangChain Callback | 本地轻量追踪，无需额外服务 |

---

## 快速开始

### 前置条件

- Docker Desktop（已启动）
- Python 3.10 + Anaconda
- DeepSeek API Key：[platform.deepseek.com](https://platform.deepseek.com)
- Tavily API Key：[tavily.com](https://tavily.com)

### 知识库数据

本项目的示例知识库是 Kubernetes 官方文档（`concepts/` + `tasks/`），但**文档数据本身没有随仓库提供**（体积原因，见 `.gitignore` 里的 `datas/md`）。跑通问答前需要自己准备 `.md` 文档放入 `datas/md/`：

- 用 K8s 官方文档：从 [kubernetes.io/docs](https://kubernetes.io/docs/) 或其 [GitHub 仓库](https://github.com/kubernetes/website)（`content/zh-cn/docs/concepts`、`content/zh-cn/docs/tasks`）拉取，保持 `concepts/xxx/*.md`、`tasks/xxx/*.md` 的两级目录结构（下面的权限映射按这个结构分类）
- 换成自己的文档：放入 `datas/md/` 后，需要同步改 `documents/write_milvus.py` 里的 `DIR_TO_DEPT` 目录映射——不需要改任何路由关键词表（本项目不做检索前路由，纯 Corrective RAG，见「整体架构」），检索+相关性过滤对任意领域都能直接用

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
POSTGRES_URI=postgresql://raguser:ragpass@localhost:15432/ragdb
NO_PROXY=localhost,127.0.0.1
```

`POSTGRES_URI` 对应 docker-compose 里 `rag-postgres` 服务的默认账号密码，用于 LangGraph 的 `PostgresSaver` 会话持久化，注意宿主机端口映射是 `15432`（容器内仍是 5432，见 docker-compose.yml）。`NO_PROXY` 这行必须保留，防止本地 Milvus / Postgres 连接被代理拦截。

### 3. 启动 Milvus + Postgres

```bash
docker compose up -d
```

启动后有四个容器：`rag-milvus-etcd`、`rag-milvus-minio`、`rag-milvus-standalone`（端口 `19530`）、`rag-postgres`（端口 `15432`→容器内 `5432`）。等待约 90 秒，确认全部状态均为 `healthy`：

```bash
docker compose ps
```

### 4. 数据入库（只需跑一次）

将 `.md` 文档按 `concepts/xxx/*.md`、`tasks/xxx/*.md` 的两级目录结构放入 `datas/md/`（见上方「知识库数据」），然后执行：

```bash
python -m documents.write_milvus
```

日志输出 `document_registry 注册完成，共 N 个文档` 即为成功。

文档的部门权限（`dept_id`）按一级子目录自动判断，映射表在 `documents/write_milvus.py` 的 `DIR_TO_DEPT`（如 `concepts/security` → `engineering`，`tasks/job` → `product`），目录不在映射表里默认归 `public`。换成自己的文档目录结构时需要同步改这份映射表。

> **注意**：本项目不做检索前路由（问题一律先检索、检索完再判断够不够用），换成自己的文档后不需要改任何关键词表；只需要确认 `graph2/grade_documents_node.py` 的 `RERANKER_THRESHOLD` 对你的语料是合适的（当前 `0.3`，是在 K8s 语料上标定出来的，换领域后语义相关性分数分布可能不同，建议重新过一遍标定，方法见 `verify/threshold_calibration.py` 的思路——该脚本本身未随仓库提供，见「项目结构」）。

### 5. 启动问答

**方式 A：CLI 交互**

```bash
python -m graph2.graph_2
```

进入交互循环，输入问题即可，输入 `q` 退出。

> 必须用 `python -m graph2.graph_2`（从项目根目录，作为模块运行），不能直接 `python graph2/graph_2.py`——后者会导致 `sys.path` 里没有项目根目录，触发 `ModuleNotFoundError: No module named 'tracing'`。

**方式 B：FastAPI 服务**

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI：http://localhost:8000/docs
- 健康检查：`GET /health`
- 问答接口：`POST /chat`，请求体 `{"question": "...", "dept_id": "public"}`

默认走「生产快速路径」：跳过阻塞式质量门禁（幻觉检测 + 答案质量评估），响应后由 `BackgroundTasks` 异步补跑质检，结果写日志；与 CLI 方式（两个质量门禁全开、同步阻塞）相比响应更快，但不保证已过质检。

---

## 文档管理

入库完成后，使用 `manage_docs.py` 对单篇文档进行增删改，无需重新全量入库：

```bash
# 查看所有已注册文档（doc_id、chunk 数、更新时间）
python manage_docs.py list

# 查看某篇文档详情（registry 记录 vs Milvus 实际 chunk 数对比）
python manage_docs.py status <doc_id>

# 新增或更新一篇文档（幂等：内容未变自动跳过，dept_id 按目录自动判断）
python manage_docs.py update datas/md/tasks/debug/troubleshoot-clusters.md

# 手动指定部门权限（覆盖目录自动映射）
python manage_docs.py update datas/md/concepts/security/xxx.md --dept engineering

# 删除一篇文档（会有确认提示）
python manage_docs.py delete <doc_id>
```

`dept_id` 可选值：`engineering` / `ops` / `product` / `public`

---

## 核心设计说明

### 为什么用混合检索 + Reranker 三层漏斗

纯语义检索对型号、专有名词等精确词汇不敏感；纯 BM25 不理解同义词和近义表达。BM25 负责精确词命中，BGE 负责语义泛化，RRF 融合取最优。

RRF 是 bi-encoder 级别的粗排（速度快，top-20 候选），召回率高但精度有限。CrossEncoder Reranker 对每个 `(query, chunk)` 对做完整交叉注意力打分，精度更高但计算量大，所以只对粗排的 20 个候选做精排，最终保留 top-4 送进生成。

### 为什么分块和检索都用 BGE，不用 OpenAI Embedding

两个模型的向量空间不对齐：用 A 模型切出的语义边界，B 模型检索时理解的"相似"是另一套度量。统一用 BGE 消除这个不一致性，同时 BGE 对中文语料更准，且完全离线无外部依赖。

### 为什么用 LangGraph 而不是普通 Chain

普通 Chain 是线性管道，无法表达循环重试和条件跳转。LangGraph 用 StateGraph 建模，天然支持有向图中的循环和条件边。

### 为什么权限过滤用 Milvus pre-filter 而不是结果层过滤

pre-filter 在向量检索前即排除越权文档，越权内容不参与 ANN 计算，不会出现在候选池里。结果层过滤是检索完再裁剪，越权文档已经参与了检索，top-k 名额被占用，最终有效结果数减少。两路（dense + sparse）必须各自注入 `expr`，否则 BM25 候选池仍有越权文档经 RRF 合并后排上来。

### 为什么文档更新用"先删后插"而不是"先插后删"

先插后删无空窗期，但需要在 schema 里加 `doc_version` 字段区分新旧 chunk。pymilvus 2.5.6 不支持在线加字段（无 `add_collection_field`），且 `content_hash` 等元数据本身也不应该进向量库（不参与检索，SQLite 更合适）。先删后插实现简单，空窗期为秒级（知识库场景更新频率低，可接受），delete 失败时有旧 chunk 快照自动回滚。

---

## 常见问题

**Q：Milvus 连接失败 `Connection refused`**

等待约 90 秒让容器完全启动，确认四个容器（`rag-milvus-etcd`、`rag-milvus-minio`、`rag-milvus-standalone`、`rag-postgres`）状态均为 `healthy`：

```bash
docker compose ps
```

若仍失败，重启：

```bash
docker compose restart
```

---

**Q：所有问题都走网络搜索，不走本地知识库**

不做检索前路由（Corrective RAG 设计，见「整体架构」），所有问题都先检索再判断——如果这里出问题，大概率是 `grade_documents` 把检索到的文档全过滤掉了。先确认 `graph2/grade_documents_node.py` 的 `RERANKER_THRESHOLD`（当前 `0.3`）和 `graph2/grade_web_results_node.py` 的 `WEB_RERANKER_THRESHOLD`（当前 `0.1`）对你的语料是不是设得过高，再确认 `datas/md/` 下的文档确实和问题相关领域一致。这两个值是在 K8s 示例语料上标定出来的，换成自己的文档后语义相关性分数分布可能不同，建议重新标定。

查看已注册文档：

```bash
python manage_docs.py list
```

---

**Q：想彻底关闭网络搜索降级，只用知识库自己的内容回答**

设置 `ENABLE_WEB_SEARCH=false`（`.env` 或环境变量）后重启服务。此时知识库未覆盖（改写1次仍无相关文档）会直接判定"未找到相关资料"，不再降级 Tavily 网搜——适合纯内部专有知识库场景，避免网上搜到的通用做法和内部规范冲突。默认是 `true`，保持网搜降级行为。这是进程启动时读一次的部署级配置，不支持单次请求覆盖。

---

**Q：`with_structured_output` 报 `BadRequestError`**

确认所有 `with_structured_output()` 调用均加了 `method="function_calling"`，DeepSeek 不支持 OpenAI 默认的 Structured Outputs 接口。

涉及文件：`graph2/grader_chain.py`（消融对照组，`grade_documents` 默认不再调用）、`graph2/grade_hallucinations_chain.py`、`graph2/grade_answer_chain.py`。

---

**Q：BGE 模型下载失败或速度慢**

首次运行会自动从 HuggingFace 下载 `BAAI/bge-small-zh-v1.5`（约 90MB）和 `BAAI/bge-reranker-base`。国内网络可设置镜像：

```bash
$env:HF_ENDPOINT="https://hf-mirror.com"  # Windows PowerShell
export HF_ENDPOINT=https://hf-mirror.com  # Linux/macOS
```

---

**Q：入库进程卡住不退出**

多进程写入时若解析进程异常退出，写入进程可能挂死。直接 `Ctrl+C` 终止后重新运行即可，Milvus 数据已持久化，不会丢失已写入的内容。

---

**Q：更新文档后 status 显示 registry 和 Milvus chunk 数不一致**

直接重新 update 该文档，`upsert_document` 会强制检测内容变化并重建 chunk：

```bash
python manage_docs.py update datas/md/<相对路径>.md
```
