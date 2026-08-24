"""
批量 RAG 评估脚本（RAGAS）

用途：
  记录当前配置下的 RAG 质量基线，供消融实验前后对比使用。
  当前基线：grade_documents 使用 DeepSeek LLM 打分。
  对比目标：grade_documents 改为 bge-reranker-base cross-encoder 后重跑。

用法：
  pip install ragas datasets       # 首次安装
  python tests/batch_ragas_eval.py                     # 全量跑（约30-40分钟）
  python tests/batch_ragas_eval.py --label optimized   # 优化后对比跑
  python tests/batch_ragas_eval.py --limit 5           # 快速验证脚本是否跑通

输出（eval_results/ 目录）：
  {timestamp}_{label}_raw.json    每条问题的路由、检索到的 context、生成答案
  {timestamp}_{label}_scores.json RAGAS 指标汇总（faithfulness / answer_relevancy / context_relevancy）

消融实验关键指标：
  context_relevancy  — 衡量 grade_documents 过滤效果（cross-encoder 是否比 LLM 更准）
  faithfulness       — 生成答案有无幻觉
  answer_relevancy   — 答案是否回答了问题
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracing.context import TraceContext
from tracing.handler import TraceHandler

# ── 测试问题集（93条：原58条 + 多chunk综合题10 + 知识库外10 + 语义不匹配15）──────
QUESTIONS = [
    # 架构与核心概念
    "CGroup v2 相比 v1 做了哪些改进？",
    "Kubernetes 节点上的 kubelet 的主要职责是什么？",
    "控制平面和工作节点之间的通信方式有哪些？",
    "Kubernetes 中的 controller 是什么，它如何实现自愈？",
    "什么是 Lease 对象，它在 Kubernetes 中有什么用途？",
    "Kubernetes 的垃圾回收机制是如何工作的？",
    "Cloud Controller Manager 负责哪些功能？",
    "什么是 mixed-version proxy，解决了什么问题？",
    "Kubernetes API 的扩展方式有哪些？",
    "什么是 Finalizer，删除带 Finalizer 的对象时会发生什么？",
    "Label 和 Annotation 的区别是什么，各自的使用场景？",
    "Namespace 的作用是什么，哪些资源是 namespace 级别的？",
    # 工作负载
    "Deployment 滚动更新的默认策略是什么，如何控制更新速度？",
    "StatefulSet 和 Deployment 的核心区别是什么？",
    "DaemonSet 的典型使用场景有哪些？",
    "CronJob 和 Job 的关系是什么，CronJob 如何处理并发执行？",
    "ReplicaSet 和 Deployment 的关系是什么？",
    "Pod 的生命周期有哪些阶段？",
    "Init Container 和普通容器的执行顺序是什么？",
    "Sidecar Container 和 Init Container 的区别？",
    "Pod 的 QoS 等级有哪几种，如何判断一个 Pod 属于哪个等级？",
    "HPA 是如何根据指标触发扩缩容的？",
    "Pod Disruption Budget 的作用是什么？",
    "Ephemeral Container 是什么，它的主要用途是什么？",
    # 服务与网络
    "Kubernetes Service 有哪几种类型，各自的适用场景？",
    "ClusterIP 是如何分配的，用完了会怎样？",
    "Ingress 和 Service 的区别是什么？",
    "EndpointSlice 解决了 Endpoint 的什么问题？",
    "NetworkPolicy 如何控制 Pod 之间的流量？",
    "DNS 在 Kubernetes 中如何为 Service 和 Pod 解析名称？",
    "什么是拓扑感知路由，它的作用是什么？",
    "Kubernetes 如何支持 IPv4/IPv6 双栈？",
    # 存储
    "PersistentVolume 和 PersistentVolumeClaim 的关系是什么？",
    "StorageClass 的作用是什么，什么是动态供应？",
    "Volume 的 accessModes 有哪几种，区别是什么？",
    "什么是 VolumeSnapshot，它能做什么？",
    "临时卷有哪些类型，和普通 Volume 的区别？",
    "存储容量跟踪解决了什么问题？",
    "什么是 Projected Volume，可以把哪些来源挂载进来？",
    # 配置与安全
    "ConfigMap 和 Secret 的区别是什么，Secret 有哪几种类型？",
    "Liveness Probe、Readiness Probe 和 Startup Probe 各自的作用？",
    "容器的资源 requests 和 limits 分别控制什么，不设置 limits 有什么风险？",
    "RBAC 中 Role 和 ClusterRole 的区别？",
    "Pod Security Standards 定义了哪几个安全级别？",
    "ServiceAccount 的作用是什么，Pod 如何使用它访问 API？",
    "Secret 的最佳实践有哪些？",
    "kubeconfig 文件包含哪几个关键部分？",
    # 调度与故障排查
    "kube-scheduler 调度一个 Pod 分哪几个阶段？",
    "Taint 和 Toleration 如何配合使用，典型场景是什么？",
    "节点压力驱逐什么时候触发，驱逐顺序是什么？",
    "Pod 一直处于 Pending 状态，排查思路是什么？",
    "Pod CrashLoopBackOff 通常由哪些原因引起，如何排查？",
    "Service 无法访问时，应该从哪几个层面排查？",
    "DNS 解析失败怎么排查？",
    "如何安全地驱逐一个节点？",
    "如何用 kubectl 进入一个正在运行的容器调试？",
    "StatefulSet 的 Pod 卡在 Terminating 状态怎么处理？",
    "如何查看 Pod 失败的原因，包括已退出的容器日志？",

    # ── 多 chunk 综合题（需要跨文档合并信息才能完整回答）────────────────────────
    "Deployment、StatefulSet、DaemonSet 三种控制器分别适合什么场景，如何在它们之间做选择？",
    "Pod 从提交到真正运行的完整流程是什么，哪些组件依次参与？",
    "Kubernetes 的认证、授权、准入控制三个阶段各自做什么，一个请求通过的完整链路是什么？",
    "Liveness、Readiness、Startup 三种探针在 Pod 生命周期中各自的触发时机和失败后果是什么？",
    "K8s 网络模型中，一个外部请求进来到达 Pod 经过了哪些组件，各自负责什么？",
    "PVC 绑定到 PV 的完整流程是什么，动态供应和静态供应有什么区别？",
    "控制平面各组件（API Server、etcd、scheduler、controller-manager）分别负责什么，如何协作？",
    "HPA、VPA、Cluster Autoscaler 分别解决什么层面的扩缩容问题，有什么区别？",
    "ConfigMap 和 Secret 都可以作为环境变量或 Volume 挂载，两种使用方式各有什么优缺点？",
    "当一个节点出现故障时，Kubernetes 会经历哪些自愈步骤，各组件如何参与？",

    # ── 知识库外的题（不含 K8s 关键词，应路由到 web_search）────────────────────
    "Helm Chart 的目录结构是什么，values.yaml 的作用是什么？",
    "ArgoCD 如何实现 GitOps，Application 资源的同步策略有哪些？",
    "Prometheus 的 PromQL 中 rate() 和 irate() 的区别是什么，各自适合什么场景？",
    "Fluentd 和 Fluent Bit 的区别是什么，各自适合什么场景？",
    "Terraform state 文件的作用是什么，为什么不能直接手动修改？",
    "Harbor 镜像仓库如何配置垃圾回收策略？",
    "Velero 的备份原理是什么，它如何实现数据恢复？",
    "Jaeger 分布式追踪中 Span 和 Trace 的关系是什么？",
    "OPA（Open Policy Agent）是什么，Gatekeeper 如何用它来约束资源创建？",
    "Crossplane 和 Terraform 在管理云资源方面有什么本质区别？",

    # ── 字面不匹配但语义相关（问法口语化，测 reranker 语义桥接能力）────────────
    "为什么我的 Pod 总是被 kill 掉，能不能让 Kubernetes 知道它还活着？",
    "节点资源快不够用了，怎么防止重要的 Pod 被驱逐？",
    "怎么让同一个集群里的不同团队互不干扰，配额和权限分开？",
    "应用启动很慢，怎么告诉 Kubernetes 它还没准备好接流量？",
    "同一套配置想跑多个有状态的实例，每个实例有自己的独立存储，怎么做？",
    "怎么防止某个 Pod 把节点的内存全用完影响其他人？",
    "集群里的应用想安全地调用云服务 API，不想把密钥写死在代码里怎么做？",
    "某类任务必须固定跑在高配节点上，普通节点不允许调度，怎么实现？",
    "请求量突然翻倍，怎么让集群自动扩出更多实例来扛？",
    "数据库密码不想写进容器镜像，Kubernetes 里怎么安全地存这些配置？",
    "发版出问题了怎么快速切回上一个 Deployment 版本？",
    "Pod 崩了之后集群怎么决定要不要重启它，判断依据是什么？",
    "应用更新的时候怎么保证有旧 Pod 在跑，不会全部同时停掉？",
    "集群里的流量怎么尽量在同一个可用区内转发，减少跨区延迟？",
    "怎么让某个应用在每台节点上都跑一份，新节点加进来也自动部署？",
]


# ── 单条问题执行 ───────────────────────────────────────────────────────────────

def run_single(question: str, dept_id: str = "public", label: str = "",
               quality_gate: bool = True, relevance_filter: bool = True) -> dict:
    from graph2.graph_2 import graph

    inputs = {
        "question": question,
        "original_question": question,
        "documents": [],
        "parent_contexts": [],
        "generation": "",
        "transform_count": 0,
        "not_useful_count": 0,
        "generation_count": 0,
        "hallucination_result": "",
        "answer_result": "",
    }

    # thread_id 前缀带 label，spans 表没有专门的 run 标签字段，
    # 之后用 WHERE thread_id LIKE '{label}%' 把这次批量评估的 span 从
    # 交互式问答的历史 trace 里筛出来。
    thread_id = f"{label}::{uuid.uuid4()}"
    ctx = TraceContext(thread_id=thread_id)
    handler = TraceHandler(ctx)
    config = {
        "callbacks": [handler],
        "configurable": {
            "thread_id": thread_id,
            "trace_ctx": ctx,
            "dept_id": dept_id,
            "enable_hallucination_check": quality_gate,
            "enable_answer_check": quality_gate,
            "enable_relevance_filter": relevance_filter,
        },
    }

    generation = ""
    # assemble_context 的输出（父块），不是 grade_documents 的输出（小块）——
    # generate_node2.py:format_docs 真正喂给 LLM 的是 parent_contexts，用小块去
    # 判 faithfulness 判的是 LLM 根本没直接看到的文本，判断依据会错位。
    contexts = []
    route = "vectorstore"
    t0 = time.time()

    for output in graph.stream(inputs, config=config):
        for key, value in output.items():
            if key == "assemble_context" and "parent_contexts" in value:
                contexts = [item["content"] for item in value["parent_contexts"]]
            if key == "web_search":
                route = "web_search"
            if "generation" in value and value["generation"]:
                generation = value["generation"]

    return {
        "question": question,
        "answer": generation,
        "contexts": contexts,
        "route": route,
        "elapsed_s": round(time.time() - t0, 2),
        "trace_id": ctx.trace_id,
    }


# ── LLM-as-Judge 评估（替代 RAGAS，避免 DeepSeek API 兼容问题）─────────────────

_FAITHFULNESS_PROMPT = """你是一个严格的评估员。
参考文档：
{context}

生成答案：
{answer}

问题：生成答案中的每一个实质性陈述，是否都能在参考文档中找到依据？
只回答 yes 或 no，不要解释。"""

_RELEVANCY_PROMPT = """你是一个严格的评估员。
用户问题：{question}
生成答案：{answer}

问题：这个答案是否直接回答了用户的问题？
只回答 yes 或 no，不要解释。"""


def _judge_one(llm, prompt: str) -> float:
    try:
        resp = llm.invoke(prompt).content.strip().lower()
        return 1.0 if resp.startswith("yes") else 0.0
    except Exception:
        return float("nan")


def _is_fallback(answer: str) -> bool:
    """三个 fallback 节点（hallucination_fallback/not_useful_fallback/
    web_search_fallback）的输出都以 ⚠️ 开头，且都不是"直接回答问题"这种形态——
    是系统主动声明"答不了，给你原始资料/建议换问法"。拿 faithfulness/
    answer_relevancy 这两个衡量"生成答案质量"的指标去判它，答案本身就不是
    直接回答，answer_relevancy 几乎必然判 0，会把这类输出和"生成质量差"
    混为一谈，掩盖了两者本质不同：前者是系统主动声明不确定（设计意图），
    后者才是真的生成质量问题。"""
    return answer.startswith("⚠")


def run_llm_judge(rag_results: list, label: str, timestamp: str) -> str:
    from llm_models.all_llm import llm

    # 只评估走了本地知识库且检索到内容的条目
    vectorstore_results = [r for r in rag_results if r["route"] == "vectorstore" and r["contexts"]]
    # fallback 输出单独统计，不进 faithfulness/answer_relevancy 的均值计算——
    # 这两个指标衡量的是"生成的答案质量"，fallback 根本不是一次生成尝试的结果，
    # 是系统在多次重试后主动放弃生成、转而输出原始资料或提示语。混进均值会让
    # "系统正确地拒绝瞎答"和"系统生成质量差"变成同一个数字，无法区分。
    fallback = [r for r in vectorstore_results if _is_fallback(r["answer"])]
    valid = [r for r in vectorstore_results if not _is_fallback(r["answer"])]
    if not valid:
        print("[Judge] 没有可评估的 vectorstore 结果")
        return None

    print(f"\n[Judge] 开始 LLM-as-Judge 评估，共 {len(valid)} 条...")

    per_question = []
    for i, r in enumerate(valid, 1):
        context_text = "\n".join(r["contexts"][:3])  # 最多取前3个 chunk

        f_score = _judge_one(llm, _FAITHFULNESS_PROMPT.format(
            context=context_text, answer=r["answer"]
        ))
        ar_score = _judge_one(llm, _RELEVANCY_PROMPT.format(
            question=r["question"], answer=r["answer"]
        ))
        per_question.append({
            "question": r["question"],
            "faithfulness": f_score,
            "answer_relevancy": ar_score,
        })
        print(f"  [{i:02d}/{len(valid)}] faithfulness={f_score}  answer_relevancy={ar_score}")

    valid_f  = [x["faithfulness"]     for x in per_question if x["faithfulness"] == x["faithfulness"]]
    valid_ar = [x["answer_relevancy"] for x in per_question if x["answer_relevancy"] == x["answer_relevancy"]]
    summary = {
        "faithfulness":     round(sum(valid_f)  / len(valid_f),  4) if valid_f  else float("nan"),
        "answer_relevancy": round(sum(valid_ar) / len(valid_ar), 4) if valid_ar else float("nan"),
    }

    output = {
        "timestamp": timestamp,
        "label": label,
        "eval_method": "llm-as-judge (DeepSeek)",
        "n_total": len(rag_results),
        "n_evaluated": len(valid),
        "n_web_search":  sum(1 for r in rag_results if r["route"] == "web_search"),
        "n_error":       sum(1 for r in rag_results if r["route"] == "error"),
        "n_no_context":  sum(1 for r in rag_results if r["route"] == "vectorstore" and not r["contexts"]),
        # fallback：知识库路径检索到了 context，但最终没有正常生成答案（三个
        # fallback 节点之一兜底），不计入 faithfulness/answer_relevancy 均值，
        # 单独报告——这个比例本身是系统"多少比例的问题拒绝直接作答"的信号，
        # 和"生成答案质量"是两件不同的事，混在一起看不清任何一个。
        "n_fallback":    len(fallback),
        "fallback_rate": round(len(fallback) / len(vectorstore_results), 4) if vectorstore_results else float("nan"),
        "fallback_questions": [r["question"] for r in fallback],
        "summary": summary,
        "per_question": per_question,
    }

    os.makedirs("eval_results", exist_ok=True)
    scores_path = f"eval_results/{timestamp}_{label}_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== LLM-as-Judge 评估结果 ({label}) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  fallback_rate: {output['fallback_rate']}  ({len(fallback)}/{len(vectorstore_results)} 条知识库路径最终未正常生成)")
    print(f"\n详细结果已保存: {scores_path}")
    return scores_path


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label",  default="baseline_deepseek_grader",
                        help="运行标签，区分消融实验的不同配置")
    parser.add_argument("--limit",  type=int, default=0,
                        help="只跑前 N 条（0=全部），用于快速验证脚本")
    parser.add_argument("--no-quality-gate", action="store_true",
                        help="关闭幻觉检测+答案质量检测（消融用；默认两个门禁都开）")
    parser.add_argument("--no-relevance-filter", action="store_true",
                        help="关闭 grade_documents 的相关性阈值过滤（消融用；不重排、"
                             "不打分，直接取RRF融合池前4，候选池宽度不变，只去掉过滤这一步）")
    args = parser.parse_args()

    quality_gate = not args.no_quality_gate
    relevance_filter = not args.no_relevance_filter
    questions = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("eval_results", exist_ok=True)

    print(f"标签: {args.label}  共 {len(questions)} 条问题  "
          f"quality_gate={quality_gate}  relevance_filter={relevance_filter}\n")

    raw_results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i:02d}/{len(questions)}] {q[:45]}...")
        try:
            result = run_single(q, label=args.label, quality_gate=quality_gate,
                                 relevance_filter=relevance_filter)
        except Exception as e:
            print(f"  [ERROR] {e}")
            result = {"question": q, "answer": "", "contexts": [], "route": "error", "elapsed_s": 0}
        raw_results.append(result)
        print(f"  route={result['route']}  contexts={len(result['contexts'])}  {result['elapsed_s']}s")

    # 保存原始结果
    raw_path = f"eval_results/{timestamp}_{args.label}_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已保存: {raw_path}")

    # 统计摘要
    total_elapsed = sum(r["elapsed_s"] for r in raw_results)
    n_web = sum(1 for r in raw_results if r["route"] == "web_search")
    n_no_ctx = sum(1 for r in raw_results if r["route"] == "vectorstore" and not r["contexts"])
    print(f"\n=== 原始统计 ===")
    print(f"  总耗时:           {total_elapsed:.1f}s")
    print(f"  走本地知识库:     {len(questions) - n_web} 条")
    print(f"  走 web_search:   {n_web} 条（路由认为不属于 K8s 领域）")
    print(f"  检索后无 context: {n_no_ctx} 条（所有 chunk 被 grade_documents 过滤）")

    # LLM-as-Judge 评估
    run_llm_judge(raw_results, args.label, timestamp)


if __name__ == "__main__":
    main()
