import os
import re
import tempfile
from typing import List, Tuple

from langchain_experimental.text_splitter import SemanticChunker

from llm_models.embeddings_model import bge_embedding
from utils.log_utils import log
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

# 小块目标区间：必须显著小于 BGE-small-zh-v1.5 / bge-reranker-base 的 512 token 上限
# （实测 max_position_embeddings 分别是 512 / 514），且 CrossEncoder 把 query 和
# doc 拼成一个序列过模型，要给 query 留空间。
SMALL_CHUNK_MIN = 200
SMALL_CHUNK_MAX = 300

# 父块长度上界：超过则在 assemble_context 节点退化为"命中小块 + 相邻若干小块"，
# 不返回整章。取值依据见 CLAUDE.md 相应记录（真实K8s语料388文件章节长度分布实测：
# 上界2000对应约5.5%章节触发退化）。
PARENT_UPPER_BOUND = 2000

# 统一入库判据下限：短于此不承载检索价值（实测多是被 Unstructured 切碎的代码块
# 片段/命令残段，如"将 kubectl drain"）。
MIN_CHUNK_LEN = 20

FRONTMATTER_RE = re.compile(r'^---\s*\n.*?\n---\s*\n', re.DOTALL)

# 中文翻译版 K8s 文档普遍保留英文原文、用 HTML 注释隐藏（渲染后不可见），和 Hugo
# 的 {{< comment >}} 短代码是同一性质问题的两种语法，需要在解析前一并剥掉。个别
# _index.md（纯目录导航页）整篇除 frontmatter 外只剩这种注释，剥完后交给下面的
# 空内容判断统一跳过，不需要单独处理这种退化情况。
HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)

# c类短代码：包裹型，删标记留内容（note/caution/warning 这类提示框，tab/tabs/table
# 是UI容器，example 包裹的是内嵌链接文字）
WRAPPING_KEEP_TAGS = [
    'alert', 'pageinfo', 'tab', 'tabs',
    'caution', 'details', 'example', 'highlight',
    'mermaid', 'note', 'table', 'warning',
]

# heading 短代码译法：语料实证结果（同文件内英文原文HTML注释 + 中文正式译文对照
# 找到的），不是猜测措辞——这几个词会拼进 chunk 参与 BM25 分词和向量计算，必须
# 和语料其余部分用词一致。seealso 语料无实证，用中文技术文档通用译法兜底。
HEADING_LABELS = {
    'prerequisites': '先决条件',
    'objectives': '目标',
    'whatsnext': '接下来',
    'cleanup': '清理',
    'seealso': '参见',
}


def _strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub('', text, count=1)


def _clean_shortcodes(text: str) -> str:
    """
    Hugo 短代码按语义分四类处理，不能按"自闭合/包裹"这种形态分类——heading 短代码
    形态上是自闭合，但参数携带的是整个标题的全部文字，按形态分类会把它和纯装饰性
    的自闭合标记（如 feature-state）混为一谈，误删导致标题消失、挂在其下的子级
    元素全部失去父级锚点（详见 CLAUDE.md 对应记录）。
    """
    # comment 短代码是 Hugo 构建时注释，渲染后不输出，整块删除（含内容）
    for _ in range(3):
        before = text
        pattern = re.compile(
            r'\{\{[<%]\s*comment\b[^}]*[>%]\}\}.*?\{\{[<%]\s*/\s*comment\s*[>%]\}\}',
            re.DOTALL,
        )
        text = pattern.sub('', text)
        if text == before:
            break

    # c类：包裹型，删标记留内容，循环处理嵌套（如 note 包 tab）
    for _ in range(5):
        before = text
        for tag in WRAPPING_KEEP_TAGS:
            pattern = re.compile(
                r'\{\{[<%]\s*' + tag + r'\b[^}]*[>%]\}\}(.*?)\{\{[<%]\s*/\s*' + tag + r'\s*[>%]\}\}',
                re.DOTALL,
            )
            text = pattern.sub(lambda m: m.group(1), text)
        if text == before:
            break

    # b类：自闭合但参数携带真实文字，取值替换，不能走下面的兜底删除
    text = re.sub(
        r'\{\{%\s*heading\s+"([a-zA-Z_-]+)"\s*%\}\}',
        lambda m: HEADING_LABELS.get(m.group(1), m.group(1)),
        text,
    )

    def _glossary_tooltip_sub(m):
        attrs = m.group(1)
        tm = re.search(r'text="([^"]*)"', attrs)
        if tm:
            return tm.group(1)
        tm2 = re.search(r'term_id="([^"]*)"', attrs)
        return tm2.group(1) if tm2 else ''
    text = re.sub(r'\{\{<\s*glossary_tooltip\s+([^}]*?)\s*>\}\}', _glossary_tooltip_sub, text)

    def _link_sub(m):
        tm = re.search(r'text="([^"]*)"', m.group(1))
        return tm.group(1) if tm else ''
    text = re.sub(r'\{\{<\s*link\s+([^}]*?)\s*>\}\}', _link_sub, text)

    def _figure_sub(m):
        tm = re.search(r'alt="([^"]*)"', m.group(1))
        return tm.group(1) if tm else ''
    text = re.sub(r'\{\{<\s*figure\s+([^}]*?)\s*>\}\}', _figure_sub, text)

    # a类：其余全部自闭合标记（feature-state/skew/param/version-check/api-reference/
    # code_sample/codenew/code/include/legacy-repos-deprecation/toc/thirdparty-content/
    # glossary_definition），整段删除——内容要么是构建时动态注入的（源文件里没有字面
    # 值可提取），要么指向语料范围外的文件，没有可恢复的自然语言内容
    text = re.sub(r'\{\{[<%][^}]*[>%]\}\}', '', text)

    return text


def _preprocess_markdown_text(raw_text: str) -> str:
    text = _strip_frontmatter(raw_text)
    text = HTML_COMMENT_RE.sub('', text)
    text = _clean_shortcodes(text)
    return text


class MarkdownParser:
    """
    负责markdown文件的解析、章节合并（父块候选）、小块切分。

    small-to-big 架构下的职责变化：
      merge_title_content() 产出的每个 Document 本身就是一个章节，天然是父块候选，
      不需要额外设计"怎么分父块"。
      本类新增的工作是：对每个章节，判断是否需要切成多个小块（_split_section），
      以及生成小块与父块之间的关联信息（parent_seq / chunk_offset），供上层
      （write_milvus.py / lifecycle.py）在拿到 doc_id 后生成真正的 parent_id。

    入库判据（parse_markdown_to_documents）：不再按 Unstructured 给的元素类型
    （category=='content'）筛选——那是结构角色标签，不是内容价值标签，会同时
    误伤孤立正文段落、放过没有实质内容的裸标题。改为统一按长度判据处理：裸
    Title（从未合并过任何内容）不入库，其余一律按 MIN_CHUNK_LEN / SMALL_CHUNK_MAX
    / PARENT_UPPER_BOUND 三档处理，不区分内容来自哪种结构角色。
    """

    def __init__(self):
        self.text_splitter = SemanticChunker(
            bge_embedding, breakpoint_threshold_type="percentile"
        )

    def parse_markdown_to_documents(self, md_file: str, encoding='utf-8') -> Tuple[List[Document], List[dict]]:
        """
        返回 (chunks, parent_records)：
          chunks: 小块 Document 列表，metadata 里含：
            - parent_seq: 该小块所属内容单元在文档内的序号（int），若该单元"切不动"
              （长度 <= SMALL_CHUNK_MAX，只产出1个小块）则为 None，表示不建父块
            - chunk_offset: 该小块在其所属内容单元内的顺序位置（int，从0开始），
              parent_seq 为 None 时同样为 None
            - section_seq: 该小块所属内容单元在文档内的原始顺序号（int，从0开始），
              **不管这个单元有没有被切分、有没有父块都会赋值**——D7 排序要求
              "同一文档内按原序排列"对所有 context 项统一适用。
          parent_records: 父块候选记录列表（dict），每条对应一个"被真正切分过"
            （产出 >=2 个小块）的内容单元，含 seq / title / content / char_count。
        """
        documents = self.parse_markdown(md_file)
        if not documents:
            return [], []
        log.info(f'文件解析后的docs长度: {len(documents)}')

        merged_documents = self.merge_title_content(documents)
        log.info(f'文件合并后的长度: {len(merged_documents)}')

        chunks: List[Document] = []
        parent_records: List[dict] = []

        # 裸 Title（从未合并过任何内容，category 仍是 'Title'）没有检索价值，排除；
        # 其余全部（含合并后的章节、孤立正文段落、原本会被静默丢弃的未归类元素）
        # 一律进入下面的长度判据，不再按结构角色区分
        content_units = [d for d in merged_documents if d.metadata.get('category') != 'Title']

        for section_seq, unit in enumerate(content_units):
            if len(unit.page_content) < MIN_CHUNK_LEN:
                continue

            # _body/title_path 是 merge_title_content 阶段的内部记账字段，Milvus
            # schema 没有对应字段，切块前清掉，避免作为 metadata 写入时报错
            unit.metadata.pop('_body', None)
            unit.metadata.pop('title_path', None)

            small_docs = self._split_section(unit)

            if len(small_docs) <= 1:
                # 切不动：不建父块，parent_id 留空，但 section_seq 照记
                for d in small_docs:
                    d.metadata['parent_seq'] = None
                    d.metadata['chunk_offset'] = None
                    d.metadata['section_seq'] = section_seq
                chunks.extend(small_docs)
            else:
                for offset, d in enumerate(small_docs):
                    d.metadata['parent_seq'] = section_seq
                    d.metadata['chunk_offset'] = offset
                    d.metadata['section_seq'] = section_seq
                chunks.extend(small_docs)
                parent_records.append({
                    'seq': section_seq,
                    'title': unit.metadata.get('title'),
                    'content': unit.page_content,
                    'char_count': len(unit.page_content),
                })

        log.info(f'切分后小块数: {len(chunks)}，产生父块数: {len(parent_records)}')
        return chunks, parent_records

    def _split_section(self, section: Document) -> List[Document]:
        """
        把一个内容单元切成若干 200~300 字的小块。

        算法（工程实现选择，不是逐字规定）：
          1. 单元本身 <= SMALL_CHUNK_MAX：不切，整体当一个小块。
          2. 否则先用 SemanticChunker 切出语义候选段。
          3. 贪心收敛到目标区间：
             - 过小的段（< SMALL_CHUNK_MIN）与后一段合并，直到达到区间或用完。
             - 过大的段（> SMALL_CHUNK_MAX）先尝试用 SemanticChunker 递归细分
               （最多2层，避免死循环/过度递归开销）；如果细分后单段仍然超限
               （说明这段文本内部语义高度连贯、没有更细的边界可切），退化为
               按 SMALL_CHUNK_MAX 定长硬切——这是唯一偏离"不硬切"字面表述的
               地方，因为此时已经没有语义边界信息可用。
        """
        text = section.page_content
        if len(text) <= SMALL_CHUNK_MAX:
            return [Document(page_content=text, metadata=dict(section.metadata))]

        pieces = self._semantic_pieces(section, depth=0)
        merged = self._greedy_merge(pieces, section.metadata)
        return merged

    def _semantic_pieces(self, doc: Document, depth: int) -> List[str]:
        """用 SemanticChunker 切一次，返回纯文本段列表。"""
        try:
            split_docs = self.text_splitter.split_documents([doc])
            pieces = [d.page_content for d in split_docs if d.page_content.strip()]
        except Exception as e:
            log.warning(f'SemanticChunker 切分失败，退化为整段: {e}')
            pieces = [doc.page_content]
        if not pieces:
            pieces = [doc.page_content]
        return pieces

    def _greedy_merge(self, pieces: List[str], base_metadata: dict) -> List[Document]:
        """贪心合并/细分语义段，收敛到 [SMALL_CHUNK_MIN, SMALL_CHUNK_MAX] 区间。"""
        # 第一步：对过大的段做一次递归细分（最多2层）
        expanded: List[str] = []
        for p in pieces:
            if len(p) > SMALL_CHUNK_MAX:
                expanded.extend(self._shrink_piece(p, depth=0))
            else:
                expanded.append(p)

        # 第二步：贪心合并过小的段
        result: List[str] = []
        buffer = ""
        for p in expanded:
            candidate = buffer + p if buffer else p
            if len(candidate) <= SMALL_CHUNK_MAX:
                buffer = candidate
                if len(buffer) >= SMALL_CHUNK_MIN:
                    result.append(buffer)
                    buffer = ""
            else:
                # 加入 p 会超限：先把 buffer 单独存（哪怕不到 MIN，也好过丢内容）
                if buffer:
                    result.append(buffer)
                if len(p) <= SMALL_CHUNK_MAX:
                    buffer = p
                else:
                    # 极端情况：单段仍然超限（细分未能收敛），硬切兜底
                    result.extend(self._hard_split(p))
                    buffer = ""
        if buffer:
            result.append(buffer)

        return [Document(page_content=r, metadata=dict(base_metadata)) for r in result]

    def _shrink_piece(self, text: str, depth: int) -> List[str]:
        """对过大的语义段递归细分，最多2层，仍超限则不再递归（留给上层硬切兜底）。"""
        if depth >= 2 or len(text) <= SMALL_CHUNK_MAX:
            return [text]
        sub_pieces = self._semantic_pieces(Document(page_content=text, metadata={}), depth + 1)
        if len(sub_pieces) <= 1:
            # 没切出更细的边界，停止递归
            return [text]
        out = []
        for sp in sub_pieces:
            if len(sp) > SMALL_CHUNK_MAX:
                out.extend(self._shrink_piece(sp, depth + 1))
            else:
                out.append(sp)
        return out

    def _hard_split(self, text: str) -> List[str]:
        """按 SMALL_CHUNK_MAX 定长切割，仅在语义切分无法收敛时作为最后兜底。"""
        return [text[i:i + SMALL_CHUNK_MAX] for i in range(0, len(text), SMALL_CHUNK_MAX)]

    def parse_markdown(self, md_file: str) -> List[Document]:
        """
        解析前先做预处理：剥 frontmatter、剥隐藏的 HTML 注释（中文翻译版文档惯例）、
        清理 Hugo 短代码。预处理后如果没有实质内容（纯目录导航页 _index.md 常见，
        除 frontmatter 外只剩注释），直接返回空列表，不送入 Unstructured——避免
        整篇只剩注释时底层解析崩溃（Invalid input object: NoneType）。
        """
        with open(md_file, encoding='utf-8') as f:
            raw_text = f.read()
        text = _preprocess_markdown_text(raw_text)
        if not text.strip():
            log.info(f'{md_file} 预处理后无实质内容，跳过')
            return []

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.md')
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                f.write(text)
            loader = UnstructuredMarkdownLoader(
                file_path=tmp_path,
                mode='elements',
                strategy='fast'
            )
            docs = [doc for doc in loader.lazy_load()]
        finally:
            os.remove(tmp_path)

        # Unstructured 是拿临时文件路径解析的，source/filename/file_directory 这几个
        # 元数据字段现在指向临时文件，要改回真实文件路径，否则 trace 回溯查不到源文件
        real_filename = os.path.basename(md_file)
        real_directory = os.path.dirname(md_file)
        for doc in docs:
            doc.metadata['source'] = md_file
            doc.metadata['filename'] = real_filename
            if 'file_directory' in doc.metadata:
                doc.metadata['file_directory'] = real_directory

        return docs

    def merge_title_content(self, datas: List[Document]) -> List[Document]:
        """
        三个分支处理章节合并；三个分支都没命中（含"父级引用存在但找不到父级"这种
        K8s 多级嵌套的情况）的元素不再静默丢弃，统一收进结果，交给
        parse_markdown_to_documents 按长度判据决定入不入库——静默丢弃是这个项目
        反复踩过的坑（as_retriever 的 filter 被 hybrid_search 忽略、
        FILENAME_TO_DEPT 整表失效、manage_docs 与 write_milvus 的 doc_id 算法
        不一致），这里不再加第四个。
        """
        merged_data = []
        parent_dict = {}  # 是一个字典，保存所有的父document， key为当前父document的ID
        for document in datas:
            metadata = document.metadata
            if 'languages' in metadata:
                metadata.pop('languages')

            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            hit = False

            if category == 'NarrativeText' and parent_id is None:  # 是否为：内容document
                merged_data.append(document)
                hit = True

            if category == 'Title':
                own_title_text = document.page_content
                if parent_id in parent_dict:
                    # 面包屑只拼标题路径（title_path），不拼父级已经合并进去的正文——
                    # 否则子标题会把父章节的正文一起吸收进自己的 page_content，多级
                    # 嵌套下同一段正文会在父子两处重复出现（详见 CLAUDE.md 对应记录）
                    parent_title_path = parent_dict[parent_id].metadata.get(
                        'title_path', parent_dict[parent_id].page_content
                    )
                    title_path = parent_title_path + ' -> ' + own_title_text
                else:
                    title_path = own_title_text
                document.metadata['title'] = own_title_text
                document.metadata['title_path'] = title_path
                document.metadata['_body'] = ''
                document.page_content = title_path
                parent_dict[element_id] = document
                hit = True

            if category != 'Title' and parent_id:
                if parent_id in parent_dict:  # K8s 多级嵌套时 parent 可能不是 Title，跳过
                    parent_doc = parent_dict[parent_id]
                    body = parent_doc.metadata.get('_body', '')
                    body = (body + ' ' + document.page_content) if body else document.page_content
                    parent_doc.metadata['_body'] = body
                    parent_doc.page_content = parent_doc.metadata['title_path'] + '\n' + body
                    parent_doc.metadata['category'] = 'content'
                    hit = True
                else:
                    merged_data.append(document)
                    hit = True

            if not hit:
                merged_data.append(document)

        # 处理字典
        if parent_dict is not None:
            merged_data.extend(parent_dict.values())

        return merged_data


def assign_parent_ids(chunks: List[Document], parent_records: List[dict], doc_id: str) -> Tuple[List[Document], List[dict]]:
    """
    把 parse_markdown_to_documents() 产出的 parent_seq（章节内序号，doc_id 未知时的占位）
    转换成真正的 parent_id = doc_id + "_" + seq（明文拼接，不用 hash——K8s 文档标题
    大量重复如 Overview/Example，hash 不可读，序号+doc_id 查 trace 时一眼能看出是
    哪篇文档第几章）。

    调用时机：doc_id 在 write_milvus.py / lifecycle.py 里才确定（从文件相对路径算出），
    parser 本身不知道 doc_id，所以分两步：parser 产出 parent_seq 占位 -> 这里绑定 doc_id。
    """
    for chunk in chunks:
        seq = chunk.metadata.pop('parent_seq', None)
        chunk.metadata['parent_id'] = f"{doc_id}_{seq}" if seq is not None else None

    for record in parent_records:
        record['parent_id'] = f"{doc_id}_{record['seq']}"
        record['doc_id'] = doc_id

    return chunks, parent_records


if __name__ == '__main__':
    file_path = r'E:\my_project\RAG_PROJECT\datas\md\tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    chunks, parent_records = parser.parse_markdown_to_documents(file_path)
    for item in chunks:
        print(f"元数据: {item.metadata}")
        print(f"doc的内容: {item.page_content}\n")
        print("------" * 10)
    print(f"父块数: {len(parent_records)}")
