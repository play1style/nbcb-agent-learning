"""RAG 组件：BM25 稀疏检索索引（无嵌入后端时的生产级降级方案）。

BM25（Okapi BM25）是 Elasticsearch、Lucene 等生产搜索引擎的默认排序算法，
在真实 RAG 系统中常与向量检索组成"混合检索（Hybrid Search）"。
本实现仅依赖标准库，采用中文字符二元组（bigram）分词，保证内网环境可运行。
"""
import math
import re
from collections import Counter


def tokenize(text):
    """中文字符二元组 + 英文数字单词的混合分词。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    chars = [c for c in text if "一" <= c <= "鿿"]
    tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    tokens.extend(chars)  # 单字兜底，提升短查询召回
    return tokens


class BM25Index:
    """Okapi BM25 索引。k1=1.5、b=0.75 为业界常用参数。"""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.doc_tf = []       # 每个分片的词频 Counter
        self.doc_len = []      # 每个分片的词数
        self.avg_len = 0.0
        self.df = Counter()    # 每个词的文档频率
        self.n_docs = 0

    def build(self, chunks):
        """对分片列表建立索引。"""
        self.chunks = list(chunks)
        self.doc_tf = []
        self.doc_len = []
        self.df = Counter()
        for chunk in self.chunks:
            # 标题与章节词权重高于正文（重复计入词频实现加权）
            weighted = (chunk.doc_title + " ") * 3 + (chunk.section + " ") * 2 + chunk.text
            tf = Counter(tokenize(weighted))
            self.doc_tf.append(tf)
            self.doc_len.append(sum(tf.values()))
            for term in tf:
                self.df[term] += 1
        self.n_docs = len(self.chunks)
        self.avg_len = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        return self

    def search(self, query, top_k=3):
        """返回 [(chunk, score), ...]，按 BM25 得分降序，仅保留正得分。"""
        scores = []
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                norm = 1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1)
                scores.append((i, idf * f * (self.k1 + 1) / (f + self.k1 * norm)))
        total = Counter()
        for i, s in scores:
            total[i] += s
        ranked = sorted(total.items(), key=lambda kv: kv[1], reverse=True)
        return [(self.chunks[i], s) for i, s in ranked[:top_k] if s > 0]
