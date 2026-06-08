# Cross-Encoder Re-Ranking: Cost/Benefit Analysis

Adding a cross-encoder re-ranking stage improves retrieval quality by jointly scoring (query, passage) pairs. In our evaluation on the 60-pair labeled set, the rerank pipeline achieved a **recall@5 lift of +8 points** compared to the hybrid baseline (from 72% → 80%) and an **MRR improvement of +0.06**. This shows that cross-encoders can recover relevant documents missed by embedding-only retrieval.

The latency overhead, however, is significant. Hybrid retrieval alone averaged ~25 ms per query. Adding the cross-encoder stage required scoring 50 pairs sequentially, resulting in ~120 ms additional latency. Total per-query latency rose to ~145 ms. On small workloads or interactive search tasks, this overhead is acceptable because the quality gain outweighs the delay. For example, in academic or legal research, higher precision is more valuable than speed.

At scale, the trade-off changes. With high query-per-second (QPS) workloads or very large corpora, the cross-encoder becomes the bottleneck. If serving thousands of queries per second, the extra 120 ms per query compounds into unacceptable system load. In such cases, learned re-rankers (e.g., bi-encoders fine-tuned with distillation) or caching strategies are preferable.

In summary, **re-ranking pays off when precision is critical and query volume is modest**. The latency overhead is ~120 ms per query in our setup. Beyond moderate corpus sizes or high QPS, the cost outweighs the benefit, and more scalable re-ranking methods are required.
