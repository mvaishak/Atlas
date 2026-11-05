> Historical source proposal. The approved [plan](PLAN.md), [evaluation contract](EVALUATION.md), and [roadmap](ROADMAP.md) take precedence. This text is retained for provenance; it is not an implementation checklist.

# Project Proposal: AtlasRank

An intelligent product discovery and ranking platform, built with production-grade ML systems engineering

I want to propose something more ambitious than a typical portfolio project, but also something that is realistic to complete and genuinely useful for the roles you're pursuing.

Build an end-to-end ML platform that learns to retrieve, rank, and recommend products, serves predictions through real APIs, and continuously evaluates model quality, latency, and reliability.

Not a chatbot. Not another RAG application. And not simply training a recommendation model in a Jupyter notebook.

The aim is to demonstrate that you can own the entire lifecycle of a production-oriented ML system, from data engineering and model development to inference, experimentation, and monitoring.

THE FLAGSHIP PROJECT

# AtlasRank: Intelligent Discovery & ML Serving Platform

Search relevance • Neural retrieval • Learning to rank • Online inference • ML observability

Modeling

Train intelligent retrieval and ranking models

Systems

Serve models and optimize latency

Reliability

Evaluate, monitor, and safely deploy

Target roles

Machine Learning Engineer · Applied ML Engineer · Search & Ranking Engineer · ML Systems Engineer · ML Platform Engineer

## 1. What the project actually does

Imagine you're building the intelligence behind the search and discovery experience of an online marketplace.

A user searches:

> "Comfortable wireless headphones for long flights under $150"

The system needs to understand the query, find relevant products, distinguish exact matches from substitutes, rank the results, and return them quickly.

A second part of the platform can eventually incorporate user behavior and preferences for personalization.

This creates several engineering problems:

- How do we train embeddings that capture product-query relevance?
- Can a learned ranker outperform lexical and pretrained embedding baselines?
- How do we retrieve candidates efficiently from a large product catalog?
- How do we minimize inference latency without sacrificing relevance?
- How do we detect when changes to a model degrade quality?
- Can we roll back a bad model deployment automatically?

Those questions map directly to the ML and systems work you've been looking at in Glean, Pinterest, Instacart, and ML infrastructure postings.

## 2. The complete system architecture

Proposed architecture

Data ingestion & validation

Product metadata · Relevance labels · Interaction events (optional)Parquet · Polars/PySpark · Data quality checks

Offline training

Embedding models · Learning-to-rank · MLflow

Retrieval indexes

BM25 · FAISS/Qdrant · Product features

Online retrieval API

Query processing · Hybrid candidate generation · Feature lookup

Ranking service

Learned reranking · Candidate filtering · Top-K results

User-facing search demo

Query · Ranked products · Model comparison · Latency

ML evaluation & observability

Offline relevance · p95 latency · Drift · Model versions · Regression gates

The training system publishes versioned models and indexes. The online system serves them independently, while observability collects metrics across both paths.

This is the architecture I'd expect you to be able to draw on a whiteboard, implement, benchmark, and explain in an ML system design interview.

## 3. The dataset: use a real industry benchmark

For the core project, I'd use the [Amazon Shopping Queries Dataset (ESCI)](https://github.com/amazon-science/esci-data?utm_source=chatgpt.com).

# 130K+

Unique queries in the larger dataset

# 2.6M+

Query-product judgments

# 4

Relevance classes

These are published dataset characteristics, not project results. The dataset provides Exact, Substitute, Complement, and Irrelevant labels along with product metadata.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



Why this is a good choice: you can evaluate relevance against real human annotations rather than inventing success criteria for a demo.

But there's an important distinction: ESCI is a query-product relevance dataset, not a user-behavior recommendation dataset.

I don't want us to pretend we're demonstrating personalized recommendation performance when the data doesn't support that claim.

For the eventual personalization extension, we can use the [Synerise RecSys Challenge 2025 dataset](https://github.com/Synerise/recsys2025?utm_source=chatgpt.com), which contains real, anonymized shopping interaction events such as purchases, cart changes, and visits. It has a noncommercial license, so we'd check its usage conditions.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



These would be two explicitly separated evaluation tracks, not artificially joined datasets.

## 4. The ML research and engineering work

This is the most important component. I want you to do actual modeling rather than just calling an embedding API.

Component A — Hybrid retrieval

Implement BM25 and dense embedding retrieval, then combine their candidates using reciprocal rank fusion or learned weighting.

Train or fine-tune a dual-encoder model in PyTorch using query-product relevance supervision.

Compare pretrained embeddings against your fine-tuned representations.

Component B — Learning to rank

Implement an XGBoost LambdaMART-style ranker using retrieval scores, lexical signals, and product attributes.

Then compare it against a neural reranker, potentially using a cross-encoder.

Study the tradeoff between ranking quality and inference cost. This is where your modeling ability becomes obvious.

Component C — Inference optimization

Profile query encoding, ANN lookup, feature construction, and reranking separately.

Experiment with batching, caching, quantization, and candidate-set size.

Instead of reporting just model accuracy, show the complete quality-versus-latency tradeoff.

Component D — Production evaluation

Build a reproducible test suite that compares models on relevance, performance, data quality, and robustness.

Include model versioning, regression checks, failure analysis, and a rollback demonstration.

This is where you connect AtlasRank with your MLWorks experience.

### The experiments I'd want to see

A polished repository should contain an actual experiment table comparing models and retrieval strategies.

| Experiment                   | Purpose                              |
| ---------------------------- | ------------------------------------ |
| BM25 only                    | Establish lexical retrieval baseline |
| Pretrained bi-encoder        | Measure zero-shot dense retrieval    |
| Fine-tuned bi-encoder        | Demonstrate representation learning  |
| BM25 + dense hybrid          | Test complementary retrieval signals |
| Hybrid + LambdaMART          | Demonstrate learning-to-rank         |
| Hybrid + neural reranker     | Test accuracy/latency tradeoffs      |
| Quantized or batched serving | Measure performance optimization     |

Crucially, these are experiments to run, not results we already know.

A fine-tuned model might lose to a strong baseline. That's still an interesting result if the experimental design is rigorous.

## 5. Define the evaluation contract before training

A convincing project needs measurable success criteria.

| Dimension               | Metrics                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| Retrieval quality       | Recall\@20, Recall\@100<br>Compare lexical, dense and hybrid retrieval                     |
| Ranking quality         | NDCG\@10, MRR\@10<br>Measure whether reranking improves relevance                          |
| Serving performance     | p50/p95/p99 latency<br>Benchmark identical workloads and hardware                          |
| Throughput              | Requests per second<br>Measure throughput under concurrency                                |
| Model efficiency        | Memory, index size, inference time<br>Quantify deployment tradeoffs                        |
| Robustness              | Performance by query category and language<br>Identify distribution-specific failure modes |
| Operational reliability | Error rate, recovery time, rollback test<br>Verify system behavior when components fail    |

I would establish several engineering acceptance criteria ahead of time, such as complete reproducibility, a functioning retrieval-to-ranking API, baseline comparisons, and a load-test report.

I would not promise arbitrary relevance improvements or a production-level p95 latency before benchmarking the hardware. Those need to be measured.

We also need to avoid common evaluation mistakes: leakage between train and test queries, tuning on the test set, evaluating rerankers only on labeled candidates while claiming whole-catalog retrieval quality, and treating simulated load as real user traffic.

For ESCI, I'd report both candidate-set reranking performance and full-index retrieval performance, clearly identifying the difference.

## 6. Technology stack

I wouldn't use 25 different frameworks just to populate your resume. Each technology should have a clear reason to exist.

| Layer            | Proposed technologies                                 | What it demonstrates                    |
| ---------------- | ----------------------------------------------------- | --------------------------------------- |
| Data engineering | Python, Polars, Parquet; PySpark for larger pipelines | Scalable preprocessing and validation   |
| Model training   | PyTorch, Sentence Transformers                        | Representation learning                 |
| Ranking          | XGBoost / LambdaMART                                  | Learning-to-rank                        |
| Search           | BM25, FAISS                                           | Lexical and vector retrieval            |
| Model tracking   | MLflow                                                | Reproducibility and versioning          |
| Serving          | FastAPI, Docker                                       | Production-oriented API engineering     |
| Caching          | Redis, when justified                                 | Online latency optimization             |
| Load testing     | Locust or k6                                          | Throughput and latency characterization |
| Observability    | Prometheus, Grafana                                   | Infrastructure and service metrics      |
| Testing / CI     | Pytest, GitHub Actions                                | Software reliability                    |

I'd keep the first deployment on a single machine or modest cloud VM, with separate logical services. Kubernetes is a worthwhile optional extension, not a prerequisite.

This is also a good fit for your M4 MacBook Air for initial development. Fine-tuning larger encoders, neural reranking, or large-scale benchmarking may benefit from a rented or university GPU. We'd reserve GPU use for experiments that need it rather than continuously running cloud infrastructure.

## 7. The feature that would make this project stand out

Most ranking projects stop here:

Train model → calculate NDCG → put results in a dashboard.

I want yours to demonstrate what happens when an existing production model needs to be replaced safely.

Here's the scenario:

Model deployment experiment

1. Deploy Model A behind the ranking API and establish baseline quality and service metrics.
2. Train Model B using a changed architecture, dataset, or ranking objective.
3. Run automated offline evaluation and compare against the previously approved model.
4. Send replayed queries to Model B in shadow mode, without affecting the primary response.
5. Introduce a deliberate regression or latency fault to demonstrate detection and rollback.

For the demo, you could show two ranking models running side by side, how their outputs differ, which queries regress, and how serving latency changes.

The system could then block a model release because it fails a quality or performance gate.

This is the kind of engineering story I think would connect exceptionally well to your MLWorks experience.

It is also a great interview story because it brings together modeling, infrastructure, debugging, experiment design, and real-world engineering judgment.


## 9. What your final portfolio should contain

The finished project should have considerably more than a GitHub repository with source code.

A well-engineered repository

Modular training and inference code, configuration files, tests, reproducible experiment instructions, and a clear README.

A benchmark report

Tables and graphs showing quality, latency, throughput, memory, model tradeoffs, and meaningful ablations.

A working technical demo

Search for products, compare rankings from different models, inspect scores and timings, and demonstrate a regression.

A short technical case study

Explain the problem, architecture, model choices, experiments, failures, results, and what you'd change in a real deployment.

A 3–5 minute walkthrough

Show how a query flows through the system, how models are evaluated, and what happens during a failed deployment.

A live cloud deployment would be helpful, but I would not make maintaining an expensive always-on service a requirement. A reliable local Docker demo and a recorded benchmark can be enough to show engineering competence.

## 10. Which jobs would this help you target?

| Role from your research            | What AtlasRank demonstrates                                                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| Glean: ML Engineer, Search Quality | Retrieval, ranking, relevance evaluation, model development                                     |
| Pinterest: MLE / MLE II            | Representation learning, scalable ML, ranking, experimentation                                  |
| Instacart: Ads Response Prediction | Ranking foundations, supervised modeling, feature engineering; not direct ad-auction experience |
| Glean: AI/ML Infrastructure        | Model serving, versioning, reliability, performance testing                                     |
| Brain Co.: ML Engineer, Platform   | Model comparison, evaluation systems, reusable capabilities                                     |
| ML Systems / Inference roles       | Deployment and inference profiling; not necessarily GPU kernel expertise                        |

This won't make you qualified for every role automatically.

For example, a dedicated GPU inference engineer role may expect CUDA, Triton, distributed GPU serving, and kernel-level optimization beyond what the core project demonstrates.

But it would give you direct evidence for the common ground between Applied ML and ML Systems roles.

## 11. How this would appear on your resume

We shouldn't invent numerical accomplishments before building anything, but this is the kind of resume story we want to earn:

AtlasRank | End-to-End Neural Retrieval & Ranking Platform

Illustrative bullets — not yet achieved

- Engineered a multi-stage product search system combining fine-tuned dual-encoder retrieval, hybrid lexical/semantic search, and learning-to-rank models, benchmarking relevance against established baselines.
- Built containerized low-latency ML serving APIs with caching, concurrency testing, and model-level profiling, measuring quality-throughput-latency tradeoffs.
- Developed reproducible ML training and evaluation workflows with versioned experiments, automated regression gates, observability, and model rollback.

Once the experiments are complete, we'd replace the generic wording with actual numbers and keep only the strongest two or three bullets.

## 12. One important design decision

You already have PlayNext, which is a recommender system.

I do not want AtlasRank to become PlayNext 2.0 with a few additional tools.

The differentiation should be clear:

| PlayNext                                   | AtlasRank                                               |
| ------------------------------------------ | ------------------------------------------------------- |
| User-item recommendation                   | Query-product retrieval and ranking                     |
| Collaborative filtering and hybrid scoring | Fine-tuned representation learning and learning-to-rank |
| Offline recommendation metrics             | Separate retrieval, ranking and serving benchmarks      |
| Streamlit recommendation demo              | Independently deployed and tested inference services    |
| Model-centric portfolio project            | Full ML lifecycle and systems engineering project       |

That makes AtlasRank a complementary project rather than a repetition of your previous work.

## 13. What I would deliberately leave out

For version 1, I would not add LLM agents, GraphRAG, Kubernetes clusters, distributed GPU training, Kafka, reinforcement learning, or an elaborate frontend merely to make the architecture look impressive.

Every added component needs a measurable engineering reason.

I'd rather see you build a ranking system with genuinely good baselines, an excellent evaluation setup, strong tests, and defensible performance results than build a massive collection of loosely connected microservices.

The core project's best feature is its depth, not its number of technologies.

## Final recommendation

I think AtlasRank gives you the strongest balance of the things you said you actually enjoy: building intelligent models and engineering the systems behind them.

It builds directly on your professional background while adding missing proof of modern model serving, learned ranking, deployment infrastructure, and rigorous systems evaluation.

My one adjustment before implementation would be to write a strict one-page project specification freezing the dataset, baseline models, evaluation protocol, MVP scope, and resource budget. That prevents scope creep and gives us a clear definition of success.

If you build this well, your strongest positioning becomes:

An ML engineer with real production experience who can independently develop intelligent models, deploy them as reliable systems, and empirically optimize their quality and performance.

That's a considerably stronger story for your target roles than being another candidate with several RAG demos.