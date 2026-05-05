# AI Research Intelligence Engine Plan

## Goal

Build a domain-specific AI/ML portfolio project over arXiv ML papers that:

- supports Standard RAG and GraphRAG
- compares both pipelines side by side
- evaluates them with RAGAS and manual analysis
- teaches the core engineering ideas behind ingestion, retrieval, graph modeling, and evaluation

The project should be built in phases so that every phase produces a usable artifact and teaches one major concept well.

***

## Implementation status

| Phase | Name | Status |
| --- | --- | --- |
| Phase 1 | Standard RAG: ingestion + `/query/rag` | Done |
| Phase 2 | GraphRAG: graph build + `/query/graphrag` | Done |
| Phase 3 | Evaluation + A/B comparison (latency/cost + judge metrics) | Done |
| Phase 4 | Multi-agent orchestration with LangGraph | Done |
| Phase 5 | Frontend and developer-facing visualization | Done |
| Phase 6 | Tracing + portfolio packaging | In progress |

### Current interview narrative

Phase 1:

> I built an ingestion pipeline over arXiv ML papers using local embeddings with `nomic-embed-text` via Ollama, so embedding does not require API cost. Chunks are stored in Pinecone for similarity search and in a local ChunkStore keyed by `paper_id` and `chunk_index`, which becomes important when graph edges need to point back to evidence.

Phase 2:

> I extended the baseline with a Neo4j knowledge graph. Gemini extracts typed triples from each chunk in JSON mode with a constrained schema: five node types, high-confidence relationships, and evidence-linked edges. At query time, GraphRAG runs seed retrieval, graph expansion, evidence chunk lookup, filtered Pinecone retrieval, and graph-aware synthesis.

Phase 3:

> I evaluated Standard RAG and GraphRAG with RAGAS across faithfulness, answer relevancy, context precision, and context recall on a curated benchmark with factual, entity-centric, and relational questions. GraphRAG should win on relational questions, while Standard RAG should remain faster and cheaper for simple factual questions. Showing that tradeoff honestly makes the project credible.

Phase 4:

> I added a LangGraph orchestration layer with specialized agents. A Query Classifier routes simple questions directly to vector retrieval, skipping Neo4j. Complex relational questions trigger planning and parallel retrieval. An Evidence Merger deduplicates sources, flags `CONTRADICTS` edges as contested claims, and passes a structured evidence packet to the Synthesizer. Every response includes an `agent_trace` with route decisions, tools used, timing, and cost.

Phase 5:

> I built a minimal React UI to demo and debug the system: side-by-side answers (RAG vs GraphRAG vs Agent), an agent trace timeline, a returned subgraph visualization for GraphRAG, and an evaluation dashboard to compare latency/cost and judge metrics.

Phase 6:

> I added LangSmith observability end-to-end with request trace IDs and per-step spans (retrieval, graph expansion, merge, synthesis, and judge scoring). That makes it easy to explain system behavior during a demo and to debug regressions.

***

## Big Picture

### Current system

```text
                         User Question
                               |
                               v
                     +--------------------+
                     | FastAPI Endpoints  |
                     +--------------------+
                         |            |
                         |            |
                         v            v
                  /query/rag    /query/graphrag
                         |            |
                         |            |
                         v            v
                  Vector only     Vector + Graph
                   retrieval         retrieval
                         \            /
                          \          /
                           v        v
                        Answer synthesis
                               |
                               v
                          Final response


                    Offline ingestion pipeline

     arXiv fetch -> PDF parse -> chunk -> embed -> Pinecone
                               \
                                -> entity extraction -> Neo4j
```

### Target multi-agent system

```text
                         User Question
                               |
                               v
                      POST /query/agent
                               |
                               v
                     LangGraph agent graph
                  /            |              \
                 v             v               v
          QueryClassifier   Planner       Routing policy
                 |             |               |
                 |             v               |
                 |       decomposed tasks      |
                 |             |               |
                 +-------------+---------------+
                               |
                               v
              +----------------+----------------+
              |                                 |
              v                                 v
        VectorRetriever                  GraphRetriever
              |                                 |
              +----------------+----------------+
                               |
                               v
                        EvidenceMerger
                               |
                               v
                         Synthesizer
                               |
                               v
             structured answer + citations + agent_trace[]


                    Direct endpoints remain available

           POST /query/rag       -> Standard RAG only
           POST /query/graphrag  -> 5-step GraphRAG
           POST /evaluate        -> RAGAS A/B report
```

### What each subsystem teaches

- **Ingestion** teaches data pipelines, document cleaning, metadata handling
- **Standard RAG** teaches embeddings, retrieval, prompt grounding, citations
- **GraphRAG** teaches knowledge graphs, schema design, Cypher, multi-hop reasoning
- **Evaluation** teaches measurement, tradeoffs, faithfulness, cost awareness
- **Multi-agent orchestration** teaches routing, decomposition, parallel retrieval, traceability, and stateful workflows

***

## Core guiding principles

1. Build the simplest working version first
2. Keep the domain narrow: arXiv ML papers only
3. Prove value with comparison, not just architecture
4. Delay orchestration until the pipelines are understood
5. Track cost, latency, and answer quality from the beginning
6. Prefer clean graph schema over many noisy relations

***

## Phase roadmap

## Phase 0 — Foundations and design

### Objective

Define the scope, dataset, evaluation questions, and system boundaries before coding.

### Why this phase matters

Without a clear scope, GraphRAG projects become large, vague, and hard to evaluate. This phase keeps the project focused and interview-ready.

### Build in this phase

1. Lock the domain to arXiv ML papers
2. Pick 2 to 4 subtopics:
   - hallucination mitigation
   - reasoning in LLMs
   - chain-of-thought
   - RLHF
3. Define 10 to 15 benchmark questions
4. Define success criteria for the project
5. Decide what metadata to keep for each paper and chunk

### Concepts to understand

- **Standard RAG**: retrieve similar chunks and augment the prompt
- **GraphRAG**: retrieve chunks plus explicit entity relationships
- **Multi-hop question**: a question requiring connections across entities or documents
- **Grounded answer**: an answer supported by retrieved evidence

### Deliverables

- finalized topic scope
- benchmark question set draft
- project folder layout
- paper metadata schema draft

### Suggested outputs

- `data/benchmark/questions.json`
- a short design note describing the evaluation question categories

### Resources

- Pinecone RAG fundamentals
- LangChain recursive splitter docs
- GraphRAG paper
- Neo4j Cypher manual

***

## Phase 1 — Dataset ingestion and Standard RAG baseline

### Objective

Build a working retrieval system over arXiv papers. This is the baseline and safety net.

### Why this phase matters

Even if GraphRAG is delayed, this phase gives a working project. It also creates the baseline needed for later comparison.

### System diagram

```text
arXiv API -> PDFs -> parser -> cleaned text -> chunker -> embeddings -> Pinecone
                                                             |
                                                             v
                                                      query embedding
                                                             |
                                                             v
                                                       top-k chunks
                                                             |
                                                             v
                                                            LLM
                                                             |
                                                             v
                                                   answer with citations
```

### Files to build

1. `backend/config.py`
2. `backend/ingestion/arxiv_fetcher.py`
3. `backend/ingestion/pdf_parser.py`
4. `backend/ingestion/chunker.py`
5. `backend/ingestion/embedder.py`
6. `backend/vector_store/pinecone_client.py`
7. `backend/vector_store/retriever.py`
8. `backend/pipelines/rag_pipeline.py`
9. `backend/routers/query.py`
10. `backend/main.py`

### Step-by-step build order

#### Step 1. Configuration and environment

Create a settings module that loads:

- OpenAI key
- Anthropic key
- Pinecone key
- index name
- environment flags

Learn:

- environment-driven configuration
- secrets handling
- separation between code and deployment config

#### Step 2. Fetch papers from arXiv

Build a fetcher that:

- searches by category and keyword
- stores paper metadata
- downloads PDFs locally

Start with 20 papers, not 100.

Learn:

- reproducible datasets
- rate limits
- canonical paper IDs

Keep metadata like:

- arxiv\_id
- title
- authors
- summary
- categories
- published date
- pdf\_url
- local file path

#### Step 3. Parse PDFs

Use `unstructured` to extract text.

Learn:

- PDF text extraction is noisy
- headers, footers, references, and broken line wraps reduce retrieval quality
- parsed text should be cached to avoid repeated processing

Store parsed output as structured JSON if possible so re-ingestion is easier.

#### Step 4. Chunk text

Start with:

- 700 to 1000 token equivalent chunks
- 100 to 150 overlap
- section-aware chunking when possible

Learn:

- chunk size affects retrieval quality
- overlap protects continuity between chunks
- research papers need more context than generic documents

Chunk metadata should include:

- paper\_id
- title
- authors
- chunk\_index
- section
- source path

#### Step 5. Generate embeddings

Use `text-embedding-3-small`.

Learn:

- embeddings convert text into vectors
- cosine similarity compares semantic closeness
- embedding cost matters at scale

#### Step 6. Store vectors in Pinecone

Create one Pinecone index and upsert:

- vector id
- embedding values
- metadata
- original chunk text or reference to stored chunk text

Learn:

- index dimensions must match the embedding model
- metadata filtering helps later
- vector store is optimized for similarity search, not relational queries

#### Step 7. Retrieve relevant chunks

At query time:

1. embed the user query
2. retrieve top-k similar chunks
3. build a clean evidence list

Learn:

- retrieval quality determines answer quality
- top-k that is too small misses context
- top-k that is too large adds noise

#### Step 8. Build the RAG answer pipeline

The RAG pipeline should:

1. retrieve chunks
2. format a grounded prompt
3. ask the LLM to answer only from provided evidence
4. return answer plus cited sources

Learn:

- prompt structure affects groundedness
- citations increase trust
- the LLM should be allowed to say "insufficient evidence"

#### Step 9. Expose `/query/rag`

Create a FastAPI endpoint for the baseline system.

Request:

- question
- optional top\_k

Response:

- answer
- citations
- retrieved chunks
- latency
- token usage if available

### Deliverable

A query like:

`What methods reduce hallucinations in LLMs?`

returns a grounded answer supported by chunks from the ingested papers.

### What you should understand before leaving this phase

- how embeddings work conceptually
- why chunking is not trivial
- why retrieval quality is more important than prompt cleverness
- why the baseline must be correct before GraphRAG is added

### Resources

- Pinecone RAG guide
- LangChain text splitter docs
- OpenAI embeddings docs
- FastAPI docs

***

## Phase 2A — Knowledge graph construction

### Objective

Extract entities and relationships from document chunks and store them in Neo4j.

### Why this phase matters

This is the differentiator. It turns unstructured paper text into a queryable relationship graph.

### System diagram

```text
document chunks -> entity extraction prompt -> triples -> normalization -> Neo4j

example:
Paper A -> PROPOSES -> Method X
Author B -> AUTHORED -> Paper A
Author B -> AFFILIATED_WITH -> Institution C
Paper A -> CITES -> Paper D
```

### Keep the first graph schema small

#### Node types

- `Paper`
- `Author`
- `Institution`
- `Concept`
- `Method`

#### Relationship types

- `AUTHORED`
- `AFFILIATED_WITH`
- `STUDIES`
- `PROPOSES`
- `CITES`

Delay noisier relations until later.

### Files to build

1. `backend/graph_store/neo4j_client.py`
2. `backend/graph_store/entity_extractor.py`
3. `backend/graph_store/graph_writer.py`
4. optional normalization helper

### Step-by-step build order

#### Step 1. Connect to Neo4j

Use the official Python driver.

Learn:

- sessions
- transactions
- parameterized Cypher
- node and relationship modeling

#### Step 2. Design the extraction prompt

Prompt the model to return structured JSON only.

Important rules:

- restrict allowed entity types
- restrict allowed relationship types
- prefer precision over recall
- skip uncertain triples

Learn:

- extraction quality depends heavily on schema constraints
- too-open extraction creates graph noise

#### Step 3. Normalize entities

Before writing to Neo4j, normalize names:

- lowercase for matching
- trim whitespace
- clean punctuation
- use canonical paper IDs from arXiv when available

Learn:

- entity deduplication is one of the hardest parts of graph systems
- a noisy graph weakens GraphRAG badly

#### Step 4. Write graph data

Use `MERGE` for nodes and relationships where appropriate so repeated runs do not duplicate the graph.

Learn:

- idempotent ingestion
- property graph structure
- how Cypher differs from SQL

### Deliverable

Open Neo4j Browser and inspect a graph where:

- authors connect to papers
- papers connect to methods or concepts
- institutions connect to authors

### What you should understand before leaving this phase

- why graph schema design matters more than the extraction prompt alone
- why normalization is essential
- what Neo4j stores that a vector DB does not

### Resources

- Neo4j Cypher manual
- Neo4j GraphAcademy courses
- GraphRAG paper sections on graph indexing

***

## Phase 2B — Graph retrieval and GraphRAG answering

### Objective

Use both Pinecone and Neo4j at query time to answer relationship-heavy questions better than standard RAG.

### Why this phase matters

This phase turns the graph from a visual artifact into a reasoning aid.

### System diagram

```text
User query
   |
   +--> query embedding ---------> Pinecone top-k chunks
   |
   +--> query entity extraction -> Neo4j neighborhood query
                                      |
                                      v
                               nodes + edges summary
                                      |
                                      v
                        combined evidence for synthesis model
                                      |
                                      v
                                   final answer
```

### Files to build

1. `backend/graph_store/graph_retriever.py`
2. `backend/pipelines/graphrag_pipeline.py`
3. update `backend/routers/query.py`

### Step-by-step build order

#### Step 1. Extract query entities

Identify important entities or concepts in the user question.

Learn:

- query understanding matters for graph lookup
- query entity extraction can be simpler than ingestion extraction

#### Step 2. Retrieve vector evidence

Reuse the standard RAG retriever.

Learn:

- GraphRAG should build on the baseline, not replace it entirely

#### Step 3. Retrieve graph evidence

Start with small traversals:

- 1-hop neighborhood first
- 2-hop only if needed

Example Cypher idea:

```text
MATCH (n {name_normalized: $entity})-[r*1..2]-(m)
RETURN n, r, m
LIMIT 50
```

Learn:

- multi-hop retrieval can improve relational questions
- large subgraphs create prompt noise

#### Step 4. Summarize the graph context

Do not dump raw graph objects into the LLM.

Convert graph results into structured evidence like:

- Author A authored Paper B
- Paper B proposes Method C
- Author A is affiliated with Institution D

Learn:

- the LLM needs concise evidence, not database internals

#### Step 5. Merge vector and graph evidence

Build a prompt that includes:

- the question
- relevant text passages
- graph facts
- instructions to cite both when used

Learn:

- GraphRAG is dual evidence retrieval, not just graph lookup

### Deliverable

`/query/graphrag` returns:

- answer
- retrieved chunks
- graph evidence
- graph visualization JSON
- latency and token metrics

### What you should understand before leaving this phase

- when GraphRAG helps and when it adds noise
- why graph evidence must be summarized before prompting
- why vector retrieval still matters even in GraphRAG

### Resources

- GraphRAG paper
- Neo4j Cypher manual
- current LangGraph documentation only as background, not required for this phase

***

## Phase 3 — Evaluation and A/B comparison

### Objective

Measure the difference between Standard RAG and GraphRAG on the same question set.

### Why this phase matters

This is what turns the project into a credible ML engineering portfolio piece.

### Evaluation diagram

```text
benchmark questions
      |
      +--> Standard RAG ----> answers ----\
      |                                    \
      +--> GraphRAG -------> answers -------> evaluator -> report
                                           /
                          latency + cost --/
```

### Files to build

1. `backend/evaluation/benchmark_questions.py`
2. `backend/evaluation/cost_tracker.py`
3. `backend/evaluation/ragas_evaluator.py`
4. `backend/routers/evaluate.py`

### Step-by-step build order

#### Step 1. Create a benchmark question set

Use three categories:

- simple factual or summarization
- entity-centric
- multi-hop relational

Learn:

- a benchmark should test different query types
- GraphRAG should not be expected to win every category

#### Step 2. Capture operational metrics

Record:

- latency
- prompt tokens
- completion tokens
- estimated total cost

Learn:

- good ML systems are judged by quality and efficiency

#### Step 3. Run RAGAS

Evaluate both pipelines on:

- faithfulness
- answer relevancy
- context precision
- context recall

Learn:

- automated evaluation is useful but imperfect
- it should be complemented by manual analysis

#### Step 4. Add manual error analysis

For a smaller subset of questions, manually inspect:

- whether the answer is actually correct
- whether citations are useful
- whether graph evidence helped

Learn:

- evaluation frameworks can miss subtle failure modes
- strong projects show both numbers and examples

### Deliverable

`/evaluate` returns:

- per-question results for both pipelines
- aggregated RAGAS scores
- latency and cost comparison
- a short summary of where GraphRAG helped

### What you should understand before leaving this phase

- how to compare retrieval systems rigorously
- why automated metrics are not enough by themselves
- how to explain tradeoffs honestly in interviews

### Resources

- RAGAS docs
- Arize and other evaluation blogs for LLM systems
- GraphRAG paper for framing global vs local reasoning gains

***

## Phase 4 — Multi-agent orchestration with LangGraph

### Objective

Add a stateful orchestration layer that chooses the cheapest sufficient retrieval path for each query, decomposes complex questions, runs retrievers in parallel when useful, and returns a transparent execution trace.

### Why this phase matters

This is the "latest AI systems" layer of the project. The goal is not to make every step agentic. The goal is to show judgment: simple factual questions should use the fast vector path, while relational and multi-hop questions should trigger graph retrieval, planning, and evidence reconciliation.

### Endpoint

Add:

- `POST /query/agent`

Keep these direct endpoints available for debugging and A/B comparison:

- `POST /query/rag`
- `POST /query/graphrag`
- `POST /evaluate`

### Agent graph

```text
User query
   |
   v
QueryClassifier
   |
   +---------------- simple ----------------+
   |                                        |
   v                                        |
VectorRetriever                             |
   |                                        |
   +--------------------+-------------------+
                        |
                    Synthesizer
                        |
                        v
                     answer


User query
   |
   v
QueryClassifier
   |
   +--------------- relational / multi-hop ---------------+
                                                           |
                                                           v
                                                        Planner
                                                           |
                                    +----------------------+----------------------+
                                    |                                             |
                                    v                                             v
                             VectorRetriever                              GraphRetriever
                                    |                                             |
                                    +----------------------+----------------------+
                                                           |
                                                           v
                                                    EvidenceMerger
                                                           |
                                                           v
                                                     Synthesizer
                                                           |
                                                           v
                                       structured answer + citations + agent_trace[]
```

### Agents to build

1. `QueryClassifier`
   - labels the query as `simple_factual`, `summarization`, `entity_centric`, `relational`, or `multi_hop`
   - decides whether Neo4j is needed
   - emits a confidence score and routing reason

2. `Planner`
   - decomposes relational questions into sub-questions
   - identifies likely paper, method, concept, and author entities
   - decides whether to run vector retrieval, graph retrieval, or both

3. `VectorRetriever`
   - reuses the Phase 1 retriever
   - returns ranked text chunks with `paper_id`, `chunk_index`, score, and source title

4. `GraphRetriever`
   - reuses the Phase 2B GraphRAG retrieval logic
   - returns graph edges, graph evidence chunks, related papers, and contested edges

5. `EvidenceMerger`
   - deduplicates chunks by `(paper_id, chunk_index)`
   - merges graph facts with text evidence
   - detects `CONTRADICTS` edges and marks them as contested claims
   - optionally reranks the final evidence packet before synthesis

6. `Synthesizer`
   - chooses the prompt style based on route:
     - concise factual answer
     - literature synthesis
     - comparison table
     - contested findings summary
   - answers only from retrieved evidence
   - returns citations and insufficient-evidence notes when needed

### Shared state shape

```text
{
  "question": "...",
  "query_type": "relational",
  "route": "vector_only | graph_only | hybrid_parallel",
  "sub_questions": [],
  "vector_contexts": [],
  "graph_edges": [],
  "merged_contexts": [],
  "contested_claims": [],
  "answer": "...",
  "citations": [],
  "agent_trace": [
    {
      "agent": "QueryClassifier",
      "decision": "hybrid_parallel",
      "latency_ms": 42,
      "tokens": 120,
      "cost_usd": 0.0001
    }
  ]
}
```

### Deliverable

`/query/agent` returns:

- final answer
- citations
- retrieved text contexts
- graph evidence summary
- contested claims, if any
- selected route
- `agent_trace[]` with latency, token, cost, and decision metadata

### What you should understand before leaving this phase

- why orchestration is different from retrieval
- why state machines fit multi-step agent workflows
- when a pipeline deserves agent-style routing
- how routing can reduce latency and cost without reducing answer quality
- how to make agent systems observable enough to debug

### Resources

- LangGraph docs and tutorials
- LangSmith tracing docs if you later add observability
- RAGAS agent and tool-use metrics for later evaluation

### Resume bullets unlocked by this phase

- Built a LangGraph multi-agent RAG orchestrator with query classification, planning, parallel retrieval, evidence merging, and grounded synthesis.
- Reduced unnecessary graph lookups by routing simple factual questions to vector retrieval only.
- Added traceable agent execution with per-node latency, token, cost, and route-decision metadata.
- Designed contested-claim handling for graph edges that represent contradictory findings across papers.

***

## Phase 5 — Frontend and developer-facing visualization

### Objective

Make the project easier to understand visually and easier to demo.

### Why this phase matters

A simple frontend helps communicate the project clearly even if the primary goal is learning rather than polished product design.

### Frontend pages

- `Query.tsx`: ask a question and compare RAG vs GraphRAG answers
- `AgentTrace.tsx`: show route decisions, agent timings, and cost per step
- `Graph.tsx`: show graph subgraph returned for GraphRAG
- `Evaluation.tsx`: show comparison table and charts
- show `trace_id` and link to LangSmith for each run

### Suggested visuals

- answer comparison view for RAG, GraphRAG, and Agent RAG
- graph visualization with highlighted query entity nodes
- route decision timeline
- latency bar chart
- cost bar chart
- RAGAS score comparison chart

### Deliverable

A minimal UI that helps you inspect and explain the system.

### What you should understand before leaving this phase

- how system outputs should be exposed for debugging and explanation
- why visibility is important for trust
- how trace views make agent systems easier to debug

***

## Phase 6 — Polish, tracing, deployment, and portfolio packaging

### Objective

Turn the project into a strong portfolio artifact after the core technical value is proven.

### Optional additions

- LangSmith tracing (implemented)
- deployment
- README diagrams
- architecture screenshots
- Neo4j Browser screenshots
- benchmark result tables
- short technical write-up

### Best portfolio assets

- architecture diagram
- screenshot of graph visualization
- benchmark table comparing both pipelines
- short write-up describing where GraphRAG wins and loses

### Deliverable

A polished portfolio repository and demo-ready presentation.

***

## Recommended implementation order summary

```text
Phase 0: scope + benchmark + schema
Phase 1: ingestion + Standard RAG
Phase 2A: graph construction
Phase 2B: GraphRAG retrieval
Phase 3: RAGAS evaluation + A/B comparison
Phase 4: LangGraph multi-agent orchestration
Phase 5: frontend and visualization
Phase 6: tracing, deployment, and portfolio packaging
```

***

## Learning checklist by topic

### Retrieval

- what embeddings represent
- why cosine similarity works
- chunking tradeoffs
- metadata filters
- grounding and citations

### Graphs

- nodes and relationships
- schema design
- normalization and deduplication
- Cypher basics
- why multi-hop reasoning is different from similarity search

### Evaluation

- faithfulness
- answer relevance
- context precision
- context recall
- cost and latency tradeoffs

### Backend engineering

- FastAPI routing
- Pydantic schemas
- config management
- idempotent ingestion
- error handling and retries

***

## High-impact additions for a stronger AI/ML portfolio

These additions are not all required, but they are the ones most likely to make the project stand out for AI/ML engineering roles.

### Retrieval quality upgrades

- Add hybrid retrieval: dense embeddings plus sparse keyword retrieval for exact paper, method, and acronym matches.
- Add a reranker after retrieval so final evidence is ordered by relevance before synthesis.
- Add section-aware chunk metadata such as `abstract`, `method`, `experiments`, `limitations`, and `references`.
- Add citation validation so every cited `[paper_id:chunk_index]` appears in the returned context.

### Graph quality upgrades

- Add entity resolution beyond lowercasing, especially for method aliases and author name variants.
- Add graph quality checks:
  - duplicate node rate
  - dangling edge count
  - low-confidence edge count
  - relation distribution by type
- Add graph provenance fields: extraction model, extraction timestamp, source chunk, confidence, and prompt version.

### Evaluation upgrades

- Store evaluation runs as artifacts so results are reproducible.
- Compare `rag`, `graphrag`, and `agent` on the same benchmark.
- Track quality, latency, token usage, estimated cost, route selected, and graph hops used.
- Add manual error categories:
  - missing evidence
  - weak citation
  - irrelevant retrieval
  - graph noise
  - synthesis hallucination

### Agent and observability upgrades

- Add LangSmith or OpenTelemetry tracing for each agent node.
- Return `agent_trace[]` from `/query/agent` for local debugging and frontend visualization.
- Add route accuracy evaluation: did the classifier choose the cheapest route that still answered correctly?
- Add a failure fallback: if graph retrieval fails, continue with vector retrieval and mark the response as degraded.

### Production-readiness upgrades

- Add Docker Compose for FastAPI, Neo4j, and local Ollama setup notes.
- Add CI checks for linting, basic tests, and import health.
- Add request IDs and structured JSON logs.
- Add rate limiting and input size limits.
- Treat LLM outputs as untrusted: validate JSON, constrain Cypher, and never execute model-generated database queries directly.

### Portfolio deliverables

- One architecture diagram showing ingestion, RAG, GraphRAG, evaluation, and multi-agent routing.
- One benchmark table with quality, latency, and cost.
- One graph visualization screenshot.
- One agent trace screenshot.
- One short technical blog-style write-up explaining when GraphRAG helps and when it is overkill.

***

## Risks and how to handle them

### Risk 1. PDF parsing noise

Mitigation:

- cache parsed output
- inspect bad papers manually
- remove references or appendix if they pollute retrieval

### Risk 2. Graph noise from bad extraction

Mitigation:

- keep schema small
- prefer precision over recall
- normalize entities
- validate a small subset manually before large ingestion

### Risk 3. GraphRAG adds too much prompt noise

Mitigation:

- summarize graph evidence before prompting
- limit hop count and edge count
- compare 1-hop vs 2-hop retrieval

### Risk 4. Evaluation is hard for open-ended questions

Mitigation:

- create question categories
- use both RAGAS and manual review
- avoid claiming exact scientific truth when evidence is mixed

***

## What success looks like

By the end of the project, you should be able to say:

- I built a domain-specific RAG system over arXiv ML papers
- I extended it with a Neo4j knowledge graph to support GraphRAG
- I compared Standard RAG and GraphRAG on curated benchmark questions
- I measured quality, latency, and cost
- I understand when GraphRAG helps and when it is unnecessary

That is the core portfolio outcome.

***

## Immediate next step

Since the core system is implemented end-to-end, the next best steps are portfolio packaging:

1. capture screenshots: UI (Query/Graph/AgentTrace/Evaluation), Neo4j Browser, and LangSmith traces
2. run evaluation with a fixed configuration and paste the summary table into the README
3. record a short demo video following the recommended flow (UI → LangSmith trace)
4. optionally add deployment and CI if you want “production-readiness” signals

***

## What was added beyond the original plan

- LangSmith tracing across RAG, GraphRAG, Agent, and evaluation judge scoring (per-step spans + metadata)
- request `trace_id` propagation (response field + `X-Trace-Id` header) and frontend “Copy/Open LangSmith”
- evaluation includes optional Agent comparison and judge metrics aggregation
