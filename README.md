# Agentic Research RAG

A production-oriented Agentic Retrieval-Augmented Generation system
for complex question answering over scientific research papers.

## Goal

Build an end-to-end research assistant capable of:

- hybrid retrieval
- dense retrieval
- sparse retrieval
- reciprocal rank fusion
- cross-encoder reranking
- query routing
- query decomposition
- multi-hop retrieval
- query rewriting
- self-correcting retrieval
- grounded generation
- citation verification
- hallucination detection

The project will initially operate over approximately 10,000
domain-specific research papers and later scale beyond 50,000 papers.

## Research Domain

The initial corpus focuses on:

- Retrieval-Augmented Generation
- Information Retrieval
- Dense Retrieval
- Sparse Retrieval
- Hybrid Retrieval
- Neural Retrieval
- Reranking
- Multi-hop Question Answering
- LLM Agents
- Query Rewriting
- Grounding
- Hallucination Detection
- Citation Verification

## Data Source

Research-paper metadata and available full text are sourced from OpenAlex.

## Development Philosophy

The project is built incrementally using feature branches.

Each major component is:

1. implemented
2. tested
3. evaluated
4. reviewed
5. merged into main

## Current Phase

OpenAlex data ingestion pipeline.