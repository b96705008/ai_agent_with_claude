# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI agent project demonstrating OpenAI API usage and RAG (Retrieval-Augmented Generation) implementation. The project consists of:

- `example.py`: A basic script that uses the OpenAI client to generate text responses using GPT-4.1
- `rag_agent.py`: A LangGraph-based RAG agent that combines document retrieval with text generation

## Architecture

The project contains two main components:

### example.py
- OpenAI client initialization
- API call to generate text using the `responses.create` method
- Output printing

### rag_agent.py (LangGraph RAG Pipeline)
- **SimpleRAGAgent**: Main agent class implementing RAG workflow
- **Vector Store**: FAISS-based document storage with embeddings
- **LangGraph Workflow**: Stateful graph with nodes for retrieve → format → generate
- **Document Processing**: Text splitting and chunking for optimal retrieval
- **TruLens Integration**: Optional observability with TruGraph for tracing and evaluation

## Development Commands

To run the basic OpenAI example:
```bash
python example.py
```

To run the RAG agent demo:
```bash
python rag_agent.py
```

### Dependencies
Install required packages:
```bash
pip install openai langgraph langchain langchain-openai langchain-community faiss-cpu trulens
```

## Key Notes

- The project uses OpenAI's API with model "gpt-4.1"
- No testing framework, linting, or build tools are currently configured
- No package management files (requirements.txt, pyproject.toml, etc.) are present