# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI agent project demonstrating OpenAI API usage and RAG (Retrieval-Augmented Generation) implementation. The project consists of:

- `rag_agent.py`: A LangGraph-based RAG agent that combines document retrieval with text generation

## Architecture

The project contains two main components:

### rag_agent.py (LangGraph RAG Pipeline)
- **SimpleRAGAgent**: Main agent class implementing RAG workflow with configurable parameters
- **Vector Store**: FAISS-based document storage with OpenAI embeddings
- **LangGraph Workflow**: Stateful graph with nodes for retrieve → format → generate
- **Document Processing**: RecursiveCharacterTextSplitter for text chunking (default 1000 chars)
- **TruLens Integration**: Optional observability with TruGraph for tracing and evaluation (enabled by default)
- **Error Handling**: Comprehensive exception handling with logging
- **Environment Setup**: Requires OPENAI_API_KEY environment variable

## Development Commands

### Environment Setup
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
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

### TruLens Dashboard
The agent includes TruLens observability. After running queries, you can view traces and evaluations in the TruLens dashboard, which starts automatically in the demo.

## Key Features

- **Configurable Model**: Uses OpenAI's API with "gpt-4" by default (configurable)
- **Document Chunking**: Configurable chunk size (default: 1000 characters with 20% overlap)
- **Similarity Search**: Retrieves top 3 most relevant document chunks
- **State Management**: Uses TypedDict for workflow state tracking
- **Observability**: Built-in TruLens integration for monitoring and evaluation
- **Error Resilience**: Comprehensive error handling with informative messages

## Technical Notes

- No testing framework, linting, or build tools are currently configured
- No package management files (requirements.txt, pyproject.toml, etc.) are present
- TruLens database is reset on each agent initialization