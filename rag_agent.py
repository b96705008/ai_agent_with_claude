"""Simple RAG (Retrieval-Augmented Generation) agent using LangGraph.

This module implements a basic RAG pipeline that:
1. Retrieves relevant documents from a vector store
2. Formats the context from retrieved documents
3. Generates answers using the retrieved context

The agent uses LangGraph for workflow orchestration and FAISS
for vector similarity search.
"""
import contextlib
import logging
import os
from typing import Any, List, Optional, TypedDict

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from trulens.apps.langgraph import TruGraph
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

os.environ["TRULENS_OTEL_TRACING"] = "1"


class RAGState(TypedDict):
    """State object for the RAG workflow.

    Attributes:
        question: The user's input question
        documents: Retrieved documents from vector store
        context: Formatted context string from documents
        answer: Generated answer from the LLM
        messages: Conversation history as LangChain messages
    """
    question: str
    documents: List[Document]
    context: str
    answer: str
    messages: List[Any]


class SimpleRAGAgent:
    """A simple RAG agent using LangGraph for workflow orchestration.

    This agent implements a three-step RAG pipeline:
    1. Document retrieval using vector similarity search
    2. Context formatting from retrieved documents
    3. Answer generation using retrieved context

    The agent uses FAISS as the vector store and OpenAI for embeddings and generation.
    """

    def __init__(
        self,
        model: str = 'gpt-4',
        temperature: float = 0,
        chunk_size: int = 1000,
        enable_trulens: bool = True,
        app_name: str = "RAG_Agent",
        app_version: str = "v1.0"
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "Please set the OPENAI_API_KEY environment variable."
            )
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings()
        self.vector_store: Optional[FAISS] = None
        self.compiled_graph = None  # Cache for the compiled graph
        self.tru_recorder = None  # TruLens recorder
        self.enable_trulens = enable_trulens
        self.app_name = app_name
        self.app_version = app_version
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
        )

        # Initialize TruLens session if enabled
        if self.enable_trulens:
            self.tru_session = TruSession()
            self.tru_session.reset_database()

    def setup_vector_store(self, documents: List[str]):
        """Initialize vector store with provided documents"""
        docs = []
        for i, doc_text in enumerate(documents):
            chunks = self.text_splitter.split_text(doc_text)
            for j, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": f"doc_{i}", "chunk": j}
                ))

        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on the question"""
        if not self.vector_store:
            state["documents"] = []
            return state

        docs = self.vector_store.similarity_search(
            state["question"],
            k=3
        )
        state["documents"] = docs
        return state

    def format_context(self, state: RAGState) -> RAGState:
        """Format retrieved documents into context string"""
        context_parts = []
        for doc in state["documents"]:
            context_parts.append(f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}")

        state["context"] = "\n\n---\n\n".join(context_parts)
        return state

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using retrieved context"""
        prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so clearly.

Context:
{state["context"]}

Question: {state["question"]}

Answer:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        state["answer"] = response.content
        state["messages"] = [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["answer"])
        ]
        return state

    def create_graph(self):
        """Create and compile the RAG workflow graph"""
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("format", self.format_context)
        workflow.add_node("generate", self.generate_answer)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "format")
        workflow.add_edge("format", "generate")
        workflow.add_edge("generate", END)

        compiled_graph = workflow.compile()

        # Wrap with TruGraph for observability if enabled
        if self.enable_trulens:
            self.tru_recorder = TruGraph(
                compiled_graph,
                app_name=self.app_name,
                app_version=self.app_version
            )
        return compiled_graph

    def query(self, question: str) -> str:
        """Main interface to query the RAG agent"""
        if not self.vector_store:
            return "Please set up the vector store first using setup_vector_store()"

        try:
            # Create and cache graph on first use
            if self.compiled_graph is None:
                self.compiled_graph = self.create_graph()

            # Use the appropriate invoke method based on TruLens enablement
            init_state = {
                "question": question,
                "documents": [],
                "context": "",
                "answer": "",
                "messages": []
            }

            cm = (
                self.tru_recorder
                if self.enable_trulens and self.tru_recorder
                else contextlib.nullcontext()
            )
            with cm:
                result = self.compiled_graph.invoke(init_state)
            return result["answer"]
        except Exception as e:
            logging.exception("Error during RAG query")
            return f"An error occurred: {str(e)}"


def main():
    """Main function to demonstrate the RAG agent functionality with TruLens."""
    # Example usage with TruLens observability enabled
    agent = SimpleRAGAgent(
        enable_trulens=True,
        app_name="Simple_RAG_Demo",
        app_version="v1.0"
    )

    # Sample documents for demonstration
    sample_docs = [
        """
        LangGraph is a library for building stateful, multi-actor applications with LLMs.
        It extends LangChain Expression Language with the ability to coordinate multiple chains
        across multiple steps of computation in a cyclic manner. It's inspired by Pregel and Apache Beam.
        """,
        """
        RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval
        with text generation. It works by first retrieving relevant documents from a knowledge base,
        then using those documents as context to generate more accurate and informed responses.
        """,
        """
        Vector stores are databases optimized for storing and querying high-dimensional vectors.
        They are commonly used in machine learning applications for similarity search and
        recommendation systems. FAISS is a popular vector store implementation.
        """
    ]

    # Set up the vector store with sample documents
    agent.setup_vector_store(sample_docs)

    # Example queries
    questions = [
        "What is LangGraph?",
        "How does RAG work?",
        "What are vector stores used for?"
    ]

    print("RAG Agent Demo with TruLens Observability")
    print("=" * 50)

    for question in questions:
        print(f"\nQuestion: {question}")
        answer = agent.query(question)
        print(f"Answer: {answer}")
        print("-" * 30)

    # Display TruLens observability info
    if agent.enable_trulens and agent.tru_session:
        print("\n" + "=" * 50)
        print("TruLens Observability Enabled")
        print("Check TruLens dashboard for detailed traces and evaluations")
        print("Run 'tru.run_dashboard()' to view the dashboard")
        run_dashboard(agent.tru_session)


if __name__ == "__main__":
    main()
