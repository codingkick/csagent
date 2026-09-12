import os
from typing import List
from dotenv import load_dotenv
from pydantic import Field
from app.schemas import CustomerMessageRequest, AgentResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()


def format_docs(docs: List[Document]) -> str:
    """Helper to format retrieved historical threads into a clean text block."""
    if not docs:
        return "No relevant historical resolutions found."
    formatted = []
    for doc in docs:
        content = f"Historical Customer Issue:\n{doc.page_content}"
        brand_reply = doc.metadata.get("brand_reply")
        if brand_reply:
            content += f"\nAppleSupport Resolution:\n{brand_reply}"
        formatted.append(content)
    return "\n\n---\n\n".join(formatted)


class PineconeIntegratedRetriever(BaseRetriever):
    """
    Custom LangChain Retriever using Pinecone Integrated Inference.
    Pinecone handles text embedding internally on the server side.
    """
    index: object = Field(exclude=True)
    top_k: int = 3
    namespace: str = "__default__"

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        try:
            # Note: Pinecone's index.search requires positional parameter namespace
            response = self.index.search(
                namespace=self.namespace,
                query={"inputs": {"text": query}, "top_k": self.top_k}
            )
            hits = response.get("result", {}).get("hits", [])
            docs = []
            for hit in hits:
                fields = hit.get("fields", {})
                text = fields.get("text", "")
                if text:
                    docs.append(Document(page_content=text, metadata=fields))
            return docs
        except Exception as e:
            print(f"Pinecone Integrated Search notice: {e}")
            return []

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class SupportAgent:
    """
    Support Agent Pipeline.
    Uses Pinecone's Integrated Hosted Inference (server-side embedding)
    via custom LangChain Retriever and Gemini LLM for intent classification,
    reply drafting, and human escalation routing.
    """

    def __init__(self):
        # Retrieve API Keys & Model Names from environment
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "customer-support-resolutions")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "__default__")

        # 1. Initialize Pinecone Client & Index
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(self.index_name)

        # 2. Initialize Custom Pinecone Integrated Retriever
        self.retriever = PineconeIntegratedRetriever(
            index=self.index,
            top_k=3,
            namespace=self.namespace
        )

        # 3. Prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """Act as an AI customer support agent for the brand '{brand}'.
            Here are historical resolutions from similar past customer issues:
            <historical_resolutions>
            {context}
            </historical_resolutions>

            Customer Message: "{message}"

            Task:
            1. Identify the intent of the message.
            2. Provide a confidence score (0.0 to 1.0) for your intent classification.
            3. Draft a polite reply grounded in the historical resolutions above.
            4. Decide whether the message can be auto-handled (True) or requires human escalation (False).
            5. If auto_handled is False, provide a clear escalation_reason.
            """
        )

    async def process_message(self, request: CustomerMessageRequest) -> AgentResponse:
        # Retrieve context asynchronously via LangChain Retriever
        print("working")
        docs = await self.retriever.ainvoke(request.message)
        context_str = format_docs(docs)
        print(context_str)

        # Initialize Gemini LLM with structured output binding
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0,
            google_api_key=self.api_key,
        )
        structured_llm = llm.with_structured_output(AgentResponse)

        # Build & execute chain asynchronously
        chain = self.prompt | structured_llm
        response = await chain.ainvoke({
            "message": request.message,
            "brand": request.brand,
            "context": context_str,
        })
        return response


# Singleton instance of the SupportAgent pipeline
agent = SupportAgent()
