import os
from dotenv import load_dotenv
from app.schemas import CustomerMessageRequest, AgentResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

def format_docs(docs) -> str:
    """Helper to format retrieved historical threads into a clean text block."""
    if not docs:
        return "No relevant historical resolutions found."
    return "\n\n---\n\n".join(
        f"Resolution Snippet:\n{doc.page_content}" for doc in docs
    )

class SupportAgent:
    """
    Support Agent Pipeline.
    Integrates Gemini LLM via LangChain to perform intent classification,
    reply drafting, and human escalation decisions.
    """

    def __init__(self):
        # Retrieve Gemini API Key & Model Name from environment
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "customer-support-resolutions")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "models/text-embedding-001"),
            google_api_key=self.api_key,
        )
        # 2. Connect to existing production Vector DB Index
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Prompt template with proper input variable placeholders {message} and {brand}
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
        # Initialize Gemini LLM with structured output binding
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0,
            google_api_key=self.api_key,
        )
        structured_llm = llm.with_structured_output(AgentResponse)

        # Build & execute chain asynchronously
        chain = self.prompt | structured_llm
        response = await chain.ainvoke({"message": request.message, "brand": request.brand, "context": format_docs(self.retriever.invoke(request.message))})
        return response


# Singleton instance of the SupportAgent pipeline
agent = SupportAgent()
