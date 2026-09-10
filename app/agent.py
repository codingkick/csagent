import os
from dotenv import load_dotenv
from app.schemas import CustomerMessageRequest, AgentResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


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

        # Prompt template with proper input variable placeholders {message} and {brand}
        self.prompt = ChatPromptTemplate.from_template(
            """Act as an AI customer support agent for the brand '{brand}'.
Generate a structured response for the customer message below.

Customer Message: "{message}"

Task:
1. Identify the intent of the message from the allowed IntentCategory options.
2. Provide a confidence score (0.0 to 1.0) for your intent classification.
3. Draft a polite, helpful reply for the customer.
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
        response = await chain.ainvoke({"message": request.message, "brand": request.brand})
        return response


# Singleton instance of the SupportAgent pipeline
agent = SupportAgent()
