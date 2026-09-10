from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    REFUND_REQUEST = "refund_request"
    ORDER_STATUS = "order_status"
    TECHNICAL_ISSUE = "technical_issue"
    ACCOUNT_ACCESS = "account_access"
    GENERAL_INQUIRY = "general_inquiry"
    UNKNOWN = "unknown"


class CustomerMessageRequest(BaseModel):
    user_id: str = Field(..., description="Unique ID for the customer", examples=["user_123"])
    tweet_id: Optional[str] = Field(None, description="Optional tweet or thread ID", examples=["tweet_98765"])
    message: str = Field(..., description="Incoming customer support query", examples=["My order hasn't arrived yet, can you check status?"])
    brand: Optional[str] = Field("default_brand", description="Targeted brand name", examples=["AppleSupport"])


class AgentResponse(BaseModel):
    intent: IntentCategory = Field(..., description="Classified intent of customer message")
    intent_confidence: float = Field(..., description="Confidence score (0.0 to 1.0)", examples=[0.95])
    drafted_reply: str = Field(..., description="Drafted reply grounded in historical resolution data")
    auto_handled: bool = Field(..., description="True if message can be auto-handled, False if escalation needed")
    escalation_reason: Optional[str] = Field(None, description="Reason for human escalation if auto_handled is False")
    retrieved_context_ids: List[str] = Field(default_factory=list, description="IDs of historical threads used as RAG context")
