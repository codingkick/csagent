import asyncio
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from app.schemas import CustomerMessageRequest, AgentResponse
from app.agent import agent

app = FastAPI(
    title="Hiver AI Support Agent API",
    description="FastAPI microservice for processing customer messages, intent classification, grounded reply drafting, and escalation routing.",
    version="1.0.0",
)


@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    """Returns status OK if service is healthy."""
    return {"status": "ok", "service": "support-agent-apis"}


@app.post(
    "/api/v1/message",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Process incoming customer message",
    description="Classifies incoming message into intent, drafts grounded response, and decides auto-handling vs escalation.",
)
async def handle_customer_message(request: CustomerMessageRequest):
    try:
        response = await agent.process_message(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing support request: {str(e)}",
        )


@app.post(
    "/api/v1/message/stream",
    summary="Stream response tokens (SSE)",
    description="Streams the agent's drafted reply token-by-token using Server-Sent Events (SSE).",
)
async def stream_customer_message(request: CustomerMessageRequest):
    async def event_generator():
        processed = await agent.process_message(request)

        # 1. Yield Metadata Event
        yield f"data: {{'intent': '{processed.intent.value}', 'auto_handled': {str(processed.auto_handled).lower()}}}\n\n"

        # 2. Yield Word/Token Stream
        words = processed.drafted_reply.split(" ")
        for word in words:
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.04)

        # 3. Stream Completion Event
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
