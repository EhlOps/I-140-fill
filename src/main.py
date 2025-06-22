from fastapi import FastAPI
from fastapi.responses import FileResponse

from agent import run_graph

app = FastAPI()


@app.get("/agent")
async def chatbot_endpoint(user_input: str):
    return await run_graph(user_input)


@app.get("/download")
async def download_endpoint():
    return FileResponse(
        "/code/src/i-140-filled.pdf",
        media_type="application/pdf",
        filename="i-140-filled.pdf",
    )
