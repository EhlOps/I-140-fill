import os
import zipfile

from fastapi import FastAPI
from fastapi.responses import FileResponse

from agent import run_graph

app = FastAPI()


@app.get("/agent")
async def chatbot_endpoint(user_input: str):
    return await run_graph(user_input)


@app.get("/download")
async def download_endpoint(user_input: str):
    with open("/code/src/addition_information.json", "wb") as f:
        f.write(user_input.encode("utf-8"))
    with zipfile.ZipFile("/code/src/i-140.zip", "w") as zipf:
        zipf.write("/code/src/i-140-filled.pdf", arcname="i-140-filled.pdf")
        zipf.write(
            "/code/src/addition_information.json", arcname="addition_information.json"
        )
    if os.path.exists("/code/src/addition_information.json"):
        os.remove("/code/src/addition_information.json")
    return FileResponse(
        "/code/src/i-140.zip",
        media_type="application/zip",
        filename="i-140.zip",
    )
