import asyncio
import base64
import os
import tempfile
import uuid
import logging

import requests
import torch
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse
from transformers import pipeline, WhisperFeatureExtractor, WhisperTokenizerFast, WhisperForConditionalGeneration
from .diarization_pipeline import diarize
from .schema import TranscribeRequest, WebhookBody

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("whisper.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

admin_key = os.environ.get(
    "ADMIN_KEY",
)

hf_token = os.environ.get(
    "HF_TOKEN",
)

# fly runtime env https://fly.io/docs/machines/runtime-environment
fly_machine_id = os.environ.get(
    "FLY_MACHINE_ID",
)

local_files_only = False
model_id = os.environ.get("MODEL_ID", "openai/whisper-large-v3")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_count = torch.cuda.device_count()
logger.info(f'available devices: {device_count}')
for i in range(device_count):
    logger.info(torch.cuda.get_device_name(device=f"cuda:{i}"))
logger.info(f'current device: {torch.cuda.current_device()}')

torch_dtype = torch.float16
logger.info(f"Using device: {device}")
logger.info("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    local_files_only=local_files_only,
).to(device)
logger.info("Loading tokenizer...")
tokenizer = WhisperTokenizerFast.from_pretrained(
    model_id, local_files_only=local_files_only
)
logger.info("Loading features extractor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    model_id, local_files_only=local_files_only
)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    model_kwargs={"use_flash_attention_2": True},
    torch_dtype=torch_dtype,
    device=device,
)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model="openai/whisper-large-v3",
#     torch_dtype=torch.float16,
#     device="cuda:0",
#     model_kwargs=({"attn_implementation": "flash_attention_2"}),
# )

app = FastAPI()
loop = asyncio.get_event_loop()
running_tasks = {}


def process(
    url: str,
    task: str,
    language: str | None,
    batch_size: int,
    timestamp: str,
    diarize_audio: bool,
    webhook: WebhookBody | None = None,
    task_id: str | None = None,
):
    errorMessage: str | None = None
    outputs = {}
    try:
        generate_kwargs = {
            "task": task,
            "language": language,
        }

        outputs = pipe(
            url,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if timestamp == "word" else True,
        )

        if diarize_audio is True:
            speakers_transcript = diarize(
                hf_token,
                url,
                outputs,
            )
            outputs["speakers"] = speakers_transcript
    except asyncio.CancelledError:
        errorMessage = "Task Cancelled"
    except Exception as e:
        errorMessage = str(e)

    if task_id is not None:
        del running_tasks[task_id]

    if webhook is not None:
        webhookResp = (
            {"output": outputs, "status": "completed", "task_id": task_id}
            if errorMessage is None
            else {"error": errorMessage, "status": "error", "task_id": task_id}
        )

        if fly_machine_id is not None:
            webhookResp["fly_machine_id"] = fly_machine_id

        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(webhookResp),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs


@app.middleware("http")
async def admin_key_auth_check(request: Request, call_next):
    if admin_key is not None:
        if ("x-admin-api-key" not in request.headers) or (
            request.headers["x-admin-api-key"] != admin_key
        ):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response


@app.post("/")
def root(request: TranscribeRequest):
    if request.diarize_audio is True and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if request.is_async is True and request.webhook is None:
        raise HTTPException(
            status_code=400, detail="Webhook is required for async tasks"
        )

    task_id = request.managed_task_id if request.managed_task_id is not None else str(uuid.uuid4())

    try:
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            audio_data = base64.b64decode(request.audio)
            fp.write(audio_data)
            url = fp.name
            if request.is_async is True:
                backgroundTask = asyncio.ensure_future(
                    loop.run_in_executor(
                        None,
                        process,
                        url,
                        request.task,
                        request.language,
                        request.batch_size,
                        request.timestamp,
                        request.diarize_audio,
                        request.webhook,
                        task_id,
                    )
                )
                running_tasks[task_id] = backgroundTask
                resp = {
                    "detail": "Task is being processed in the background",
                    "status": "processing",
                    "task_id": task_id,
                }
            else:
                running_tasks[task_id] = None
                outputs = process(
                    url,
                    request.task,
                    request.language,
                    request.batch_size,
                    request.timestamp,
                    request.diarize_audio,
                    request.webhook,
                    task_id,
                )
                resp = {
                    "output": outputs,
                    "status": "completed",
                    "task_id": task_id,
                }
            if fly_machine_id is not None:
                resp["fly_machine_id"] = fly_machine_id
            return resp
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {"tasks": list(running_tasks.keys())}


@app.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]

    if task is None:
        return {"status": "processing"}
    elif task.done() is False:
        return {"status": "processing"}
    else:
        return {"status": "completed", "output": task.result()}


@app.delete("/cancel/{task_id}")
def cancel(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]
    if task is None:
        return HTTPException(status_code=400, detail="Not a background task")
    elif task.done() is False:
        task.cancel()
        del running_tasks[task_id]
        return {"status": "cancelled"}
    else:
        return {"status": "completed", "output": task.result()}
