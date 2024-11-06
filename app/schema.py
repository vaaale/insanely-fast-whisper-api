from typing import Literal, List

from pydantic import BaseModel


class WebhookBody(BaseModel):
    url: str
    header: dict[str, str] = {}


class TranscribeRequest(BaseModel):
    audio: List[str] | str
    task: Literal["transcribe", "translate"] = "transcribe"
    language: str | None = None
    batch_size: int = 64
    timestamp: Literal["chunk", "word"] = "chunk"
    diarize_audio: bool = True
    webhook: WebhookBody | None = None
    is_async: bool = False
    managed_task_id: str | None = None
