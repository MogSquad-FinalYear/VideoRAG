"""
VideoRAG — Status Router
GET /status/{task_id} — Check processing task status.
"""
from fastapi import APIRouter, HTTPException
from backend.models import TaskStatus

router = APIRouter()


@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    """Get the status of a video processing task."""
    from backend.routers.execute import tasks

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    t = tasks[task_id]
    return TaskStatus(
        task_id=t["task_id"],
        status=t["status"],
        progress=t["progress"],
        message=t["message"],
        video_id=t.get("video_id"),
    )
