"""Router para gerenciamento do Threshold Dinâmico."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from ..services.threshold_manager import ThresholdManager


router = APIRouter(prefix="/threshold", tags=["Threshold Dinâmico"])

threshold_manager = ThresholdManager()


class FeedbackRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    notified: bool
    executed: bool
    fbm_score: float = Field(..., ge=0)


class ThresholdConfigRequest(BaseModel):
    threshold: Optional[float] = Field(None, ge=0)
    min_threshold: Optional[float] = Field(None, ge=0)
    max_threshold: Optional[float] = Field(None, ge=0)
    adjustment_step: Optional[float] = Field(None, gt=0)


class NotifyCheckRequest(BaseModel):
    fbm_score: float = Field(..., ge=0)


class ThresholdResponse(BaseModel):
    user_id: str
    current_threshold: float
    message: str


class FeedbackResponse(BaseModel):
    feedback_type: str
    old_threshold: float
    new_threshold: float
    adjustment: float
    fbm_score: float
    stats: Dict[str, int]


class NotifyCheckResponse(BaseModel):
    should_notify: bool
    fbm_score: float
    threshold: float
    margin: float
    confidence: float


@router.get("/{user_id}", response_model=ThresholdResponse)
async def get_threshold(user_id: str):
    threshold = threshold_manager.get_threshold(user_id)
    
    return ThresholdResponse(
        user_id=user_id,
        current_threshold=threshold,
        message=f"Threshold atual: {threshold}"
    )


@router.post("/{user_id}/feedback", response_model=FeedbackResponse)
async def register_feedback(user_id: str, feedback: FeedbackRequest):
    """
    Registra feedback de uma notificação e ajusta o threshold.
    
    Tipos de feedback:
    - **VP**: Notificou + Executou → Mantém threshold
    - **VN**: Notificou + Não Executou → Aumenta threshold
    - **FP**: Não Notificou + Executou → Diminui threshold
    - **FN**: Não Notificou + Não Executou → Mantém threshold
    """
    result = threshold_manager.update_threshold(
        user_id=user_id,
        hour=feedback.hour,
        notified=feedback.notified,
        executed=feedback.executed,
        fbm_score=feedback.fbm_score
    )
    
    return FeedbackResponse(**result)


@router.post("/{user_id}/check", response_model=NotifyCheckResponse)
async def check_should_notify(user_id: str, request: NotifyCheckRequest):
    result = threshold_manager.should_notify(user_id, request.fbm_score)
    return NotifyCheckResponse(**result)


@router.get("/{user_id}/stats")
async def get_statistics(user_id: str):
    stats = threshold_manager.get_statistics(user_id)
    return stats


@router.get("/{user_id}/history")
async def get_history(
    user_id: str,
    limit: int = Query(100, ge=1, le=1000)
):
    history = threshold_manager.get_history(user_id, limit)
    
    return {
        "user_id": user_id,
        "count": len(history),
        "history": history
    }


@router.post("/{user_id}/config")
async def configure_threshold(user_id: str, config: ThresholdConfigRequest):
    if config.threshold is None and config.min_threshold is None and \
       config.max_threshold is None and config.adjustment_step is None:
        raise HTTPException(
            status_code=400,
            detail="Pelo menos um parâmetro deve ser fornecido"
        )
    
    result = threshold_manager.set_threshold(
        user_id=user_id,
        threshold=config.threshold or threshold_manager.get_threshold(user_id),
        min_threshold=config.min_threshold,
        max_threshold=config.max_threshold,
        adjustment_step=config.adjustment_step
    )
    
    return {"user_id": user_id, "updated": True, **result}


@router.post("/{user_id}/reset")
async def reset_threshold(user_id: str):
    result = threshold_manager.reset_threshold(user_id)
    return {"user_id": user_id, **result}


@router.get("/{user_id}/decision/{fbm_score}")
async def get_decision(user_id: str, fbm_score: float):
    result = threshold_manager.should_notify(user_id, fbm_score)
    
    emoji = "✅" if result["should_notify"] else "❌"
    
    return {
        "user_id": user_id,
        "fbm_score": fbm_score,
        "threshold": result["threshold"],
        "decision": emoji + (" NOTIFICAR" if result["should_notify"] else " NÃO NOTIFICAR"),
        "should_notify": result["should_notify"],
        "margin": round(result["margin"], 2)
    }
