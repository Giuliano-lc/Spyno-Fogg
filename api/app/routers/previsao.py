"""Rota /previsao - Usa modelo PPO para prever melhores horários."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.services import StorageService, ThresholdManager
from app.rl.trainer import RLTrainer


router = APIRouter(prefix="/previsao", tags=["Previsão"])

storage = StorageService()
threshold_manager = ThresholdManager()


class HourPrediction(BaseModel):
    hour: int = Field(..., description="Hora do dia (0-23)")
    recommended: bool = Field(..., description="Se deve notificar")
    probability: float = Field(..., description="Probabilidade (0-1)")
    fbm_score: int = Field(..., description="Score FBM")
    motivation: int = Field(..., description="Motivação (0-4)")
    ability: int = Field(..., description="Habilidade (0-4)")
    trigger: int = Field(..., description="Gatilho (0-6)")
    sleeping: bool = Field(..., description="Dormindo")
    above_threshold: bool = Field(True, description="FBM >= threshold")
    threshold: float = Field(15.0, description="Threshold dinâmico")


class DayPredictionResponse(BaseModel):
    user_id: str
    model_loaded: bool
    method: str = Field(..., description="'ppo' ou 'heuristic'")
    recommended_hours: List[int] = Field(..., description="Top 3 horas (filtradas)")
    recommended_hours_raw: List[int] = Field([], description="Antes do filtro")
    current_threshold: float = Field(15.0, description="Threshold dinâmico")
    all_hours: List[HourPrediction]
    summary: Dict[str, Any]


class PredictionRequest(BaseModel):
    hours_data: List[Dict[str, Any]] = Field(..., min_length=24, max_length=24)


def calculate_fbm_from_hour_data(hour_data: Dict) -> Dict[str, int]:
    mf = hour_data.get("motivation_factors", {})
    af = hour_data.get("ability_factors", {})
    tf = hour_data.get("trigger_factors", {})
    ctx = hour_data.get("context", {})
    
    # Motivação
    m = (1 if mf.get("valence", 0) == 1 else 0) + \
        1 + \
        (1 if mf.get("last_activity_score", 0) == 1 else 0) + \
        (1 if mf.get("hours_slept_last_night", 0) >= 7 else 0)
    
    # Habilidade
    a = (1 if af.get("cognitive_load", 1) == 0 else 0) + \
        (1 if af.get("activities_performed_today", 0) <= 1 else 0) + \
        (1 if af.get("time_since_last_activity_hours", 0) >= 1 else 0) + \
        (1 if af.get("confidence_score", 0) >= 4 else 0)
    
    # Gatilho
    sleeping = tf.get("sleeping", False)
    if sleeping:
        t = 0
    else:
        t = 1 + \
            (1 if tf.get("arousal", 0) == 1 else 0) + \
            (1 if tf.get("location", "") == "home" else 0) + \
            (1 if tf.get("motion_activity", "") == "stationary" else 0) + \
            (1 if ctx.get("day_period", 0) == 1 else 0) + \
            (1 if ctx.get("is_weekend", False) else 0)
    
    return {
        "motivation": m,
        "ability": a,
        "trigger": t,
        "fbm_score": m * a * t,
        "sleeping": sleeping
    }


def get_trainer_for_user(user_id: str) -> RLTrainer:
    trainer = RLTrainer(model_path=f"models/ppo_{user_id}")
    trainer.load_model()
    return trainer


@router.get(
    "/previsao/{user_id}",
    response_model=DayPredictionResponse,
    summary="Prevê melhores horários para notificar"
)
async def get_prediction(user_id: str) -> DayPredictionResponse:
    history = storage.get_user_history(user_id)
    
    if not history["days"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuário '{user_id}' não possui histórico"
        )
    
    last_day = history["days"][-1]
    
    day_observations = []
    for hour_data in last_day["hours"]:
        fbm = calculate_fbm_from_hour_data(hour_data)
        ctx = hour_data.get("context", {})
        
        day_observations.append({
            "hour": hour_data["hour"],
            "day_period": ctx.get("day_period", 0),
            "is_weekend": ctx.get("is_weekend", False),
            "motivation": fbm["motivation"],
            "ability": fbm["ability"],
            "trigger": fbm["trigger"],
            "fbm_score": fbm["fbm_score"],
            "sleeping": fbm["sleeping"]
        })
    
    trainer = get_trainer_for_user(user_id)
    model_loaded = trainer.model is not None
    
    prediction = trainer.predict_next_day(day_observations)
    
    current_threshold = threshold_manager.get_threshold(user_id)
    
    all_hours = []
    for obs in day_observations:
        pred = next(
            (p for p in prediction.get("all_predictions", []) if p["hour"] == obs["hour"]),
            {"recommended_action": 0, "notify_probability": 0}
        )
        
        above_threshold = obs["fbm_score"] >= current_threshold
        
        all_hours.append(HourPrediction(
            hour=obs["hour"],
            recommended=pred.get("recommended_action", 0) == 1 and above_threshold,
            probability=pred.get("notify_probability", obs["fbm_score"] / 100),
            fbm_score=obs["fbm_score"],
            motivation=obs["motivation"],
            ability=obs["ability"],
            trigger=obs["trigger"],
            sleeping=obs["sleeping"],
            above_threshold=above_threshold,
            threshold=current_threshold
        ))
    
    recommended_hours_raw = prediction.get("recommended_hours", [])[:5]
    
    recommended_hours_filtered = [
        h for h in recommended_hours_raw 
        if any(hour.hour == h and hour.above_threshold for hour in all_hours)
    ][:3]
    
    awake_hours = [h for h in all_hours if not h.sleeping]
    
    summary = {
        "total_recommended_raw": len(recommended_hours_raw),
        "total_recommended_filtered": len(recommended_hours_filtered),
        "filtered_out": len(recommended_hours_raw) - len(recommended_hours_filtered),
        "current_threshold": current_threshold,
        "avg_fbm_awake": sum(h.fbm_score for h in awake_hours) / len(awake_hours) if awake_hours else 0,
        "hours_above_threshold": len([h for h in awake_hours if h.above_threshold]),
        "best_fbm_hour": max(all_hours, key=lambda x: x.fbm_score).hour if all_hours else None,
        "sleeping_hours": len([h for h in all_hours if h.sleeping]),
        "days_of_history": history["total_days"]
    }
    
    return DayPredictionResponse(
        user_id=user_id,
        model_loaded=model_loaded,
        method="ppo" if model_loaded else "heuristic",
        recommended_hours=recommended_hours_filtered,
        recommended_hours_raw=recommended_hours_raw,
        current_threshold=current_threshold,
        all_hours=all_hours,
        summary=summary
    )


@router.post(
    "/{user_id}/custom",
    response_model=DayPredictionResponse,
    summary="Previsão com dados customizados",
    description="Faz previsão usando dados fornecidos em vez do histórico."
)
async def get_custom_prediction(
    user_id: str,
    request: PredictionRequest
) -> DayPredictionResponse:
    """Previsão com dados customizados."""
    
    # Constrói observações
    day_observations = []
    for hour_data in request.hours_data:
        fbm = calculate_fbm_from_hour_data(hour_data)
        ctx = hour_data.get("context", {})
        
        day_observations.append({
            "hour": hour_data.get("hour", 0),
            "day_period": ctx.get("day_period", 0),
            "is_weekend": ctx.get("is_weekend", False),
            "motivation": fbm["motivation"],
            "ability": fbm["ability"],
            "trigger": fbm["trigger"],
            "fbm_score": fbm["fbm_score"],
            "sleeping": fbm["sleeping"]
        })
    
    # Carrega trainer e faz previsão
    trainer = get_trainer_for_user(user_id)
    model_loaded = trainer.model is not None
    
    prediction = trainer.predict_next_day(day_observations)
    
    # Formata resposta
    all_hours = []
    for obs in day_observations:
        pred = next(
            (p for p in prediction.get("all_predictions", []) if p["hour"] == obs["hour"]),
            {"recommended_action": 0, "notify_probability": 0}
        )
        
        all_hours.append(HourPrediction(
            hour=obs["hour"],
            recommended=pred.get("recommended_action", 0) == 1,
            probability=pred.get("notify_probability", obs["fbm_score"] / 100),
            fbm_score=obs["fbm_score"],
            motivation=obs["motivation"],
            ability=obs["ability"],
            trigger=obs["trigger"],
            sleeping=obs["sleeping"]
        ))
    
    recommended_hours = prediction.get("recommended_hours", [])[:3]
    awake_hours = [h for h in all_hours if not h.sleeping]
    
    summary = {
        "total_recommended": len(recommended_hours),
        "avg_fbm_awake": sum(h.fbm_score for h in awake_hours) / len(awake_hours) if awake_hours else 0,
        "best_fbm_hour": max(all_hours, key=lambda x: x.fbm_score).hour if all_hours else None,
        "sleeping_hours": len([h for h in all_hours if h.sleeping])
    }
    
    return DayPredictionResponse(
        user_id=user_id,
        model_loaded=model_loaded,
        method="ppo" if model_loaded else "heuristic",
        recommended_hours=recommended_hours,
        all_hours=all_hours,
        summary=summary
    )


@router.get(
    "/{user_id}/simples",
    summary="Previsão simplificada",
    description="Retorna apenas as horas recomendadas."
)
async def get_simple_prediction(user_id: str) -> Dict[str, Any]:
    """Retorna apenas as horas recomendadas de forma simplificada."""
    
    full_prediction = await get_prediction(user_id)
    
    return {
        "user_id": user_id,
        "recommended_hours": full_prediction.recommended_hours,
        "method": full_prediction.method,
        "message": f"Notificar preferencialmente às: {', '.join([f'{h}h' for h in full_prediction.recommended_hours])}"
    }
