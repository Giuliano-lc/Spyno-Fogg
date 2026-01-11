"""Rota /treino - Recebe JSONs diários para treinamento do modelo RL."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any

from app.models import DailyData
from app.services import StorageService


router = APIRouter(prefix="/treino", tags=["Treino"])

storage = StorageService()


class DailyDataResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    date: str
    total_days: int
    fbm_scores: List[Dict[str, Any]]
    day_metrics: Dict[str, int]
    global_metrics: Dict[str, int]


class UserHistoryResponse(BaseModel):
    user_id: str
    total_days: int
    user_profile: Dict[str, Any] | None
    metrics: Dict[str, int]
    days_summary: List[Dict[str, Any]]


class TrainingDataResponse(BaseModel):
    user_id: str
    total_samples: int
    data: List[Dict[str, Any]]


@router.post(
    "",
    response_model=DailyDataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Recebe dados diários do usuário"
)
async def receive_daily_data(daily_data: DailyData) -> DailyDataResponse:
    try:
        # Armazena e processa os dados
        history = storage.append_daily_data(daily_data)
        
        # Recupera o último dia adicionado
        last_day = history["days"][-1]
        
        return DailyDataResponse(
            success=True,
            message=f"Dados do dia {daily_data.date} recebidos e processados com sucesso",
            user_id=daily_data.user_id,
            date=daily_data.date.isoformat(),
            total_days=history["total_days"],
            fbm_scores=last_day["fbm_scores"],
            day_metrics=last_day["day_metrics"],
            global_metrics=history["metrics"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar dados: {str(e)}"
        )


@router.get(
    "/historico/{user_id}",
    response_model=UserHistoryResponse,
    summary="Recupera histórico do usuário"
)
async def get_user_history(user_id: str) -> UserHistoryResponse:
    history = storage.get_user_history(user_id)
    
    if not history["days"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuário '{user_id}' não possui histórico"
        )
    
    days_summary = []
    for day in history["days"]:
        days_summary.append({
            "date": day["date"],
            "notifications_sent": day["day_metrics"]["notifications_sent"],
            "actions_performed": day["day_metrics"]["actions_performed"],
            "vp": day["day_metrics"]["vp"],
            "fn": day["day_metrics"]["fn"]
        })
    
    return UserHistoryResponse(
        user_id=user_id,
        total_days=history["total_days"],
        user_profile=history["user_profile"],
        metrics=history["metrics"],
        days_summary=days_summary
    )


@router.get(
    "/dados-treinamento/{user_id}",
    response_model=TrainingDataResponse,
    summary="Recupera dados para treinamento RL"
)
async def get_training_data(user_id: str) -> TrainingDataResponse:
    history = storage.get_user_history(user_id)
    
    if not history["days"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuário '{user_id}' não possui dados para treinamento"
        )
    
    training_data = storage.get_training_data(user_id)
    
    return TrainingDataResponse(
        user_id=user_id,
        total_samples=len(training_data),
        data=training_data
    )


@router.delete(
    "/historico/{user_id}",
    summary="Remove histórico do usuário",
    description="Remove todo o histórico de dados de um usuário."
)
async def delete_user_history(user_id: str):
    """Remove todo o histórico de um usuário."""
    deleted = storage.delete_user_history(user_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuário '{user_id}' não encontrado"
        )
    
    return {"success": True, "message": f"Histórico do usuário '{user_id}' removido"}


@router.get(
    "/usuarios",
    summary="Lista usuários com histórico",
    description="Retorna lista de todos os usuários que possuem dados."
)
async def list_users():
    """Lista todos os usuários com histórico."""
    users = storage.list_users()
    return {"users": users, "total": len(users)}


@router.post(
    "/treinar-incremental/{user_id}",
    summary="Treina modelo incrementalmente (sem salvar)",
    description="Re-treina o modelo PPO com dados históricos acumulados. Útil para aprendizado online."
)
async def train_incremental(user_id: str):
    """
    Treina o modelo RL incrementalmente com dados acumulados.
    
    Diferença vs /salvar-modelo:
    - Este endpoint: Treina mas NÃO salva em disco (mais rápido)
    - /salvar-modelo: Treina E salva em disco (mais lento)
    
    Use este durante simulação para aprendizado online.
    Use /salvar-modelo ao final para persistir.
    """
    from app.rl.trainer import RLTrainer
    
    # Carrega dados de treinamento
    training_data = storage.get_training_data(user_id)
    
    if not training_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuário '{user_id}' não possui dados para treinar"
        )
    
    # Cria e treina modelo
    trainer = RLTrainer(model_path=f"models/ppo_{user_id}")
    
    # 🔥 CARREGA MODELO EXISTENTE (se houver)
    loaded = trainer.load_model()
    
    trainer.add_day_data(training_data)
    
    # Treina com dados acumulados (10x timesteps para melhor convergência)
    training_result = trainer.train(total_timesteps=len(training_data) * 10)
    
    # 🔥 SALVA MODELO PARA PRÓXIMO DIA USAR!
    # Isso é CRÍTICO para aprendizado incremental
    model_path = trainer.save_model()
    
    return {
        "success": True,
        "message": f"Modelo treinado e salvo incrementalmente",
        "user_id": user_id,
        "training_samples": len(training_data),
        "model_loaded": loaded,
        "model_saved": model_path is not None,
        "training_result": training_result
    }


@router.post(
    "/salvar-modelo/{user_id}",
    summary="Salva o modelo RL treinado",
    description="Treina e salva o modelo PPO com os dados históricos do usuário."
)
async def save_trained_model(user_id: str):
    """
    Treina o modelo RL com dados históricos e salva em disco.
    
    IMPORTANTE: O modelo só é persistido quando este endpoint é chamado!
    Sem este passo, o modelo existe apenas em memória e é perdido.
    """
    from app.rl.trainer import RLTrainer
    
    # Carrega dados de treinamento
    training_data = storage.get_training_data(user_id)
    
    if not training_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuário '{user_id}' não possui dados para treinar"
        )
    
    # Cria e treina modelo
    trainer = RLTrainer(model_path=f"models/ppo_{user_id}")
    trainer.load_model()  # Carrega se existir
    trainer.add_day_data(training_data)
    
    # Treina com dados acumulados (10x mais timesteps para melhor convergência)
    training_result = trainer.train(total_timesteps=len(training_data) * 10)
    
    # 🔥 SALVA O MODELO EM DISCO!
    model_path = trainer.save_model()
    
    if not model_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao salvar modelo"
        )
    
    return {
        "success": True,
        "message": f"Modelo treinado e salvo com sucesso",
        "user_id": user_id,
        "model_path": model_path + ".zip",
        "training_samples": len(training_data),
        "training_result": training_result
    }
