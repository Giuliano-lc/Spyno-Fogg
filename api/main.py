"""API de Recomendação de Notificações com RL + FBM."""

import warnings
import os
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import treino_router, previsao_router, threshold_router

app = FastAPI(
    title="API de Notificações RL + FBM",
    description="API REST para recomendação de notificações usando PPO e Fogg Behavior Model.",
    version="0.1.0",
    contact={"name": "TCC - FURG"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registra rotas
app.include_router(treino_router)
app.include_router(previsao_router)
app.include_router(threshold_router)


@app.get("/", tags=["Health"])
async def root():
    """Health check da API."""
    return {
        "status": "online",
        "service": "API de Notificações RL + FBM",
        "version": "0.1.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Verificação detalhada de saúde da API."""
    return {
        "status": "healthy",
        "components": {
            "api": "ok",
            "storage": "ok"
        }
    }
