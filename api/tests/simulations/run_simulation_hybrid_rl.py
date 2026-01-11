"""
Simulação HÍBRIDA: API para dados + Gymnasium para treinamento intensivo

FLUXO:
1. Envia 30 dias via API (simula produção, valida schemas)
2. Baixa dados de treino via API
3. Treina localmente com Gymnasium/PPO (72.000 timesteps = 100 epochs)
4. Salva modelo
5. Valida decisões via API com modelo treinado
"""

import json
import requests
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
import shutil

sys.path.append(str(Path(__file__).parent.parent))

from app.rl.environment import NotificationEnv
from app.rl.trainer import RLTrainer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

API_URL = "http://localhost:8000"
API_TIMEOUT = 30


class ProgressCallback(BaseCallback):
    """Callback para mostrar progresso do treino."""
    
    def __init__(self, check_freq: int = 1000):
        super().__init__()
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Obtém métricas do environment
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                
                print(f"\n📊 Step {self.n_calls:,}/{self.locals.get('total_timesteps', 0):,}")
                print(f"   Mean Reward: {mean_reward:.2f}")
                print(f"   Mean Episode Length: {mean_length:.1f}")
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"   🎯 Novo melhor reward!")
        
        return True


def send_data_via_api(user_id: str, days_data: List[Dict]) -> bool:
    """
    Envia dados via API (simula produção).
    
    Returns:
        True se sucesso
    """
    print("📤 ETAPA 1: Enviando dados via API (simula produção)\n")
    
    for i, day_data in enumerate(days_data, 1):
        try:
            response = requests.post(
                f"{API_URL}/treino",
                json=day_data,
                headers={"Content-Type": "application/json"},
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                print(f"✅ Dia {i:2d}/{len(days_data)} enviado")
            else:
                print(f"❌ Erro dia {i}: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao enviar dia {i}: {e}")
            return False
    
    print(f"\n✅ Todos os {len(days_data)} dias enviados via API!\n")
    return True


def get_training_data_from_api(user_id: str) -> List[Dict]:
    """
    Baixa dados de treino formatados da API.
    
    Returns:
        Lista de training samples
    """
    print("📥 ETAPA 2: Baixando dados de treino da API\n")
    
    try:
        response = requests.get(
            f"{API_URL}/treino/dados-treinamento/{user_id}",
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            training_data = data.get("training_data", [])
            print(f"✅ {len(training_data)} samples baixados da API\n")
            return training_data
        else:
            print(f"❌ Erro ao baixar dados: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return []


def train_with_gymnasium(training_data: List[Dict], user_id: str, total_timesteps: int = 72000) -> PPO:
    """
    Treina modelo localmente com Gymnasium/PPO.
    
    Args:
        training_data: Dados de treino
        user_id: ID do usuário
        total_timesteps: Total de timesteps (72k = 100 epochs × 720 steps)
    
    Returns:
        Modelo PPO treinado
    """
    print("🏋️ ETAPA 3: Treinamento Intensivo com Gymnasium/PPO\n")
    print(f"Configuração:")
    print(f"  📊 Samples base: {len(training_data)}")
    print(f"  🔄 Total timesteps: {total_timesteps:,}")
    print(f"  📈 Epochs estimadas: {total_timesteps // len(training_data)}")
    print(f"  🎯 Device: {'CUDA' if PPO else 'CPU'}\n")
    
    # Cria environment
    env = NotificationEnv(training_data=training_data)
    
    # Cria modelo PPO
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print(f"🚀 GPU detectada! Treinando em: {torch.cuda.get_device_name(0)}\n")
    else:
        print(f"💻 Treinando em CPU\n")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.001,  # Learning rate mais alto para convergir mais rápido
        n_steps=min(2048, len(training_data)),
        batch_size=min(64, len(training_data)),
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        device=device
    )
    
    # Treina com callback de progresso
    callback = ProgressCallback(check_freq=1000)
    
    print("🔥 Iniciando treino...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    print(f"\n✅ Treino completo! {total_timesteps:,} timesteps processados\n")
    
    # Salva modelo
    model_path = Path(f"models/ppo_{user_id}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"💾 Modelo salvo: {model_path}.zip\n")
    
    return model


def validate_with_api(user_id: str, days_data: List[Dict]):
    """
    Valida modelo treinado simulando decisões via API.
    """
    print("🎯 ETAPA 4: Validação via API\n")
    
    metrics = {
        'vp': 0, 'vn': 0, 'fp': 0, 'fn': 0,
        'fbm_alto': {'vp': 0, 'vn': 0, 'notifications': 0},
        'fbm_baixo': {'vp': 0, 'vn': 0, 'notifications': 0}
    }
    
    for day_idx, day_data in enumerate(days_data[:5], 1):  # Valida primeiros 5 dias
        # Chama API para previsões
        try:
            hours_data = []
            for hour_data in day_data['hours']:
                # Monta observação
                obs = {
                    "hour": hour_data["hour"],
                    "day_period": hour_data["context"]["day_period"],
                    "is_weekend": hour_data["context"]["is_weekend"],
                    "sleeping": hour_data["trigger_factors"]["sleeping"],
                    "motivation": 0,  # Calculado pela API
                    "ability": 0,
                    "trigger": 0,
                    "fbm_score": 0
                }
                hours_data.append(obs)
            
            response = requests.post(
                f"{API_URL}/previsao/{user_id}/custom",
                json={"hours_data": hours_data},
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                recommended_hours = result.get("recommended_hours", [])
                print(f"Dia {day_idx}: {len(recommended_hours)} notificações recomendadas")
                
        except Exception as e:
            print(f"Erro validação dia {day_idx}: {e}")
    
    print("\n✅ Validação completa!\n")


def run_hybrid_simulation():
    """Executa simulação híbrida completa."""
    
    CONFIG = {
        "num_days": 30,
        "total_timesteps": 72000,  # 100 epochs × 720 steps
        "user_id": "user_fbm_variado",
        "seed": 42
    }
    
    print("\n" + "="*80)
    print("🔄 SIMULAÇÃO HÍBRIDA: API + Gymnasium")
    print("="*80 + "\n")
    print("Configuração:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Verifica API
    print("🔍 Verificando API...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API está rodando!\n")
        else:
            print(f"❌ API retornou status {response.status_code}")
            return
    except:
        print("❌ API não está acessível!")
        print("   Execute: python -m uvicorn main:app --port 8000\n")
        return
    
    # Limpa dados anteriores
    print("🧹 Limpando dados anteriores...")
    try:
        requests.delete(f"{API_URL}/treino/historico/{CONFIG['user_id']}", timeout=5)
        print("✅ Histórico limpo!\n")
    except:
        pass
    
    # Deleta modelo antigo
    print("🧹 Deletando modelo RL antigo...")
    model_path = Path(f"models/ppo_{CONFIG['user_id']}")
    if model_path.exists():
        shutil.rmtree(model_path)
        print("✅ Modelo antigo deletado!\n")
    else:
        print("✅ Nenhum modelo antigo encontrado\n")
    
    # Carrega perfil
    base_dir = Path(__file__).parent.parent
    profile_path = base_dir / "data" / "simulation" / "NovoPerfil" / "user_fbm_variado.json"
    
    if not profile_path.exists():
        print(f"❌ Perfil não encontrado: {profile_path}")
        return
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile_data = json.load(f)
    
    days_data = profile_data["days"]
    
    # Garante campos necessários
    for day in days_data:
        day['user_id'] = CONFIG['user_id']
        day.setdefault('timezone', 'America/Sao_Paulo')
        day.pop('user_profile', None)
    
    print(f"📂 Perfil carregado: {len(days_data)} dias\n")
    
    # ETAPA 1: Envia via API
    if not send_data_via_api(CONFIG['user_id'], days_data):
        print("❌ Falha ao enviar dados via API")
        return
    
    # ETAPA 2: Baixa dados de treino
    training_data = get_training_data_from_api(CONFIG['user_id'])
    if not training_data:
        print("❌ Falha ao baixar dados de treino")
        return
    
    # ETAPA 3: Treina com Gymnasium
    model = train_with_gymnasium(
        training_data, 
        CONFIG['user_id'],
        total_timesteps=CONFIG['total_timesteps']
    )
    
    # ETAPA 4: Valida via API
    validate_with_api(CONFIG['user_id'], days_data)
    
    print("="*80)
    print("✅ SIMULAÇÃO HÍBRIDA COMPLETA!")
    print("="*80)
    print(f"\n📊 Resumo:")
    print(f"  - Dados enviados via API: {len(days_data)} dias")
    print(f"  - Samples de treino: {len(training_data)}")
    print(f"  - Timesteps treinados: {CONFIG['total_timesteps']:,}")
    print(f"  - Epochs: ~{CONFIG['total_timesteps'] // len(training_data)}")
    print(f"  - Modelo salvo: models/ppo_{CONFIG['user_id']}.zip\n")


if __name__ == "__main__":
    run_hybrid_simulation()
