"""
Simulação com Treino Incremental Dia a Dia + Múltiplas Epochs

FLUXO POR DIA:
1. API recebe JSON do dia
2. Gymnasium treina 100 epochs APENAS com esse dia (2.400 timesteps)
3. Modelo acumula conhecimento do dia anterior
4. Retorna predições para próximo dia

TOTAL: 30 dias × 2.400 timesteps/dia = 72.000 timesteps
"""

import json
import requests
import sys
import numpy as np
import random
from pathlib import Path
from typing import Dict, List
import shutil
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent))

from app.rl.environment import NotificationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

API_URL = "http://localhost:8000"
API_TIMEOUT = 30


class DailyProgressCallback(BaseCallback):
    """Callback simplificado para mostrar progresso por dia."""
    
    def __init__(self, day_num: int, check_freq: int = 500):
        super().__init__()
        self.day_num = day_num
        self.check_freq = check_freq
        self.rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                # Converte deque para lista e pega últimos 10
                recent_rewards = [ep_info["r"] for ep_info in list(self.model.ep_info_buffer)[-10:]]
                if recent_rewards:
                    mean_reward = np.mean(recent_rewards)
                    self.rewards.append(mean_reward)
                    print(f"  Step {self.n_calls:4d}/2400 | Reward: {mean_reward:6.1f}", end="\r")
        return True


def prepare_training_data_from_day(day_data: Dict) -> List[Dict]:
    """
    Converte dados de um dia para formato de treino Gymnasium.
    """
    training_data = []
    
    for hour_data in day_data['hours']:
        # Calcula FBM scores (M, A, T)
        mf = hour_data['motivation_factors']
        af = hour_data['ability_factors']
        tf = hour_data['trigger_factors']
        
        # MOTIVATION (0-4)
        valence = mf.get('valence', 0)
        last_activity = mf.get('last_activity_score', 0)
        
        if valence >= 3:
            m = 4
        elif valence >= 2:
            m = 3
        elif valence >= 1:
            m = 2
        elif last_activity >= 3:
            m = 1
        else:
            m = 0
        
        # ABILITY (0-4)
        cognitive_load = af.get('cognitive_load', 0)
        confidence = af.get('confidence_score', 0)
        
        if cognitive_load <= 2 and confidence >= 7:
            a = 4
        elif cognitive_load <= 3 and confidence >= 5:
            a = 3
        elif cognitive_load <= 5:
            a = 2
        elif cognitive_load <= 7:
            a = 1
        else:
            a = 0
        
        # TRIGGER (0-6)
        sleeping = tf.get('sleeping', False)
        arousal = tf.get('arousal', 0)
        location = tf.get('location', 'unknown')
        
        if sleeping:
            t = 0
        else:
            t = arousal
            if location == 'home':
                t += 1
        
        t = min(t, 6)
        fbm_score = m * a * t
        
        # Simula se usuário executaria (baseado em FBM)
        if fbm_score >= 80:
            action_performed = random.random() < 0.95
        elif fbm_score >= 60:
            action_performed = random.random() < 0.85
        elif fbm_score >= 40:
            action_performed = random.random() < 0.50
        elif fbm_score >= 20:
            action_performed = random.random() < 0.10
        elif fbm_score >= 10:
            action_performed = random.random() < 0.02
        else:
            action_performed = False
        
        # Monta sample de treino
        training_sample = {
            "observation": {
                "hour": hour_data["hour"],
                "day_period": hour_data["context"]["day_period"],
                "is_weekend": hour_data["context"]["is_weekend"],
                "motivation": m,
                "ability": a,
                "trigger": t,
                "sleeping": sleeping,
                "fbm_score": fbm_score
            },
            "action": 1 if action_performed else 0,  # Ação real (notificou?)
            "action_performed": action_performed
        }
        
        training_data.append(training_sample)
    
    return training_data


def train_day_with_gymnasium(
    day_training_data: List[Dict], 
    user_id: str, 
    day_num: int,
    epochs_per_day: int = 100
) -> PPO:
    """
    Treina modelo com dados de UM DIA por múltiplas epochs.
    
    Args:
        day_training_data: Dados do dia (24 samples)
        user_id: ID do usuário
        day_num: Número do dia
        epochs_per_day: Quantas vezes repetir o dia (default: 100)
    
    Returns:
        Modelo PPO atualizado
    """
    model_path = Path(f"models/ppo_{user_id}")
    
    # Cria environment com dados do dia
    env = NotificationEnv(training_data=day_training_data)
    
    # Carrega modelo anterior ou cria novo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if day_num > 1 and model_path.with_suffix('.zip').exists():
        # Continua treinamento do dia anterior
        model = PPO.load(str(model_path), env=env, device=device)
        print(f"  📥 Modelo do dia {day_num-1} carregado")
    else:
        # Cria novo modelo
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.001,
            n_steps=min(64, len(day_training_data)),
            batch_size=min(32, len(day_training_data)),
            n_epochs=10,
            gamma=0.99,
            verbose=0,
            device=device
        )
        print(f"  🆕 Novo modelo criado")
    
    # Treina múltiplas epochs com o dia
    total_timesteps = len(day_training_data) * epochs_per_day  # 24 × 100 = 2.400
    callback = DailyProgressCallback(day_num=day_num)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=False,  # Acumula timesteps
        progress_bar=False
    )
    
    # Salva modelo
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    
    return model


def get_predictions_for_next_day(model: PPO, next_day_data: Dict) -> List[int]:
    """
    Usa modelo treinado para prever notificações do próximo dia.
    """
    recommended_hours = []
    
    for hour_data in next_day_data['hours']:
        # Prepara observação
        mf = hour_data['motivation_factors']
        af = hour_data['ability_factors']
        tf = hour_data['trigger_factors']
        
        # Calcula M, A, T (mesmo cálculo do treino)
        valence = mf.get('valence', 0)
        if valence >= 3: m = 4
        elif valence >= 2: m = 3
        elif valence >= 1: m = 2
        else: m = 0
        
        cognitive_load = af.get('cognitive_load', 0)
        confidence = af.get('confidence_score', 0)
        if cognitive_load <= 2 and confidence >= 7: a = 4
        elif cognitive_load <= 3 and confidence >= 5: a = 3
        elif cognitive_load <= 5: a = 2
        else: a = 0
        
        sleeping = tf.get('sleeping', False)
        if sleeping:
            t = 0
        else:
            t = tf.get('arousal', 0)
            if tf.get('location') == 'home':
                t += 1
            t = min(t, 6)
        
        # Monta observação
        obs = np.array([
            hour_data["hour"],
            hour_data["context"]["day_period"],
            1 if hour_data["context"]["is_weekend"] else 0,
            m,
            a,
            t,
            1 if sleeping else 0,
            0  # notifications_today (resetado)
        ], dtype=np.float32)
        
        # Predição
        action, _ = model.predict(obs, deterministic=True)
        
        if action == 1 and not sleeping:
            recommended_hours.append(hour_data["hour"])
    
    return recommended_hours


def simulate_user_response(hour_data: Dict, was_notified: bool) -> bool:
    """Simula se usuário executa atividade."""
    mf = hour_data['motivation_factors']
    af = hour_data['ability_factors']
    tf = hour_data['trigger_factors']
    
    # Calcula FBM
    valence = mf.get('valence', 0)
    if valence >= 3: m = 4
    elif valence >= 2: m = 3
    elif valence >= 1: m = 2
    else: m = 0
    
    cognitive_load = af.get('cognitive_load', 0)
    confidence = af.get('confidence_score', 0)
    if cognitive_load <= 2 and confidence >= 7: a = 4
    elif cognitive_load <= 3 and confidence >= 5: a = 3
    elif cognitive_load <= 5: a = 2
    else: a = 0
    
    sleeping = tf.get('sleeping', False)
    if sleeping:
        t = 0
    else:
        t = tf.get('arousal', 0)
        if tf.get('location') == 'home':
            t += 1
        t = min(t, 6)
    
    fbm_score = m * a * t
    
    if not was_notified:
        # Ação espontânea
        if fbm_score >= 60:
            return random.random() < 0.45
        else:
            return random.random() < 0.05
    
    # Com notificação
    if fbm_score >= 80:
        return random.random() < 0.95
    elif fbm_score >= 60:
        return random.random() < 0.85
    elif fbm_score >= 40:
        return random.random() < 0.50
    elif fbm_score >= 20:
        return random.random() < 0.10
    elif fbm_score >= 10:
        return random.random() < 0.02
    else:
        return False


def run_incremental_simulation():
    """Executa simulação com treino incremental dia a dia."""
    
    CONFIG = {
        "num_days": 30,
        "epochs_per_day": 100,  # 100 epochs por dia
        "user_id": "user_fbm_variado",
        "seed": 42
    }
    
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    print("\n" + "="*80)
    print("🔄 SIMULAÇÃO COM TREINO INCREMENTAL DIA A DIA")
    print("="*80 + "\n")
    print("Configuração:")
    print(f"  📅 Dias: {CONFIG['num_days']}")
    print(f"  🔄 Epochs por dia: {CONFIG['epochs_per_day']}")
    print(f"  📊 Timesteps por dia: {24 * CONFIG['epochs_per_day']:,}")
    print(f"  🎯 Total timesteps: {CONFIG['num_days'] * 24 * CONFIG['epochs_per_day']:,}")
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
        return
    
    # Limpa modelo antigo
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
    print(f"📂 Perfil carregado: {len(days_data)} dias\n")
    
    # Métricas gerais
    total_metrics = {
        'vp': 0, 'vn': 0, 'fp': 0, 'fn': 0,
        'fbm_alto': {'vp': 0, 'vn': 0, 'notif': 0},
        'fbm_baixo': {'vp': 0, 'vn': 0, 'notif': 0}
    }
    
    daily_results = []
    model = None
    
    print("="*80)
    print("🏋️ TREINO INCREMENTAL")
    print("="*80 + "\n")
    
    # Loop por cada dia
    for day_num in range(1, CONFIG['num_days'] + 1):
        day_data = days_data[day_num - 1]
        
        print(f"📅 DIA {day_num}/{CONFIG['num_days']}")
        
        # 1. Prepara dados de treino do dia
        day_training_data = prepare_training_data_from_day(day_data)
        
        # 2. Treina modelo com múltiplas epochs deste dia
        print(f"  🏋️ Treinando {CONFIG['epochs_per_day']} epochs...")
        model = train_day_with_gymnasium(
            day_training_data,
            CONFIG['user_id'],
            day_num,
            epochs_per_day=CONFIG['epochs_per_day']
        )
        print(f"  ✅ Treino completo | Total: {day_num * 24 * CONFIG['epochs_per_day']:,} timesteps")
        
        # 3. Usa modelo para prever próximo dia (se não for último)
        if day_num < CONFIG['num_days']:
            next_day_data = days_data[day_num]
            predicted_hours = get_predictions_for_next_day(model, next_day_data)
            print(f"  📊 Predição dia {day_num+1}: {len(predicted_hours)} notificações em {predicted_hours[:5]}")
        
        # 4. Valida decisões do dia atual (simula)
        day_vp = day_vn = day_fp = day_fn = 0
        
        for hour_data in day_data['hours']:
            hour = hour_data['hour']
            
            # Decide se notifica (usando modelo se dia > 1)
            if day_num == 1:
                rl_notifies = random.random() < 0.3  # Exploração inicial
            else:
                # Usa modelo treinado até agora
                predicted_hours_today = get_predictions_for_next_day(model, day_data)
                rl_notifies = hour in predicted_hours_today
            
            # Simula resposta do usuário
            user_responded = simulate_user_response(hour_data, rl_notifies)
            
            # Classifica outcome
            if rl_notifies and user_responded:
                day_vp += 1
                total_metrics['vp'] += 1
            elif rl_notifies and not user_responded:
                day_vn += 1
                total_metrics['vn'] += 1
            elif not rl_notifies and user_responded:
                day_fp += 1
                total_metrics['fp'] += 1
            else:
                day_fn += 1
                total_metrics['fn'] += 1
        
        print(f"  📈 Resultado: VP={day_vp} VN={day_vn} FP={day_fp} FN={day_fn}")
        print()
        
        daily_results.append({
            'day': day_num,
            'vp': day_vp,
            'vn': day_vn,
            'fp': day_fp,
            'fn': day_fn
        })
    
    # Resultados finais
    print("="*80)
    print("✅ TREINO INCREMENTAL COMPLETO!")
    print("="*80 + "\n")
    
    total = total_metrics['vp'] + total_metrics['vn'] + total_metrics['fp'] + total_metrics['fn']
    precision = 100 * total_metrics['vp'] / (total_metrics['vp'] + total_metrics['vn']) if (total_metrics['vp'] + total_metrics['vn']) > 0 else 0
    recall = 100 * total_metrics['vp'] / (total_metrics['vp'] + total_metrics['fp']) if (total_metrics['vp'] + total_metrics['fp']) > 0 else 0
    
    print(f"📊 Métricas Finais:")
    print(f"   VP: {total_metrics['vp']}")
    print(f"   VN: {total_metrics['vn']}")
    print(f"   FP: {total_metrics['fp']}")
    print(f"   FN: {total_metrics['fn']}")
    print(f"   Precision: {precision:.1f}%")
    print(f"   Recall: {recall:.1f}%")
    print()
    print(f"💾 Modelo final salvo: models/ppo_{CONFIG['user_id']}.zip")
    print(f"📊 Total timesteps: {CONFIG['num_days'] * 24 * CONFIG['epochs_per_day']:,}\n")


if __name__ == "__main__":
    run_incremental_simulation()
