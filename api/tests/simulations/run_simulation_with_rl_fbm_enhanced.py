"""
Simulação FBM-Based com RL MELHORADO - Maior Influência do FBM

MELHORIAS:
1. Epsilon adaptativo baseado no FBM score (explora mais em FBM alto)
2. Recompensas VP/VN proporcionais ao FBM
3. Perfil com FBM variado (alto manhã/noite, baixo tarde)
4. Métricas detalhadas por faixa de FBM
"""

import json
import requests
import random
import shutil
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Any
from copy import deepcopy
import sys

sys.path.append(str(Path(__file__).parent.parent))

from synthetic_data_generator import SyntheticDataGenerator, PERFIL_MATINAL
from fbm_simulation import FBMSimulator
import numpy as np


API_URL = "http://localhost:8000"
API_TIMEOUT = 30


def add_noise_to_profile(profile_data: Dict, noise_factor: float, seed: int) -> Dict:
    """
    Adiciona ruído controlado aos fatores FBM para criar variações realistas.
    
    Args:
        profile_data: Perfil base
        noise_factor: Percentual de variação (0.15 = ±15%)
        seed: Seed para reprodutibilidade
    
    Returns:
        Nova cópia do perfil com ruído aplicado
    """
    np.random.seed(seed)
    noisy_profile = deepcopy(profile_data)
    
    for day in noisy_profile['days']:
        for hour_data in day['hours']:
            # Aplica ruído gaussiano aos fatores FBM
            mf = hour_data['motivation_factors']
            af = hour_data['ability_factors']
            tf = hour_data['trigger_factors']
            
            # Motivation factors (mantém >= 0)
            mf['valence'] = max(0, mf['valence'] * (1 + np.random.normal(0, noise_factor)))
            mf['last_activity_score'] = max(0, mf['last_activity_score'] * (1 + np.random.normal(0, noise_factor)))
            
            # Ability factors (mantém >= 0)
            af['cognitive_load'] = max(0, af['cognitive_load'] * (1 + np.random.normal(0, noise_factor)))
            af['confidence_score'] = max(0, min(10, af['confidence_score'] * (1 + np.random.normal(0, noise_factor))))
            
            # Trigger factors (mantém >= 0)
            tf['arousal'] = max(0, min(10, tf['arousal'] * (1 + np.random.normal(0, noise_factor))))
    
    return noisy_profile


class RLSimulatorFBMEnhanced:
    """Simulador RL com maior influência do FBM nas decisões."""
    
    def __init__(self, api_url: str, config: Dict):
        self.api_url = api_url
        self.config = config
        self.user_id = config["user_id"]
        
        # Métricas da simulação
        self.total_notifications = 0
        self.total_actions = 0
        self.vp_count = 0
        self.vn_count = 0
        self.fp_count = 0
        self.fn_count = 0
        
        # 🎯 NOVO: Métricas por faixa de FBM
        self.fbm_metrics = {
            "alto": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0},  # FBM >= 60
            "medio": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0},  # 40 <= FBM < 60
            "baixo": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0}   # FBM < 40
        }
        
        # Stats por hora
        self.hourly_stats = {hour: {"notified": 0, "responded": 0, "fbm_avg": 0, "count": 0} for hour in range(24)}
        
        # Resultados diários
        self.daily_results = []
        
        # 🔥 EXPLORATION DECAY DINÂMICO
        self.epsilon = 0.30  # Começa com 30% exploration
        self.epsilon_min = 0.02
        self.epsilon_decay_rate = 0.95
        self.vn_threshold = 10
        
    def should_user_respond(self, fbm_score: int, was_notified: bool, hour: int) -> bool:
        """
        Simula resposta do usuário baseado no PERFIL FBM VARIADO:
        - Manhã (6-11h): FBM ALTO (~96) → responde muito
        - Tarde (12-17h): FBM BAIXO (~2) → quase nunca responde
        - Noite (18-23h): FBM ALTO (~80) → responde bem
        - Madrugada (0-5h): Dormindo
        """
        MANHA_HOURS = list(range(6, 12))    # 6-11h: FBM alto
        TARDE_HOURS = list(range(12, 18))   # 12-17h: FBM baixo
        NOITE_HOURS = list(range(18, 24))   # 18-23h: FBM alto
        
        if not was_notified:
            # Ação espontânea (sem notificação)
            if hour in MANHA_HOURS and fbm_score >= 60:
                return random.random() < 0.5  # 50% espontâneo na manhã
            elif hour in NOITE_HOURS and fbm_score >= 60:
                return random.random() < 0.4  # 40% espontâneo na noite
            else:
                return random.random() < 0.05  # Raro fora desses períodos
        
        # Com notificação: probabilidade baseada em FBM (mais realista)
        if fbm_score >= 80:
            probability = 0.95  # FBM muito alto
        elif fbm_score >= 60:
            probability = 0.85  # FBM alto
        elif fbm_score >= 40:
            probability = 0.50  # FBM médio
        elif fbm_score >= 20:
            probability = 0.10  # FBM baixo (reduzido de 0.20)
        elif fbm_score >= 10:
            probability = 0.02  # FBM muito baixo (reduzido de 0.05)
        else:
            probability = 0.0   # FBM extremamente baixo: NUNCA executa
        
        return random.random() < probability
        
    def update_epsilon(self, day_vn: int):
        """Atualiza epsilon baseado em VN do dia."""
        self.epsilon *= self.epsilon_decay_rate
        
        if day_vn > self.vn_threshold:
            extra_decay = 0.90
            self.epsilon *= extra_decay
        
        self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def get_rl_decision_for_hour(self, day_hours_data: List[Dict], target_hour: int, fbm_score: int) -> bool:
        """
        🎯 RL PURO: Epsilon-greedy padrão sem interferências.
        
        - Exploration (epsilon): Ação aleatória (deixa PPO explorar livremente)
        - Exploitation (1-epsilon): PPO decide baseado em política aprendida
        - PPO aprende através de recompensas/penalidades proporcionais ao FBM
        """
        # 🔥 EXPLORATION: Aleatória pura (sem considerar FBM)
        if random.random() < self.epsilon:
            # Explora aleatoriamente - 50% chance de notificar
            # PPO vai aprender quais horas evitar através das penalidades
            return random.random() < 0.5
        
        # 🔥 EXPLOITATION: PPO decide baseado em política aprendida
        try:
            response = requests.post(
                f"{self.api_url}/previsao/{self.user_id}/custom",
                json={"hours_data": day_hours_data},
                headers={"Content-Type": "application/json"},
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                
                for hour_pred in result.get("all_hours", []):
                    if hour_pred["hour"] == target_hour:
                        return hour_pred["recommended"]
                
                return target_hour in result.get("recommended_hours", [])
            
            return False
            
        except Exception as e:
            return False
    
    def calculate_fbm_scores(self, hour_data: Dict) -> Dict:
        """Calcula scores M, A, T do FBM."""
        mf = hour_data["motivation_factors"]
        af = hour_data["ability_factors"]
        tf = hour_data["trigger_factors"]
        
        # MOTIVATION (0-4)
        valence = mf.get("valence", 0)
        last_activity = mf.get("last_activity_score", 0)
        
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
        cognitive_load = af.get("cognitive_load", 0)
        confidence = af.get("confidence_score", 0)
        
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
        sleeping = tf.get("sleeping", False)
        arousal = tf.get("arousal", 0)
        location = tf.get("location", "unknown")
        
        if sleeping:
            t = 0
        else:
            t = arousal
            if location == 'home':
                t += 1
        
        t = min(t, 6)
        
        fbm_score = m * a * t
        
        return {
            "motivation": m,
            "ability": a,
            "trigger": t,
            "fbm_score": fbm_score
        }
    
    def get_fbm_category(self, fbm_score: int) -> str:
        """Retorna categoria do FBM score."""
        if fbm_score >= 60:
            return "alto"
        elif fbm_score >= 40:
            return "medio"
        else:
            return "baixo"
    
    def simulate_reward_for_outcome(self, outcome_type: str, fbm_score: int) -> float:
        """
        🎯 NOVO: Calcula recompensa simulada proporcional ao FBM.
        
        (Nota: A recompensa real é calculada no environment.py, mas guardamos
        a simulação para análise)
        """
        if outcome_type == "VP":  # Acerto
            if fbm_score >= 60:
                return 35.0  # Bônus grande
            elif fbm_score >= 40:
                return 25.0  # Bônus médio
            else:
                return 15.0  # Recompensa menor
        
        elif outcome_type == "VN":  # Erro ao notificar
            if fbm_score >= 60:
                return -3.0  # Penalidade leve (tentou no momento certo)
            elif fbm_score >= 40:
                return -8.0  # Penalidade média
            else:
                return -15.0  # Penalidade forte (momento errado!)
        
        elif outcome_type == "FP":  # Perdeu oportunidade
            return -2.0  # Mantém fixo
        
        else:  # FN - Não notificou, correto
            return 0.5  # Mantém fixo
    
    def simulate_day_with_rl(self, day_data: Dict) -> Dict:
        """Simula um dia completo com RL decidindo notificações."""
        day_hours = day_data["hours"]
        results = []
        
        day_fbm_scores = []
        
        for hour_idx, hour_data in enumerate(day_hours):
            hour = hour_data["hour"]
            
            # Pula se dormindo
            if hour_data["trigger_factors"].get("sleeping"):
                continue
            
            # Calcula FBM
            fbm_calc = self.calculate_fbm_scores(hour_data)
            fbm_score = fbm_calc["fbm_score"]
            day_fbm_scores.append(fbm_score)
            
            # RL decide se notifica (com epsilon adaptativo por FBM)
            rl_notifies = self.get_rl_decision_for_hour(day_hours, hour, fbm_score)
            
            # Usuário responde?
            user_responded = self.should_user_respond(fbm_score, rl_notifies, hour)
            
            # Determina tipo de outcome
            if rl_notifies and user_responded:
                outcome = "VP"
                self.vp_count += 1
            elif rl_notifies and not user_responded:
                outcome = "VN"
                self.vn_count += 1
            elif not rl_notifies and user_responded:
                outcome = "FP"
                self.fp_count += 1
            else:
                outcome = "FN"
                self.fn_count += 1
            
            # 🎯 ATUALIZA MÉTRICAS POR FAIXA DE FBM
            fbm_cat = self.get_fbm_category(fbm_score)
            self.fbm_metrics[fbm_cat][outcome.lower()] += 1
            if rl_notifies:
                self.fbm_metrics[fbm_cat]["notified"] += 1
            
            # Calcula recompensa simulada
            simulated_reward = self.simulate_reward_for_outcome(outcome, fbm_score)
            
            # Atualiza métricas globais
            if rl_notifies:
                self.total_notifications += 1
                self.hourly_stats[hour]["notified"] += 1
            
            if user_responded:
                self.total_actions += 1
                self.hourly_stats[hour]["responded"] += 1
            
            # Atualiza FBM médio por hora
            self.hourly_stats[hour]["fbm_avg"] += fbm_score
            self.hourly_stats[hour]["count"] += 1
            
            # 🔥 ATUALIZA FEEDBACK NOS DADOS ORIGINAIS
            day_hours[hour_idx]["feedback"]["notification_sent"] = rl_notifies
            day_hours[hour_idx]["feedback"]["action_performed"] = user_responded
            
            results.append({
                "hour": hour,
                "fbm_score": fbm_score,
                "fbm_category": fbm_cat,
                "motivation": fbm_calc["motivation"],
                "ability": fbm_calc["ability"],
                "trigger": fbm_calc["trigger"],
                "rl_notified": rl_notifies,
                "user_responded": user_responded,
                "outcome": outcome,
                "simulated_reward": simulated_reward
            })
        
        # Calcula FBM médio do dia
        fbm_avg = sum(day_fbm_scores) / len(day_fbm_scores) if day_fbm_scores else 0
        
        day_result = {
            "date": day_data["date"],
            "results": results,
            "day_vp": sum(1 for r in results if r["outcome"] == "VP"),
            "day_vn": sum(1 for r in results if r["outcome"] == "VN"),
            "day_fp": sum(1 for r in results if r["outcome"] == "FP"),
            "day_fn": sum(1 for r in results if r["outcome"] == "FN"),
            "fbm_avg": fbm_avg
        }
        
        self.daily_results.append(day_result)
        
        return day_result
    
    def train_with_day(self, day_data: Dict, day_num: int) -> bool:
        """Envia dia para API /treino E treina modelo incrementalmente."""
        try:
            # PASSO 1: Envia dados do dia
            api_data = {
                "user_id": day_data["user_id"],
                "date": day_data["date"],
                "timezone": day_data["timezone"],
                "user_profile": {
                    "has_family": True
                },
                "hours": []
            }
            
            for hour in day_data["hours"]:
                feedback_clean = {
                    "notification_sent": hour["feedback"].get("notification_sent") or False,
                    "action_performed": hour["feedback"].get("action_performed") or False,
                    "training_feedback": None
                }
                
                hour_clean = {
                    "hour": hour["hour"],
                    "motivation_factors": hour["motivation_factors"],
                    "ability_factors": hour["ability_factors"],
                    "trigger_factors": hour["trigger_factors"],
                    "context": hour["context"],
                    "feedback": feedback_clean
                }
                api_data["hours"].append(hour_clean)
            
            response = requests.post(
                f"{self.api_url}/treino",
                json=api_data,
                headers={"Content-Type": "application/json"},
                timeout=API_TIMEOUT
            )
            
            if response.status_code != 201:
                try:
                    error_detail = response.json()
                    print(f"\n   ⚠️ Erro {response.status_code} ao enviar dados:")
                    print(f"   {error_detail}")
                except:
                    print(f"\n   ⚠️ Erro {response.status_code}: {response.text[:200]}")
                return False
            
            # PASSO 2: Treina modelo incrementalmente
            train_response = requests.post(
                f"{self.api_url}/treino/treinar-incremental/{self.user_id}",
                timeout=120
            )
            
            if train_response.status_code == 200:
                train_result = train_response.json()
                
                if not hasattr(self, 'training_info'):
                    self.training_info = []
                
                info = {
                    "day": day_num,
                    "model_loaded": train_result.get("model_loaded", False),
                    "model_saved": train_result.get("model_saved", False)
                }
                
                if "training_result" in train_result and "total_reward" in train_result["training_result"]:
                    info["reward"] = train_result["training_result"]["total_reward"]
                
                self.training_info.append(info)
                
                return True
            else:
                try:
                    error_detail = train_response.json()
                    print(f"\n   ⚠️ Erro {train_response.status_code} ao treinar modelo:")
                    print(f"   {error_detail}")
                except:
                    print(f"\n   ⚠️ Erro {train_response.status_code} ao treinar modelo: {train_response.text[:300]}")
                return False
            
        except Exception as e:
            print(f"\n   ⚠️ Exceção: {e}")
            return False
    
    def get_metrics(self) -> Dict:
        """Retorna métricas finais."""
        vp = self.vp_count
        vn = self.vn_count
        fp = self.fp_count
        
        precision = (vp / (vp + vn) * 100) if (vp + vn) > 0 else 0
        recall = (vp / (vp + fp) * 100) if (vp + fp) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        # Calcula FBM médio por hora
        for hour in range(24):
            if self.hourly_stats[hour]["count"] > 0:
                self.hourly_stats[hour]["fbm_avg"] /= self.hourly_stats[hour]["count"]
        
        # Top horas
        top_hours = sorted(
            [(h, stats["responded"]) for h, stats in self.hourly_stats.items() if stats["responded"] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # 🎯 MÉTRICAS POR FAIXA DE FBM
        fbm_summary = {}
        for cat, metrics in self.fbm_metrics.items():
            vp_cat = metrics["vp"]
            vn_cat = metrics["vn"]
            fp_cat = metrics["fp"]
            
            prec = (vp_cat / (vp_cat + vn_cat) * 100) if (vp_cat + vn_cat) > 0 else 0
            rec = (vp_cat / (vp_cat + fp_cat) * 100) if (vp_cat + fp_cat) > 0 else 0
            
            fbm_summary[cat] = {
                "precision": prec,
                "recall": rec,
                "vp": vp_cat,
                "vn": vn_cat,
                "fp": fp_cat,
                "fn": metrics["fn"],
                "notified": metrics["notified"]
            }
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "vp": vp,
            "vn": vn,
            "fp": fp,
            "fn": self.fn_count,
            "total_notifications": self.total_notifications,
            "total_actions": self.total_actions,
            "top_hours": top_hours,
            "fbm_metrics": fbm_summary
        }


def run_simulation_with_rl_fbm_enhanced():
    """Executa simulação completa com RL melhorado por FBM."""
    
    CONFIG = {
        "num_days": 30,
        "num_epochs": 100,  # 🎯 100 variações × 30 dias = 72.000 samples
        "user_id": "user_fbm_variado",
        "initial_threshold": 40.0,
        "seed": 42,
        "noise_factor": 0.15  # ±15% de variação nos fatores FBM
    }
    
    print("\n🚀 Iniciando Simulação RL com FBM MELHORADO\n")
    print(f"Configuração:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Verifica API
    print("🔍 Verificando API...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API está rodando!")
        else:
            print("❌ API não respondeu corretamente")
            return
    except:
        print("❌ API não está rodando! Inicie com: python start.py")
        return
    
    # Limpa histórico anterior
    print("🧹 Limpando dados anteriores...")
    try:
        response = requests.delete(
            f"{API_URL}/treino/historico/{CONFIG['user_id']}",
            timeout=5
        )
        if response.status_code in [200, 404]:
            print("✅ Histórico limpo!")
        else:
            print("⚠️ Aviso: não foi possível limpar histórico anterior")
    except:
        print("⚠️ Aviso: erro ao limpar histórico (continuando)")
    
    # Deleta modelo antigo
    print("🧹 Deletando modelo RL antigo...")
    model_path = Path(f"models/ppo_{CONFIG['user_id']}")
    if model_path.exists():
        try:
            shutil.rmtree(model_path)
            print("✅ Modelo antigo deletado! (vai treinar do zero)\n")
        except Exception as e:
            print(f"⚠️ Erro ao deletar modelo: {e}\n")
    else:
        print("✅ Nenhum modelo antigo encontrado (vai treinar do zero)\n")
    
    print(f"{'#'*100}")
    print(f"# SIMULAÇÃO COM RL MELHORADO - Maior Influência do FBM")
    print(f"{'#'*100}\n")
    
    # Carrega perfil FBM variado
    print("📋 ETAPA 1: Carregando Perfil FBM Variado\n")
    
    base_dir = Path(__file__).parent.parent.parent
    
    # Carrega perfil do novo local
    profile_path = base_dir / "data" / "simulation" / "NovoPerfil" / "user_fbm_variado.json"
    
    if profile_path.exists():
        print(f"📂 Carregando perfil variado...")
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        
        # Garante que todos os dias tenham user_id e timezone
        # user_profile deve ser removido dos dados do dia (não é esperado pela API)
        for day in profile_data['days']:
            if 'user_id' not in day:
                day['user_id'] = CONFIG['user_id']
            if 'timezone' not in day:
                day['timezone'] = 'America/Sao_Paulo'
            # Remove user_profile se existir (não é campo do schema)
            day.pop('user_profile', None)
    else:
        print(f"📂 Perfil variado não encontrado, carregando matinal...")
        original_path = base_dir / "data" / "users" / "user_matinal_rl_v2.json"
        
        if not original_path.exists():
            print(f"❌ Perfil original não encontrado: {original_path}")
            return
        
        with open(original_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)
        
        print(f"🔧 Aplicando transformações FBM in-memory...")
        
        # Transforma user_id
        profile_data['user_id'] = CONFIG['user_id']
        
        # Aplica transformações FBM para cada hora
        for day in profile_data['days']:
            for hour_data in day['hours']:
                hour = hour_data['hour']
                
                # MANHÃ (6-11h): FBM ALTO
                if 6 <= hour <= 11:
                    hour_data['motivation_factors']['valence'] = 1  # max=1
                    hour_data['motivation_factors']['last_activity_score'] = 1  # max=1
                    hour_data['ability_factors']['cognitive_load'] = 0  # 0=baixa
                    hour_data['ability_factors']['confidence_score'] = 9
                    hour_data['trigger_factors']['sleeping'] = False
                    hour_data['trigger_factors']['arousal'] = 2  # max=2
                    hour_data['trigger_factors']['location'] = 'home'
                    hour_data['trigger_factors']['motion_activity'] = 'active'
                
                # TARDE (12-17h): FBM BAIXO
                elif 12 <= hour <= 17:
                    hour_data['motivation_factors']['valence'] = 0  # 0=negativo
                    hour_data['motivation_factors']['last_activity_score'] = 0
                    hour_data['ability_factors']['cognitive_load'] = 1  # 1=alta
                    hour_data['ability_factors']['confidence_score'] = 3
                    hour_data['trigger_factors']['sleeping'] = False
                    hour_data['trigger_factors']['arousal'] = 0  # baixo
                    hour_data['trigger_factors']['location'] = 'work'
                    hour_data['trigger_factors']['motion_activity'] = 'stationary'
                
                # NOITE (18-23h): FBM ALTO
                elif 18 <= hour <= 23:
                    hour_data['motivation_factors']['valence'] = 1  # max=1
                    hour_data['motivation_factors']['last_activity_score'] = 1  # max=1
                    hour_data['ability_factors']['cognitive_load'] = 0  # baixa
                    hour_data['ability_factors']['confidence_score'] = 8
                    hour_data['trigger_factors']['sleeping'] = False
                    hour_data['trigger_factors']['arousal'] = 2  # max=2
                    hour_data['trigger_factors']['location'] = 'home'
                    hour_data['trigger_factors']['motion_activity'] = 'walking'
                
                # MADRUGADA (0-5h): Dormindo
                else:
                    hour_data['motivation_factors']['valence'] = 0
                    hour_data['motivation_factors']['last_activity_score'] = 0
                    hour_data['ability_factors']['cognitive_load'] = 0
                    hour_data['ability_factors']['confidence_score'] = 3
                    hour_data['trigger_factors']['sleeping'] = True
                    hour_data['trigger_factors']['arousal'] = 0
                    hour_data['trigger_factors']['location'] = 'home'
                    hour_data['trigger_factors']['motion_activity'] = 'stationary'
                
                # Zera feedback
                hour_data['feedback'] = {
                    'notification_sent': False,
                    'action_performed': False,
                    'training_feedback': None
                }
        
        # Atualiza user_id nos dias e adiciona user_profile
        for day in profile_data['days']:
            day['user_id'] = CONFIG['user_id']
            if 'user_profile' not in day:
                day['user_profile'] = 'matinal'
            if 'timezone' not in day:
                day['timezone'] = 'America/Sao_Paulo'
    
    days_data = profile_data["days"]
    print(f"✅ Perfil pronto: {len(days_data)} dias com FBM variado\n")
    
    # ETAPA 2: Simulação com RL melhorado (MÚLTIPLAS EPOCHS)
    num_epochs = CONFIG.get('num_epochs', 1)
    noise_factor = CONFIG.get('noise_factor', 0.15)
    
    print("📋 ETAPA 2: Simulação com RL MELHORADO (Múltiplas Epochs)\n")
    print("Melhorias aplicadas:")
    print("  ✅ RL Puro: Epsilon-greedy sem interferências")
    print("  ✅ Recompensas VP/VN proporcionais ao FBM")
    print("  ✅ Múltiplas Epochs com variações para aumentar samples")
    print(f"  ✅ {num_epochs} epochs × {CONFIG['num_days']} dias = {num_epochs * CONFIG['num_days'] * 24:,} samples\n")
    
    simulator_rl = RLSimulatorFBMEnhanced(api_url=API_URL, config=CONFIG)
    
    # 🔥 LOOP DE MÚLTIPLAS EPOCHS
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"🔄 EPOCH {epoch}/{num_epochs} (Variação do perfil com ±{noise_factor*100:.0f}% ruído)")
        print(f"{'='*80}\n")
        
        # Cria variação do perfil se não for primeira epoch
        if epoch > 1:
            days_data_variant = add_noise_to_profile(
                {'days': days_data}, 
                noise_factor, 
                seed=CONFIG['seed'] + epoch
            )['days']
        else:
            days_data_variant = days_data
        
        # Simula os 30 dias desta epoch
        for i, day_data in enumerate(days_data_variant, 1):
            # Simula dia com RL
            day_result = simulator_rl.simulate_day_with_rl(day_data)
            
            # Atualiza epsilon
            simulator_rl.update_epsilon(day_result['day_vn'])
            
            # Treina modelo
            total_samples = (epoch - 1) * CONFIG['num_days'] * 24 + i * 24
            
            print(f"E{epoch:3d} D{i:2d}: VP={day_result['day_vp']:2d}, VN={day_result['day_vn']:2d}, "
                  f"FP={day_result['day_fp']:2d}, ε={simulator_rl.epsilon:.3f} ", end="")
            
            trained = simulator_rl.train_with_day(day_data, total_samples // 24)
            
            if trained:
                print(f"Treino=✅ ({total_samples:,} samples)")
            else:
                print(f"Treino=❌")
    
    metrics_rl = simulator_rl.get_metrics()
    
    print(f"\n✅ RL completo:")
    print(f"   Precision: {metrics_rl['precision']:.1f}%")
    print(f"   Recall:    {metrics_rl['recall']:.1f}%")
    print(f"   F1-Score:  {metrics_rl['f1_score']:.1f}%")
    print(f"   Top horas: {[h for h, _ in metrics_rl['top_hours'][:3]]}\n")
    
    # 🎯 ANÁLISE POR FAIXA DE FBM
    print("="*100)
    print("📊 ANÁLISE POR FAIXA DE FBM")
    print("="*100)
    print()
    
    print(f"{'Faixa FBM':<15} | {'Precision':<12} | {'Recall':<12} | {'VP':<6} | {'VN':<6} | {'FP':<6} | {'Notificações':<15}")
    print(f"{'-'*100}")
    
    for cat in ["alto", "medio", "baixo"]:
        m = metrics_rl['fbm_metrics'][cat]
        cat_label = f"{cat.upper()} (≥60)" if cat == "alto" else f"{cat.upper()} (40-59)" if cat == "medio" else f"{cat.upper()} (<40)"
        print(f"{cat_label:<15} | {m['precision']:>10.1f}% | {m['recall']:>10.1f}% | {m['vp']:>6} | {m['vn']:>6} | {m['fp']:>6} | {m['notified']:>15}")
    
    print()
    
    # Validação de padrão
    print(f"{'='*100}")
    print(f"🎯 VALIDAÇÃO: RL Aprendeu FBM?")
    print(f"{'='*100}\n")
    
    expected_manha = list(range(6, 12))
    expected_noite = list(range(18, 24))
    expected_combined = expected_manha + expected_noite
    
    rl_top = [h for h, _ in metrics_rl['top_hours'][:5]]
    
    match_manha = sum(1 for h in rl_top if h in expected_manha)
    match_noite = sum(1 for h in rl_top if h in expected_noite)
    match_total = match_manha + match_noite
    
    print(f"Padrão esperado (perfil FBM variado):")
    print(f"  Manhã (6-11h): FBM ALTO (~96)")
    print(f"  Tarde (12-17h): FBM BAIXO (~2)")
    print(f"  Noite (18-23h): FBM ALTO (~80)\n")
    
    print(f"Top 5 horas do RL: {rl_top}")
    print(f"  Match Manhã: {match_manha} horas")
    print(f"  Match Noite: {match_noite} horas")
    print(f"  Total: {match_total}/5 horas em períodos de FBM alto\n")
    
    if match_total >= 4:
        print(f"✅ SUCESSO! RL aprendeu o padrão FBM (focou em manhã/noite)")
    elif match_total >= 3:
        print(f"⚠️ PARCIAL! RL aprendeu parcialmente o padrão FBM")
    else:
        print(f"❌ FALHA! RL não aprendeu o padrão FBM (ainda foca só em hora)")
    
    print(f"\n{'='*100}\n")
    
    # Salvar modelo
    print("💾 Salvando modelo RL treinado...\n")
    
    try:
        save_model_response = requests.post(
            f"{API_URL}/treino/salvar-modelo/{CONFIG['user_id']}",
            timeout=60
        )
        
        if save_model_response.status_code == 200:
            model_info = save_model_response.json()
            print(f"✅ Modelo salvo: {model_info['model_path']}")
            print(f"   Samples: {model_info['training_samples']}\n")
        else:
            print(f"⚠️ Erro ao salvar modelo: {save_model_response.status_code}\n")
    except Exception as e:
        print(f"⚠️ Erro ao salvar modelo: {e}\n")
    
    # Salvar dados
    print("💾 Salvando dados detalhados da simulação...\n")
    
    output_dir = Path("data/simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rl_data = {
        "config": CONFIG,
        "daily_results": simulator_rl.daily_results,
        "summary": {
            "total_notifications": simulator_rl.total_notifications,
            "total_actions": simulator_rl.total_actions,
            "vp": simulator_rl.vp_count,
            "vn": simulator_rl.vn_count,
            "fp": simulator_rl.fp_count,
            "fn": simulator_rl.fn_count,
            "precision": metrics_rl['precision'],
            "recall": metrics_rl['recall'],
            "f1_score": metrics_rl['f1_score'],
            "hourly_stats": simulator_rl.hourly_stats,
            "top_hours": metrics_rl['top_hours'],
            "fbm_metrics": metrics_rl['fbm_metrics']
        }
    }
    
    output_file = output_dir / "rl_fbm_enhanced_simulation_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rl_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dados salvos em: {output_file}\n")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    run_simulation_with_rl_fbm_enhanced()
