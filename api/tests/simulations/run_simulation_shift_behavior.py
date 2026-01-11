"""
Simulação de SHIFT DE COMPORTAMENTO - Validação do Sistema RL

CENÁRIO:
- Dias 1-30:  Perfil MATINAL (FBM alto de manhã 6-11h, executa atividades de manhã)
- Dias 31-90: Perfil NOTURNO (FBM alto de noite 18-23h, executa atividades de noite)

OBJETIVO:
Validar se o sistema RL consegue:
1. Aprender o padrão matinal nos primeiros 30 dias
2. Detectar o shift de comportamento no dia 31
3. Adaptar-se ao novo padrão noturno nos dias seguintes

MECANISMO DE DETECÇÃO DE SHIFT:
- Monitora taxa de VN recente (últimos 5 dias)
- Se taxa de VN aumentar significativamente após o modelo ter "aprendido",
  aumenta epsilon temporariamente para re-explorar
- Isso permite ao modelo detectar mudanças de padrão
"""

import json
import requests
import random
import shutil
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Any, Tuple
from copy import deepcopy
import sys

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np


API_URL = "http://localhost:8000"
API_TIMEOUT = 30


# ============================================================================
# CONFIGURAÇÕES DOS PERFIS
# ============================================================================

# Perfil Matinal: FBM alto de manhã (6-11h), baixo tarde, médio noite
PERFIL_MATINAL = {
    "nome": "matinal",
    "descricao": "Usuário executa atividades principalmente de manhã",
    "horas": {
        # MANHÃ (6-11h): FBM ALTO - executa atividades
        (6, 11): {
            "motivation_factors": {
                "valence": 3,  # 0-4: 3=positivo
                "last_activity_score": 3,  # 0-4
                "hours_slept_last_night": 8
            },
            "ability_factors": {
                "cognitive_load": 2,  # 0-10: baixo
                "confidence_score": 9,  # 0-10
                "activities_performed_today": 0,
                "time_since_last_activity_hours": 12
            },
            "trigger_factors": {
                "sleeping": False,
                "arousal": 8,  # 0-10
                "location": "home",
                "motion_activity": "walking"
            },
            "prob_resposta_com_notif": 0.90,
            "prob_resposta_sem_notif": 0.50,
        },
        # TARDE (12-17h): FBM BAIXO - não executa
        (12, 17): {
            "motivation_factors": {
                "valence": 0,
                "last_activity_score": 0,
                "hours_slept_last_night": 8
            },
            "ability_factors": {
                "cognitive_load": 8,  # alto
                "confidence_score": 3,
                "activities_performed_today": 2,
                "time_since_last_activity_hours": 2
            },
            "trigger_factors": {
                "sleeping": False,
                "arousal": 2,
                "location": "work",
                "motion_activity": "stationary"
            },
            "prob_resposta_com_notif": 0.02,
            "prob_resposta_sem_notif": 0.00,
        },
        # NOITE (18-23h): FBM BAIXO para perfil matinal
        (18, 23): {
            "motivation_factors": {
                "valence": 1,
                "last_activity_score": 1,
                "hours_slept_last_night": 8
            },
            "ability_factors": {
                "cognitive_load": 7,
                "confidence_score": 4,
                "activities_performed_today": 3,
                "time_since_last_activity_hours": 1
            },
            "trigger_factors": {
                "sleeping": False,
                "arousal": 3,
                "location": "home",
                "motion_activity": "stationary"
            },
            "prob_resposta_com_notif": 0.05,
            "prob_resposta_sem_notif": 0.01,
        },
        # MADRUGADA (0-5h): Dormindo
        (0, 5): {
            "motivation_factors": {
                "valence": 0,
                "last_activity_score": 0,
                "hours_slept_last_night": 0
            },
            "ability_factors": {
                "cognitive_load": 10,
                "confidence_score": 0,
                "activities_performed_today": 0,
                "time_since_last_activity_hours": 0
            },
            "trigger_factors": {
                "sleeping": True,
                "arousal": 0,
                "location": "home",
                "motion_activity": "stationary"
            },
            "prob_resposta_com_notif": 0.0,
            "prob_resposta_sem_notif": 0.0,
        }
    }
}

# Perfil Noturno: FBM alto de noite (18-23h), baixo manhã e tarde
PERFIL_NOTURNO = {
    "nome": "noturno",
    "descricao": "Usuário executa atividades principalmente de noite (após shift)",
    "horas": {
        # MANHÃ (6-11h): FBM BAIXO após shift - não executa mais
        (6, 11): {
            "motivation_factors": {
                "valence": 0,
                "last_activity_score": 0,
                "hours_slept_last_night": 5  # Dormiu pouco
            },
            "ability_factors": {
                "cognitive_load": 8,
                "confidence_score": 3,
                "activities_performed_today": 0,
                "time_since_last_activity_hours": 8
            },
            "trigger_factors": {
                "sleeping": False,
                "arousal": 2,
                "location": "work",
                "motion_activity": "stationary"
            },
            "prob_resposta_com_notif": 0.02,
            "prob_resposta_sem_notif": 0.00,
        },
        # TARDE (12-17h): FBM BAIXO - continua não executando
        (12, 17): {
            "motivation_factors": {
                "valence": 0,
                "last_activity_score": 0,
                "hours_slept_last_night": 5
            },
            "ability_factors": {
                "cognitive_load": 8,
                "confidence_score": 3,
                "activities_performed_today": 0,
                "time_since_last_activity_hours": 4
            },
            "trigger_factors": {
                "sleeping": False,
                "arousal": 2,
                "location": "work",
                "motion_activity": "stationary"
            },
            "prob_resposta_com_notif": 0.02,
            "prob_resposta_sem_notif": 0.00,
        },
        # NOITE (18-23h): FBM ALTO - agora executa atividades
        (18, 23): {
            "motivation_factors": {
                "valence": 3,
                "last_activity_score": 3,
                "hours_slept_last_night": 5
            },
            "ability_factors": {
                "cognitive_load": 2,
                "confidence_score": 9,
                "activities_performed_today": 0,
                "time_since_last_activity_hours": 12
            },
            "trigger_factors": {
                "sleeping": False,
                "arousal": 8,
                "location": "home",
                "motion_activity": "walking"
            },
            "prob_resposta_com_notif": 0.90,
            "prob_resposta_sem_notif": 0.50,
        },
        # MADRUGADA (0-5h): Dormindo
        (0, 5): {
            "motivation_factors": {
                "valence": 0,
                "last_activity_score": 0,
                "hours_slept_last_night": 0
            },
            "ability_factors": {
                "cognitive_load": 10,
                "confidence_score": 0,
                "activities_performed_today": 0,
                "time_since_last_activity_hours": 0
            },
            "trigger_factors": {
                "sleeping": True,
                "arousal": 0,
                "location": "home",
                "motion_activity": "stationary"
            },
            "prob_resposta_com_notif": 0.0,
            "prob_resposta_sem_notif": 0.0,
        }
    }
}


def get_hour_config(hour: int, perfil: Dict) -> Dict:
    """Retorna configuração para uma hora específica baseado no perfil."""
    for (start, end), config in perfil["horas"].items():
        if start <= hour <= end:
            return config
    # Default para madrugada
    return perfil["horas"][(0, 5)]


def add_noise_to_factors(factors: Dict, noise_factor: float) -> Dict:
    """Adiciona ruído gaussiano aos fatores FBM, mantendo valores inteiros válidos."""
    noisy = deepcopy(factors)
    
    # Motivation factors
    if "motivation_factors" in noisy:
        mf = noisy["motivation_factors"]
        # valence: 0-4
        mf["valence"] = int(max(0, min(4, mf["valence"] + np.random.randint(-1, 2) * (1 if random.random() < noise_factor else 0))))
        # last_activity_score: 0-4
        mf["last_activity_score"] = int(max(0, min(4, mf["last_activity_score"] + np.random.randint(-1, 2) * (1 if random.random() < noise_factor else 0))))
        # hours_slept_last_night: 0-12
        mf["hours_slept_last_night"] = int(max(0, min(12, mf["hours_slept_last_night"] + np.random.randint(-2, 3) * (1 if random.random() < noise_factor else 0))))
    
    # Ability factors
    if "ability_factors" in noisy:
        af = noisy["ability_factors"]
        # cognitive_load: 0-10
        af["cognitive_load"] = int(max(0, min(10, af["cognitive_load"] + np.random.randint(-2, 3) * (1 if random.random() < noise_factor else 0))))
        # confidence_score: 0-10
        af["confidence_score"] = int(max(0, min(10, af["confidence_score"] + np.random.randint(-2, 3) * (1 if random.random() < noise_factor else 0))))
        # activities_performed_today: 0-10
        af["activities_performed_today"] = int(max(0, min(10, af["activities_performed_today"] + np.random.randint(-1, 2) * (1 if random.random() < noise_factor else 0))))
        # time_since_last_activity_hours: 0-24
        af["time_since_last_activity_hours"] = int(max(0, min(24, af["time_since_last_activity_hours"] + np.random.randint(-2, 3) * (1 if random.random() < noise_factor else 0))))
    
    # Trigger factors
    if "trigger_factors" in noisy:
        tf = noisy["trigger_factors"]
        if not tf.get("sleeping", False):
            # arousal: 0-10
            tf["arousal"] = int(max(0, min(10, tf["arousal"] + np.random.randint(-2, 3) * (1 if random.random() < noise_factor else 0))))
    
    return noisy


class ShiftBehaviorSimulator:
    """
    Simulador que valida a capacidade do sistema RL de detectar
    e adaptar-se a mudanças de comportamento do usuário.
    """
    
    def __init__(self, api_url: str, config: Dict):
        self.api_url = api_url
        self.config = config
        self.user_id = config["user_id"]
        
        # Métricas globais
        self.total_notifications = 0
        self.total_actions = 0
        self.vp_count = 0
        self.vn_count = 0
        self.fp_count = 0
        self.fn_count = 0
        
        # Métricas por faixa de FBM
        self.fbm_metrics = {
            "alto": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0},
            "medio": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0},
            "baixo": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0}
        }
        
        # Stats por hora
        self.hourly_stats = {hour: {"notified": 0, "responded": 0, "fbm_avg": 0, "count": 0} for hour in range(24)}
        
        # Resultados diários
        self.daily_results = []
        
        # 🔥 SISTEMA DE EXPLORAÇÃO ADAPTATIVA
        self.epsilon = 0.30  # Começa com 30% exploration
        self.epsilon_min = 0.02  # Mínimo baixo quando modelo aprende bem
        self.epsilon_max = 0.50  # Máximo para re-exploração após shift
        self.epsilon_decay_rate = 0.92  # Decay mais agressivo
        
        # 🎯 DETECÇÃO DE SHIFT BASEADA EM VP E VN
        self.vp_history = []  # Histórico de VP por dia
        self.vn_history = []  # Histórico de VN por dia
        self.shift_detected = False
        self.shift_detection_day = None
        self.window_size = 5  # Janela para detectar anomalias
        
        # Thresholds para detecção
        self.vp_high_threshold = 8  # VP alto = modelo aprendeu
        self.vp_drop_threshold = 0.5  # Queda de 50% nos VPs indica shift
        self.vn_spike_threshold = 2.0  # VN > 2x média indica shift
        
        # Controle de boost
        self.exploration_boost_days = 15  # Dias de exploração extra após shift
        self.days_since_boost = 0
        self.consecutive_good_days = 0  # Dias consecutivos com VP alto
        
        # Métricas por fase
        self.phase_metrics = {
            "fase1_matinal": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "dias": 0},
            "fase2_noturno": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "dias": 0}
        }
        
        # Métricas por período do dia
        self.period_metrics = {
            "manha": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0},
            "tarde": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0},
            "noite": {"vp": 0, "vn": 0, "fp": 0, "fn": 0, "notified": 0}
        }
        
    def get_period(self, hour: int) -> str:
        """Retorna período do dia para uma hora."""
        if 6 <= hour <= 11:
            return "manha"
        elif 12 <= hour <= 17:
            return "tarde"
        elif 18 <= hour <= 23:
            return "noite"
        else:
            return "madrugada"
    
    def detect_shift(self, day_vp: int, day_vn: int, day_num: int) -> bool:
        """
        Detecta shift de comportamento baseado em VP e VN.
        
        Lógica:
        1. VP alto consecutivo → modelo aprendeu → decay agressivo
        2. Queda súbita de VP + aumento de VN → possível shift → boost exploração
        """
        self.vp_history.append(day_vp)
        self.vn_history.append(day_vn)
        
        # Só detecta após período de aprendizado inicial
        if day_num < 10:
            return False
        
        # Em modo boost após shift
        if self.shift_detected and self.days_since_boost < self.exploration_boost_days:
            self.days_since_boost += 1
            # Se durante boost os VPs voltarem a ser altos, pode sair mais cedo
            if day_vp >= self.vp_high_threshold:
                self.consecutive_good_days += 1
                if self.consecutive_good_days >= 3:
                    print(f"\n   ✅ Re-aprendizado completo! Saindo do modo boost.")
                    self.days_since_boost = self.exploration_boost_days
            else:
                self.consecutive_good_days = 0
            return False
        
        # Análise de padrão
        if len(self.vp_history) > self.window_size:
            recent_vp = self.vp_history[-(self.window_size + 1):-1]
            recent_vn = self.vn_history[-(self.window_size + 1):-1]
            
            avg_vp = sum(recent_vp) / len(recent_vp)
            avg_vn = sum(recent_vn) / len(recent_vn)
            
            # CONDIÇÃO 1: Queda brusca de VP
            vp_dropped = avg_vp > 0 and day_vp < avg_vp * self.vp_drop_threshold
            
            # CONDIÇÃO 2: Spike de VN
            vn_spiked = avg_vn > 0 and day_vn > avg_vn * self.vn_spike_threshold
            
            # CONDIÇÃO 3: VP atual muito baixo + VN alto
            performance_collapsed = day_vp <= 2 and day_vn >= 6
            
            if (vp_dropped and vn_spiked) or performance_collapsed:
                print(f"\n   🚨 SHIFT DETECTADO!")
                print(f"      VP atual: {day_vp} (média: {avg_vp:.1f})")
                print(f"      VN atual: {day_vn} (média: {avg_vn:.1f})")
                print(f"   📈 Ativando re-exploração para adaptar ao novo padrão\n")
                
                if not self.shift_detected:
                    self.shift_detected = True
                    self.shift_detection_day = day_num
                
                self.days_since_boost = 0
                self.consecutive_good_days = 0
                return True
        
        return False
    
    def update_epsilon(self, day_vp: int, day_vn: int, day_num: int):
        """
        Atualiza epsilon com lógica adaptativa baseada em VP e VN.
        
        - VP alto consecutivo → decay AGRESSIVO (modelo aprendeu)
        - Shift detectado → boost para re-explorar
        - VN alto isolado → decay extra
        """
        shift_detected = self.detect_shift(day_vp, day_vn, day_num)
        
        if shift_detected:
            # 🔥 BOOST: Aumenta exploração após detectar shift
            self.epsilon = self.epsilon_max
            print(f"   ⬆️ Epsilon = {self.epsilon:.3f} (BOOST re-exploração)")
        
        elif day_vp >= self.vp_high_threshold:
            # ✅ VP ALTO: Modelo está acertando → decay AGRESSIVO
            self.consecutive_good_days += 1
            
            if self.consecutive_good_days >= 3:
                # 3+ dias bons consecutivos → decay muito agressivo
                self.epsilon *= 0.80  # -20% por dia
                print(f"   ⬇️ Epsilon = {self.epsilon:.3f} (VP alto consecutivo: {self.consecutive_good_days} dias)")
            else:
                self.epsilon *= 0.90  # -10% por dia
        
        elif day_vp >= 5:
            # VP médio → decay normal
            self.consecutive_good_days = 0
            self.epsilon *= self.epsilon_decay_rate
        
        else:
            # VP baixo → decay mais lento + possível micro-boost
            self.consecutive_good_days = 0
            self.epsilon *= 0.98  # Decay muito lento
            
            # Se VN muito alto, pequeno boost para explorar alternativas
            if day_vn > 8:
                self.epsilon = min(self.epsilon * 1.1, self.epsilon_max * 0.5)
        
        # Mantém dentro dos limites
        self.epsilon = max(self.epsilon_min, min(self.epsilon_max, self.epsilon))
    
    def calculate_fbm_scores(self, hour_data: Dict) -> Dict:
        """Calcula scores M, A, T do FBM."""
        mf = hour_data.get("motivation_factors", {})
        af = hour_data.get("ability_factors", {})
        tf = hour_data.get("trigger_factors", {})
        
        # MOTIVATION (0-4) baseado em valence (0-4)
        valence = mf.get("valence", 0)
        last_activity = mf.get("last_activity_score", 0)
        
        if valence >= 3:
            m = 4
        elif valence >= 2:
            m = 3
        elif valence >= 1:
            m = 2
        elif last_activity >= 2:
            m = 1
        else:
            m = 0
        
        # ABILITY (0-4) baseado em cognitive_load (0-10) e confidence (0-10)
        cognitive_load = af.get("cognitive_load", 5)
        confidence = af.get("confidence_score", 5)
        
        if cognitive_load <= 3 and confidence >= 7:
            a = 4
        elif cognitive_load <= 5 and confidence >= 5:
            a = 3
        elif cognitive_load <= 7:
            a = 2
        elif cognitive_load <= 9:
            a = 1
        else:
            a = 0
        
        # TRIGGER (0-6) baseado em arousal (0-10)
        sleeping = tf.get("sleeping", False)
        arousal = tf.get("arousal", 0)
        location = tf.get("location", "unknown")
        
        if sleeping:
            t = 0
        else:
            # arousal 0-10 → t 0-5
            t = min(5, arousal // 2)
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
    
    def should_user_respond(self, hour: int, was_notified: bool, perfil: Dict) -> bool:
        """
        Simula resposta do usuário baseado no PERFIL ATUAL.
        
        O comportamento do usuário muda de acordo com o perfil (matinal vs noturno).
        """
        config = get_hour_config(hour, perfil)
        
        if config["trigger_factors"].get("sleeping", False):
            return False
        
        if was_notified:
            prob = config["prob_resposta_com_notif"]
        else:
            prob = config["prob_resposta_sem_notif"]
        
        return random.random() < prob
    
    def get_rl_decision_for_hour(self, hour_data: List[Dict], target_hour: int, fbm_score: int) -> bool:
        """
        RL Decision: Epsilon-greedy com exploração adaptativa.
        
        - Exploration (epsilon): Ação aleatória
        - Exploitation (1-epsilon): PPO decide
        """
        # Exploration
        if random.random() < self.epsilon:
            return random.random() < 0.5
        
        # Exploitation via API
        try:
            response = requests.post(
                f"{self.api_url}/previsao/{self.user_id}/custom",
                json={"hours_data": hour_data},
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
    
    def simulate_reward_for_outcome(self, outcome_type: str, fbm_score: int) -> float:
        """Calcula recompensa simulada proporcional ao FBM."""
        if outcome_type == "VP":
            if fbm_score >= 60:
                return 35.0
            elif fbm_score >= 40:
                return 25.0
            else:
                return 15.0
        
        elif outcome_type == "VN":
            if fbm_score >= 60:
                return -3.0
            elif fbm_score >= 40:
                return -8.0
            else:
                return -15.0
        
        elif outcome_type == "FP":
            return -2.0
        
        else:  # FN
            return 0.5
    
    def generate_day_data(self, day_num: int, perfil: Dict, base_date: date, noise_factor: float = 0.15) -> Dict:
        """
        Gera dados de um dia baseado no perfil especificado.
        """
        np.random.seed(self.config["seed"] + day_num)
        
        current_date = base_date + timedelta(days=day_num - 1)
        is_weekend = current_date.weekday() >= 5
        
        hours = []
        for hour in range(24):
            config = get_hour_config(hour, perfil)
            
            # Aplica ruído aos fatores
            noisy_config = add_noise_to_factors(config, noise_factor)
            
            # Determina day_period (0=manhã, 1=meio-dia, 2=noite, 3=madrugada)
            if 6 <= hour < 10:
                day_period = 0  # manhã
            elif 10 <= hour < 18:
                day_period = 1  # meio-dia
            elif 18 <= hour < 22:
                day_period = 2  # noite
            else:
                day_period = 3  # madrugada
            
            hour_data = {
                "hour": hour,
                "motivation_factors": noisy_config["motivation_factors"],
                "ability_factors": noisy_config["ability_factors"],
                "trigger_factors": noisy_config["trigger_factors"],
                "context": {
                    "weather": random.choice(["sunny", "cloudy", "rainy"]),
                    "temperature": random.randint(15, 30),
                    "day_period": day_period,
                    "is_weekend": is_weekend
                },
                "feedback": {
                    "notification_sent": False,
                    "action_performed": False,
                    "training_feedback": None
                }
            }
            hours.append(hour_data)
        
        return {
            "user_id": self.user_id,
            "date": current_date.strftime("%Y-%m-%d"),
            "timezone": "America/Sao_Paulo",
            "hours": hours
        }
    
    def simulate_day(self, day_data: Dict, day_num: int, perfil: Dict, fase: str) -> Dict:
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
            
            # RL decide se notifica
            rl_notifies = self.get_rl_decision_for_hour(day_hours, hour, fbm_score)
            
            # Usuário responde (baseado no perfil ATUAL, não no FBM calculado)
            user_responded = self.should_user_respond(hour, rl_notifies, perfil)
            
            # Determina outcome
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
            
            # Atualiza métricas por faixa de FBM
            fbm_cat = self.get_fbm_category(fbm_score)
            self.fbm_metrics[fbm_cat][outcome.lower()] += 1
            if rl_notifies:
                self.fbm_metrics[fbm_cat]["notified"] += 1
            
            # Atualiza métricas por período
            period = self.get_period(hour)
            if period != "madrugada":
                self.period_metrics[period][outcome.lower()] += 1
                if rl_notifies:
                    self.period_metrics[period]["notified"] += 1
            
            # Atualiza métricas por fase
            self.phase_metrics[fase][outcome.lower()] += 1
            
            # Calcula recompensa simulada
            simulated_reward = self.simulate_reward_for_outcome(outcome, fbm_score)
            
            # Atualiza métricas globais
            if rl_notifies:
                self.total_notifications += 1
                self.hourly_stats[hour]["notified"] += 1
            
            if user_responded:
                self.total_actions += 1
                self.hourly_stats[hour]["responded"] += 1
            
            self.hourly_stats[hour]["fbm_avg"] += fbm_score
            self.hourly_stats[hour]["count"] += 1
            
            # Atualiza feedback nos dados originais
            day_hours[hour_idx]["feedback"]["notification_sent"] = rl_notifies
            day_hours[hour_idx]["feedback"]["action_performed"] = user_responded
            
            results.append({
                "hour": hour,
                "fbm_score": fbm_score,
                "fbm_category": fbm_cat,
                "period": period,
                "motivation": fbm_calc["motivation"],
                "ability": fbm_calc["ability"],
                "trigger": fbm_calc["trigger"],
                "rl_notified": rl_notifies,
                "user_responded": user_responded,
                "outcome": outcome,
                "simulated_reward": simulated_reward
            })
        
        fbm_avg = sum(day_fbm_scores) / len(day_fbm_scores) if day_fbm_scores else 0
        
        day_result = {
            "day_num": day_num,
            "date": day_data["date"],
            "fase": fase,
            "perfil": perfil["nome"],
            "results": results,
            "day_vp": sum(1 for r in results if r["outcome"] == "VP"),
            "day_vn": sum(1 for r in results if r["outcome"] == "VN"),
            "day_fp": sum(1 for r in results if r["outcome"] == "FP"),
            "day_fn": sum(1 for r in results if r["outcome"] == "FN"),
            "fbm_avg": fbm_avg,
            "epsilon": self.epsilon,
            "shift_detected": self.shift_detected,
            "shift_detection_day": self.shift_detection_day
        }
        
        self.daily_results.append(day_result)
        self.phase_metrics[fase]["dias"] += 1
        
        return day_result
    
    def train_with_day(self, day_data: Dict, day_num: int) -> bool:
        """Envia dia para API /treino E treina modelo incrementalmente."""
        try:
            api_data = {
                "user_id": day_data["user_id"],
                "date": day_data["date"],
                "timezone": day_data["timezone"],
                "user_profile": {"has_family": True},
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
            
            # Treina modelo incrementalmente
            train_response = requests.post(
                f"{self.api_url}/treino/treinar-incremental/{self.user_id}",
                timeout=120
            )
            
            if train_response.status_code == 200:
                return True
            else:
                try:
                    error_detail = train_response.json()
                    print(f"\n   ⚠️ Erro {train_response.status_code} ao treinar modelo:")
                    print(f"   {error_detail}")
                except:
                    print(f"\n   ⚠️ Erro {train_response.status_code}: {train_response.text[:300]}")
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
        
        # FBM médio por hora
        for hour in range(24):
            if self.hourly_stats[hour]["count"] > 0:
                self.hourly_stats[hour]["fbm_avg"] /= self.hourly_stats[hour]["count"]
        
        # Top horas por resposta
        top_hours = sorted(
            [(h, stats["responded"]) for h, stats in self.hourly_stats.items() if stats["responded"] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Métricas por faixa de FBM
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
        
        # Métricas por período
        period_summary = {}
        for period, metrics in self.period_metrics.items():
            vp_p = metrics["vp"]
            vn_p = metrics["vn"]
            fp_p = metrics["fp"]
            
            prec = (vp_p / (vp_p + vn_p) * 100) if (vp_p + vn_p) > 0 else 0
            rec = (vp_p / (vp_p + fp_p) * 100) if (vp_p + fp_p) > 0 else 0
            
            period_summary[period] = {
                "precision": prec,
                "recall": rec,
                "vp": vp_p,
                "vn": vn_p,
                "fp": fp_p,
                "fn": metrics["fn"],
                "notified": metrics["notified"]
            }
        
        # Métricas por fase
        phase_summary = {}
        for phase, metrics in self.phase_metrics.items():
            vp_ph = metrics["vp"]
            vn_ph = metrics["vn"]
            fp_ph = metrics["fp"]
            
            prec = (vp_ph / (vp_ph + vn_ph) * 100) if (vp_ph + vn_ph) > 0 else 0
            rec = (vp_ph / (vp_ph + fp_ph) * 100) if (vp_ph + fp_ph) > 0 else 0
            
            phase_summary[phase] = {
                "precision": prec,
                "recall": rec,
                "vp": vp_ph,
                "vn": vn_ph,
                "fp": fp_ph,
                "fn": metrics["fn"],
                "dias": metrics["dias"]
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
            "fbm_metrics": fbm_summary,
            "period_metrics": period_summary,
            "phase_metrics": phase_summary,
            "shift_detected": self.shift_detected,
            "shift_detection_day": self.shift_detection_day
        }


def run_shift_behavior_simulation():
    """
    Executa simulação de SHIFT DE COMPORTAMENTO.
    
    - Dias 1-30: Perfil MATINAL
    - Dias 31-90: Perfil NOTURNO
    """
    
    CONFIG = {
        "num_days": 90,
        "shift_day": 30,  # Dia do shift
        "num_epochs": 100,  # 100 variações por dia
        "user_id": "user_shift_behavior",
        "initial_threshold": 40.0,
        "seed": 42,
        "noise_factor": 0.15
    }
    
    print("\n" + "=" * 100)
    print("🔄 SIMULAÇÃO DE SHIFT DE COMPORTAMENTO")
    print("=" * 100)
    print(f"\n📋 Configuração:")
    print(f"   Total de dias: {CONFIG['num_days']}")
    print(f"   Dia do shift: {CONFIG['shift_day']}")
    print(f"   Epochs por dia: {CONFIG['num_epochs']}")
    print(f"   User ID: {CONFIG['user_id']}")
    print(f"   Seed: {CONFIG['seed']}")
    print()
    print("🎯 CENÁRIO:")
    print(f"   Dias 1-{CONFIG['shift_day']}: Perfil MATINAL (FBM alto 6-11h)")
    print(f"   Dias {CONFIG['shift_day']+1}-{CONFIG['num_days']}: Perfil NOTURNO (FBM alto 18-23h)")
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
    print("\n🧹 Limpando dados anteriores...")
    try:
        response = requests.delete(
            f"{API_URL}/treino/historico/{CONFIG['user_id']}",
            timeout=5
        )
        if response.status_code in [200, 404]:
            print("✅ Histórico limpo!")
    except:
        print("⚠️ Aviso: erro ao limpar histórico")
    
    # Deleta modelo antigo
    print("🧹 Deletando modelo RL antigo...")
    model_path = Path(f"models/ppo_{CONFIG['user_id']}")
    if model_path.exists():
        try:
            shutil.rmtree(model_path)
            print("✅ Modelo antigo deletado!")
        except Exception as e:
            print(f"⚠️ Erro ao deletar modelo: {e}")
    else:
        print("✅ Nenhum modelo antigo encontrado")
    
    print("\n" + "=" * 100)
    print("📊 INICIANDO SIMULAÇÃO")
    print("=" * 100 + "\n")
    
    simulator = ShiftBehaviorSimulator(api_url=API_URL, config=CONFIG)
    
    base_date = date(2025, 1, 1)
    
    # Loop de epochs
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"🔄 EPOCH {epoch}/{CONFIG['num_epochs']}")
        print(f"{'='*80}\n")
        
        # Atualiza seed para variação
        np.random.seed(CONFIG['seed'] + epoch)
        
        for day_num in range(1, CONFIG['num_days'] + 1):
            # Determina perfil baseado no dia
            if day_num <= CONFIG['shift_day']:
                perfil = PERFIL_MATINAL
                fase = "fase1_matinal"
            else:
                perfil = PERFIL_NOTURNO
                fase = "fase2_noturno"
            
            # Indica momento do shift
            if day_num == CONFIG['shift_day'] + 1 and epoch == 1:
                print(f"\n{'!'*80}")
                print(f"⚠️  SHIFT DE COMPORTAMENTO - Dia {day_num}")
                print(f"   Usuário mudou de perfil MATINAL para NOTURNO")
                print(f"{'!'*80}\n")
            
            # Gera dados do dia
            day_data = simulator.generate_day_data(
                day_num=day_num + (epoch - 1) * CONFIG['num_days'],
                perfil=perfil,
                base_date=base_date,
                noise_factor=CONFIG['noise_factor']
            )
            
            # Simula dia
            day_result = simulator.simulate_day(day_data, day_num, perfil, fase)
            
            # Atualiza epsilon (passa VP e VN)
            simulator.update_epsilon(day_result['day_vp'], day_result['day_vn'], day_num)
            
            # Total de samples até agora
            total_samples = (epoch - 1) * CONFIG['num_days'] * 24 + day_num * 24
            
            # Log
            shift_marker = "🔄" if simulator.shift_detected and day_num == simulator.shift_detection_day else ""
            phase_marker = "☀️" if day_num <= CONFIG['shift_day'] else "🌙"
            
            print(f"E{epoch:3d} D{day_num:2d} {phase_marker}: VP={day_result['day_vp']:2d}, VN={day_result['day_vn']:2d}, "
                  f"FP={day_result['day_fp']:2d}, ε={simulator.epsilon:.3f} {shift_marker}", end=" ")
            
            # Treina
            trained = simulator.train_with_day(day_data, total_samples // 24)
            
            if trained:
                print(f"✅ ({total_samples:,} samples)")
            else:
                print(f"❌")
    
    # Métricas finais
    metrics = simulator.get_metrics()
    
    print("\n" + "=" * 100)
    print("📊 RESULTADOS FINAIS")
    print("=" * 100)
    
    print(f"\n📈 Métricas Globais:")
    print(f"   Precision: {metrics['precision']:.1f}%")
    print(f"   Recall:    {metrics['recall']:.1f}%")
    print(f"   F1-Score:  {metrics['f1_score']:.1f}%")
    print(f"   VP: {metrics['vp']}, VN: {metrics['vn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    print(f"\n📊 Métricas por FASE:")
    print(f"{'Fase':<20} | {'Precision':<12} | {'Recall':<12} | {'VP':<6} | {'VN':<6} | {'FP':<6} | {'Dias':<6}")
    print("-" * 85)
    for phase, m in metrics['phase_metrics'].items():
        print(f"{phase:<20} | {m['precision']:>10.1f}% | {m['recall']:>10.1f}% | {m['vp']:>6} | {m['vn']:>6} | {m['fp']:>6} | {m['dias']:>6}")
    
    print(f"\n📊 Métricas por PERÍODO DO DIA:")
    print(f"{'Período':<15} | {'Precision':<12} | {'Recall':<12} | {'VP':<6} | {'VN':<6} | {'FP':<6} | {'Notificações':<15}")
    print("-" * 95)
    for period in ["manha", "tarde", "noite"]:
        m = metrics['period_metrics'][period]
        print(f"{period:<15} | {m['precision']:>10.1f}% | {m['recall']:>10.1f}% | {m['vp']:>6} | {m['vn']:>6} | {m['fp']:>6} | {m['notified']:>15}")
    
    print(f"\n🎯 Top 10 Horas por Resposta:")
    for h, count in metrics['top_hours'][:10]:
        bar = "█" * (count // 50)
        print(f"   {h:2d}h: {count:4d} respostas {bar}")
    
    print(f"\n🔍 Detecção de Shift:")
    if metrics['shift_detected']:
        print(f"   ✅ Shift DETECTADO no dia {metrics['shift_detection_day']}")
    else:
        print(f"   ❌ Shift NÃO detectado pelo sistema")
    
    # Validação
    print("\n" + "=" * 100)
    print("🎯 VALIDAÇÃO DO SISTEMA")
    print("=" * 100)
    
    # Esperado: Fase 1 deve ter VPs concentrados na manhã
    # Fase 2 deve ter VPs concentrados na noite
    
    manha_vp = metrics['period_metrics']['manha']['vp']
    noite_vp = metrics['period_metrics']['noite']['vp']
    tarde_vp = metrics['period_metrics']['tarde']['vp']
    
    print(f"\n📊 Distribuição de VP por período:")
    print(f"   Manhã (6-11h):  {manha_vp:,} VPs")
    print(f"   Tarde (12-17h): {tarde_vp:,} VPs")
    print(f"   Noite (18-23h): {noite_vp:,} VPs")
    
    # Para 90 dias com shift no dia 30:
    # - 30 dias matinais: Maioria dos VPs devem ser de manhã
    # - 60 dias noturnos: Maioria dos VPs devem ser de noite
    
    total_vp = manha_vp + noite_vp + tarde_vp
    
    if total_vp > 0:
        manha_pct = manha_vp / total_vp * 100
        noite_pct = noite_vp / total_vp * 100
        
        print(f"\n   Manhã: {manha_pct:.1f}% dos VPs")
        print(f"   Noite: {noite_pct:.1f}% dos VPs")
        
        # Como temos 30 dias matinais e 60 noturnos, esperamos mais VPs à noite
        if noite_pct > manha_pct:
            print(f"\n   ✅ CORRETO! Mais VPs à noite (esperado após shift)")
        else:
            print(f"\n   ⚠️ Atenção: Mais VPs de manhã do que à noite")
    
    # Verifica adaptação ao shift
    print(f"\n🔄 Análise de Adaptação ao Shift:")
    
    fase1_metrics = metrics['phase_metrics']['fase1_matinal']
    fase2_metrics = metrics['phase_metrics']['fase2_noturno']
    
    print(f"   Fase 1 (Matinal):  Precision={fase1_metrics['precision']:.1f}%, Recall={fase1_metrics['recall']:.1f}%")
    print(f"   Fase 2 (Noturno):  Precision={fase2_metrics['precision']:.1f}%, Recall={fase2_metrics['recall']:.1f}%")
    
    # Se precision da fase 2 é boa, o modelo se adaptou
    if fase2_metrics['precision'] >= 70:
        print(f"\n   ✅ SUCESSO! Sistema se adaptou ao novo padrão noturno")
    elif fase2_metrics['precision'] >= 50:
        print(f"\n   ⚠️ PARCIAL! Sistema está se adaptando ao novo padrão")
    else:
        print(f"\n   ❌ FALHA! Sistema não conseguiu se adaptar ao shift")
    
    # Salvar modelo
    print("\n💾 Salvando modelo RL treinado...")
    try:
        save_response = requests.post(
            f"{API_URL}/treino/salvar-modelo/{CONFIG['user_id']}",
            timeout=60
        )
        if save_response.status_code == 200:
            model_info = save_response.json()
            print(f"✅ Modelo salvo: {model_info.get('model_path', 'OK')}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar modelo: {e}")
    
    # Salvar dados
    print("\n💾 Salvando dados da simulação...")
    
    output_dir = Path(__file__).parent.parent / "data" / "simulation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "config": CONFIG,
        "scenario": {
            "description": "Shift de comportamento: Matinal -> Noturno",
            "shift_day": CONFIG['shift_day'],
            "fase1": "Dias 1-30: Perfil matinal (FBM alto 6-11h)",
            "fase2": "Dias 31-90: Perfil noturno (FBM alto 18-23h)"
        },
        "daily_results": simulator.daily_results,
        "summary": {
            "total_notifications": simulator.total_notifications,
            "total_actions": simulator.total_actions,
            "vp": simulator.vp_count,
            "vn": simulator.vn_count,
            "fp": simulator.fp_count,
            "fn": simulator.fn_count,
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score'],
            "hourly_stats": simulator.hourly_stats,
            "top_hours": metrics['top_hours'],
            "fbm_metrics": metrics['fbm_metrics'],
            "period_metrics": metrics['period_metrics'],
            "phase_metrics": metrics['phase_metrics'],
            "shift_detected": metrics['shift_detected'],
            "shift_detection_day": metrics['shift_detection_day']
        }
    }
    
    output_file = output_dir / "shift_behavior_simulation_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dados salvos em: {output_file}")
    
    print("\n" + "=" * 100)
    print("🏁 SIMULAÇÃO CONCLUÍDA!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    run_shift_behavior_simulation()
