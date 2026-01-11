"""Ambiente Gymnasium para treinamento do modelo de notificações."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class NotificationEnv(gym.Env):
    """
    Ambiente de RL para decisões de notificação.
    
    Observation Space (8 dims): hour, day_period, is_weekend, motivation, ability, trigger, sleeping, notifications_today
    Action Space: 0=não notificar, 1=notificar
    
    Rewards:
    - VP (notificou e executou): +15 a +35 (proporcional ao FBM)
    - VN (notificou e ignorou): -5 a -50 (penalização forte)
    - FP (não notificou mas executou): -2
    - FN (não notificou e não executou): +0.5
    - Penalidade adicional -5 se notificar dormindo
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        training_data: List[Dict[str, Any]] = None,
        max_notifications_per_day: int = 5
    ):
        super().__init__()
        
        self.training_data = training_data or []
        self.max_notifications_per_day = max_notifications_per_day
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([23, 3, 1, 4, 4, 6, 1, 10], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(2)
        
        self.current_step = 0
        self.current_day = 0
        self.notifications_today = 0
        self.total_reward = 0
        
        self.episode_log = []
    
    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.training_data):
            return np.zeros(8, dtype=np.float32)
        
        sample = self.training_data[self.current_step]
        obs = sample["observation"]
        
        return np.array([
            obs["hour"],
            obs["day_period"],
            1 if obs["is_weekend"] else 0,
            obs["motivation"],
            obs["ability"],
            obs["trigger"],
            1 if obs["sleeping"] else 0,
            min(self.notifications_today, 10)
        ], dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_day = 0
        self.notifications_today = 0
        self.total_reward = 0
        self.episode_log = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.current_step >= len(self.training_data):
            return self._get_observation(), 0.0, True, False, {}
        
        sample = self.training_data[self.current_step]
        obs = sample["observation"]
        actual_action = sample["action"]
        action_performed = sample["action_performed"]
        
        reward = 0.0
        result_type = ""
        fbm_score = obs.get("fbm_score", 0)
        
        if action == 1:
            self.notifications_today += 1
            
            if action_performed:
                if fbm_score >= 60:
                    reward = 35.0
                elif fbm_score >= 40:
                    reward = 25.0
                else:
                    reward = 15.0
                result_type = "VP"
            else:
                if fbm_score <= 10:
                    reward = -50.0
                elif fbm_score <= 30:
                    reward = -35.0
                elif fbm_score <= 40:
                    reward = -20.0
                elif fbm_score <= 60:
                    reward = -10.0
                else:
                    reward = -5.0
                result_type = "VN"
        else:  # Agente decide NÃO notificar
            if action_performed:
                # FP: Não notificou mas usuário FEZ - perdeu oportunidade
                # Penalidade leve pois não irritou o usuário
                reward = -2.0
                result_type = "FP"
            else:
                # FN: Não notificou e não faria mesmo - correto!
                reward = 0.5  # Pequeno reward por economizar notificação
                result_type = "FN"  # CORRIGIDO: era "VN" mas é FN
        
        # Penalidade por notificar dormindo
        if action == 1 and obs["sleeping"]:
            reward = -5.0
            result_type = "SLEEP_PENALTY"
        
        # Penalidade por excesso de notificações
        if self.notifications_today > self.max_notifications_per_day:
            reward -= 0.5
        
        self.total_reward += reward
        
        # Log
        self.episode_log.append({
            "step": self.current_step,
            "hour": obs["hour"],
            "action": action,
            "actual_action": actual_action,
            "action_performed": action_performed,
            "reward": reward,
            "result_type": result_type,
            "fbm_score": obs["fbm_score"]
        })
        
        # Avança para próximo step
        self.current_step += 1
        
        # Verifica se mudou de dia (a cada 24 horas)
        if self.current_step % 24 == 0:
            self.current_day += 1
            self.notifications_today = 0
        
        # Verifica se terminou
        terminated = self.current_step >= len(self.training_data)
        
        info = {
            "result_type": result_type,
            "notifications_today": self.notifications_today,
            "total_reward": self.total_reward
        }
        
        return self._get_observation(), reward, terminated, False, info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do episódio."""
        if not self.episode_log:
            return {}
        
        vp = sum(1 for l in self.episode_log if l["result_type"] == "VP")
        vn = sum(1 for l in self.episode_log if l["result_type"] == "VN")
        fp = sum(1 for l in self.episode_log if l["result_type"] == "FP")
        fn = sum(1 for l in self.episode_log if l["result_type"] == "FN")
        
        notifications_sent = sum(1 for l in self.episode_log if l["action"] == 1)
        
        return {
            "total_steps": len(self.episode_log),
            "total_reward": self.total_reward,
            "notifications_sent": notifications_sent,
            "vp": vp,
            "vn": vn,
            "fp": fp,
            "fn": fn,
            "precision": vp / (vp + fn) if (vp + fn) > 0 else 0,
            "recall": vp / (vp + fp) if (vp + fp) > 0 else 0
        }
    
    def predict_best_hours(self, day_observations: List[Dict]) -> List[int]:
        """
        Dado observações de um dia, retorna as horas com maior probabilidade de sucesso.
        Usado para prever os melhores horários para notificar.
        """
        hour_scores = []
        
        for obs in day_observations:
            if obs.get("sleeping", False):
                score = -1  # Nunca notificar dormindo
            else:
                # Score baseado em FBM
                fbm = obs.get("fbm_score", 0)
                hour_scores.append({
                    "hour": obs["hour"],
                    "score": fbm,
                    "motivation": obs.get("motivation", 0),
                    "ability": obs.get("ability", 0),
                    "trigger": obs.get("trigger", 0)
                })
        
        # Ordena por score e retorna top 3
        sorted_hours = sorted(hour_scores, key=lambda x: x["score"], reverse=True)
        return sorted_hours[:3]
