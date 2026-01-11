"""Trainer para modelo PPO de notificações."""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .environment import NotificationEnv


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
    
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            self.rewards.append(self.model.ep_info_buffer[-1].get("r", 0))
        return True


class RLTrainer:
    """Gerencia o treinamento incremental do modelo PPO."""
    
    def __init__(
        self,
        model_path: str = "models/ppo_notification",
        learning_rate: float = 0.0003,
        n_steps: int = 64,
        batch_size: int = 32,
        n_epochs: int = 10,
        gamma: float = 0.99,
        verbose: int = 0
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.verbose = verbose
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"🚀 GPU detectada: {torch.cuda.get_device_name(0)}")
        else:
            print(f"💻 Usando CPU para treinamento RL")
        
        self.model: Optional[PPO] = None
        self.training_history: List[Dict] = []
        self.all_training_data: List[Dict] = []
        
        self.predictions_log: List[Dict] = []
    
    def add_day_data(self, day_training_data: List[Dict[str, Any]]) -> None:
        self.all_training_data.extend(day_training_data)
    
    def train(self, total_timesteps: int = 1000) -> Dict[str, Any]:
        """Treina ou atualiza modelo com dados acumulados."""
        if not self.all_training_data:
            return {"error": "Sem dados para treinar"}
        
        env = NotificationEnv(training_data=self.all_training_data)
        
        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                n_steps=min(self.n_steps, len(self.all_training_data)),
                batch_size=min(self.batch_size, len(self.all_training_data)),
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                verbose=self.verbose,
                device=self.device
            )
        else:
            self.model.set_env(env)
        
        callback = TrainingCallback()
        self.model.learn(
            total_timesteps=min(total_timesteps, len(self.all_training_data)),
            callback=callback,
            reset_num_timesteps=False
        )
        
        eval_stats = self._evaluate()
        
        training_record = {
            "days_trained": len(self.all_training_data) // 24,
            "total_samples": len(self.all_training_data),
            "eval_stats": eval_stats
        }
        self.training_history.append(training_record)
        
        return training_record
    
    def _evaluate(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        
        env = NotificationEnv(training_data=self.all_training_data)
        obs, _ = env.reset()
        
        total_reward = 0
        predictions = []
        
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            predictions.append(int(action))
            
            if terminated or truncated:
                break
        
        stats = env.get_episode_stats()
        stats["model_predictions"] = sum(predictions)
        
        return stats
    
    def predict_next_day(self, day_observations: List[Dict]) -> Dict[str, Any]:
        """
        Prevê os melhores horários para notificar no próximo dia.
        
        Args:
            day_observations: Lista de 24 observações (uma por hora)
        
        Returns:
            Dict com horas recomendadas e scores
        """
        if self.model is None:
            # Sem modelo treinado, usa heurística baseada em FBM
            return self._heuristic_prediction(day_observations)
        
        predictions = []
        
        for obs_dict in day_observations:
            # Converte para array numpy
            obs = np.array([
                obs_dict["hour"],
                obs_dict["day_period"],
                1 if obs_dict["is_weekend"] else 0,
                obs_dict["motivation"],
                obs_dict["ability"],
                obs_dict["trigger"],
                1 if obs_dict["sleeping"] else 0,
                0  # notifications_today = 0 para previsão
            ], dtype=np.float32)
            
            # Predição do modelo
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Obtém probabilidades da ação
            obs_tensor = self.model.policy.obs_to_tensor(obs.reshape(1, -1))[0]
            action_probs = self.model.policy.get_distribution(obs_tensor).distribution.probs.detach().numpy()[0]
            
            predictions.append({
                "hour": obs_dict["hour"],
                "recommended_action": int(action),
                "notify_probability": float(action_probs[1]),
                "fbm_score": obs_dict.get("fbm_score", 0),
                "sleeping": obs_dict.get("sleeping", False)
            })
        
        # Filtra horas recomendadas para notificar
        recommended_hours = [
            p for p in predictions 
            if p["recommended_action"] == 1 and not p["sleeping"]
        ]
        
        # Ordena por probabilidade
        recommended_hours.sort(key=lambda x: x["notify_probability"], reverse=True)
        
        return {
            "all_predictions": predictions,
            "recommended_hours": [h["hour"] for h in recommended_hours],
            "top_3_hours": recommended_hours[:3],
            "total_recommended": len(recommended_hours)
        }
    
    def _heuristic_prediction(self, day_observations: List[Dict]) -> Dict[str, Any]:
        """Previsão heurística quando não há modelo treinado."""
        predictions = []
        
        for obs in day_observations:
            fbm_score = obs.get("fbm_score", 0)
            sleeping = obs.get("sleeping", False)
            
            # Heurística simples: notificar se FBM > 40 e não dormindo
            should_notify = fbm_score > 40 and not sleeping
            
            predictions.append({
                "hour": obs["hour"],
                "recommended_action": 1 if should_notify else 0,
                "notify_probability": min(fbm_score / 100, 1.0),
                "fbm_score": fbm_score,
                "sleeping": sleeping
            })
        
        recommended_hours = [p for p in predictions if p["recommended_action"] == 1]
        recommended_hours.sort(key=lambda x: x["fbm_score"], reverse=True)
        
        return {
            "all_predictions": predictions,
            "recommended_hours": [h["hour"] for h in recommended_hours],
            "top_3_hours": recommended_hours[:3],
            "total_recommended": len(recommended_hours),
            "method": "heuristic"
        }
    
    def save_model(self) -> str:
        """Salva o modelo treinado."""
        if self.model is None:
            return ""
        
        save_path = str(self.model_path)
        self.model.save(save_path)
        return save_path
    
    def load_model(self) -> bool:
        """Carrega modelo existente."""
        model_file = Path(str(self.model_path) + ".zip")
        if model_file.exists():
            # Cria ambiente dummy para carregar
            env = NotificationEnv(training_data=[])
            self.model = PPO.load(str(self.model_path), env=env, device=self.device)
            return True
        return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Retorna resumo do treinamento."""
        return {
            "total_days": len(self.all_training_data) // 24,
            "total_samples": len(self.all_training_data),
            "training_sessions": len(self.training_history),
            "history": self.training_history
        }
