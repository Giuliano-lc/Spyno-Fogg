"""Serviço de armazenamento do histórico de usuários."""

import json
import os
from datetime import date
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.models import DailyData


class StorageService:
    def __init__(self, data_dir: str = "data/users"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_user_file(self, user_id: str) -> Path:
        safe_user_id = user_id.replace("/", "_").replace("\\", "_")
        return self.data_dir / f"{safe_user_id}.json"
    
    def _get_last_training_status(self, history: Dict[str, Any]) -> bool:
        if not history["days"]:
            return True
        
        for day in reversed(history["days"]):
            for hour in reversed(day["hours"]):
                fb = hour["feedback"]
                if fb.get("action_performed") and fb.get("training_feedback"):
                    return fb["training_feedback"].get("completed_fully", True)
        
        return True
    
    def _count_total_activities(self, history: Dict[str, Any]) -> int:
        count = 0
        for day in history["days"]:
            for hour in day["hours"]:
                if hour["feedback"].get("action_performed"):
                    count += 1
        return count
    
    def _calculate_confidence_from_history(
        self, 
        total_activities: int, 
        last_completed: bool,
        base_confidence: int = 6
    ) -> int:
        """Calcula confidence_score: base + (atividades/3) + last_completed."""
        confidence = base_confidence
        confidence += min(total_activities // 3, 2)
        if last_completed:
            confidence += 1
        return max(0, min(10, confidence))
    
    def _enrich_daily_data(self, daily_data: DailyData) -> None:
        """Enriquece dados com valores calculados do histórico."""
        history = self.get_user_history(daily_data.user_id)
        last_completed = self._get_last_training_status(history)
        total_activities = self._count_total_activities(history)
        
        for hour_data in daily_data.hours:
            mf = hour_data.motivation_factors
            af = hour_data.ability_factors
            
            if mf.last_activity_score is None:
                mf.last_activity_score = 1 if last_completed else 0
            
            if af.confidence_score is None:
                af.confidence_score = self._calculate_confidence_from_history(
                    total_activities=total_activities,
                    last_completed=last_completed
                )
    
    def get_user_history(self, user_id: str) -> Dict[str, Any]:
        user_file = self._get_user_file(user_id)
        
        if not user_file.exists():
            return {
                "user_id": user_id,
                "user_profile": None,
                "days": [],
                "total_days": 0,
                "metrics": {
                    "total_notifications_sent": 0,
                    "total_actions_performed": 0,
                    "vp_count": 0,
                    "vn_count": 0,
                    "fp_count": 0,
                    "fn_count": 0
                }
            }
        
        with open(user_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def append_daily_data(self, daily_data: DailyData) -> Dict[str, Any]:
        """Concatena novo dia ao histórico e retorna histórico atualizado."""
        self._enrich_daily_data(daily_data)
        history = self.get_user_history(daily_data.user_id)
        
        existing_dates = [d["date"] for d in history["days"]]
        date_str = daily_data.date.isoformat()
        
        if date_str in existing_dates:
            raise ValueError(f"Já existem dados para {date_str}")
        
        history["user_profile"] = daily_data.user_profile.model_dump()
        fbm_scores = daily_data.calculate_fbm_scores()
        day_metrics = self._calculate_day_metrics(daily_data, fbm_scores)
        
        day_data = {
            "date": date_str,
            "timezone": daily_data.timezone,
            "hours": [h.model_dump() for h in daily_data.hours],
            "fbm_scores": fbm_scores,
            "day_metrics": day_metrics
        }
        
        history["days"].append(day_data)
        history["total_days"] = len(history["days"])
        history["metrics"]["total_notifications_sent"] += day_metrics["notifications_sent"]
        history["metrics"]["total_actions_performed"] += day_metrics["actions_performed"]
        history["metrics"]["vp_count"] += day_metrics["vp"]
        history["metrics"]["vn_count"] += day_metrics["vn"]
        history["metrics"]["fp_count"] += day_metrics["fp"]
        history["metrics"]["fn_count"] += day_metrics["fn"]
        
        self._save_history(daily_data.user_id, history)
        return history
    
    def _calculate_day_metrics(
        self, 
        daily_data: DailyData, 
        fbm_scores: List[dict]
    ) -> Dict[str, int]:
        """Calcula VP, VN, FP, FN do dia."""
        vp = vn = fp = fn = 0
        notifications_sent = 0
        actions_performed = 0
        
        for hour_data in daily_data.hours:
            fb = hour_data.feedback
            notified = fb.notification_sent
            acted = fb.action_performed
            
            if notified:
                notifications_sent += 1
            if acted:
                actions_performed += 1
            
            if notified and acted:
                vp += 1
            elif notified and not acted:
                vn += 1
            elif not notified and acted:
                fp += 1
            elif not notified and not acted:
                fn += 1
        
        return {
            "notifications_sent": notifications_sent,
            "actions_performed": actions_performed,
            "vp": vp,
            "vn": vn,
            "fp": fp,
            "fn": fn
        }
    
    def _save_history(self, user_id: str, history: Dict[str, Any]) -> None:
        user_file = self._get_user_file(user_id)
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    
    def get_training_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Retorna dados formatados para treinamento RL."""
        history = self.get_user_history(user_id)
        training_data = []
        
        for day in history["days"]:
            for i, hour in enumerate(day["hours"]):
                fbm = day["fbm_scores"][i]
                
                observation = {
                    "hour": hour["hour"],
                    "day_period": hour["context"]["day_period"],
                    "is_weekend": hour["context"]["is_weekend"],
                    "motivation": fbm["motivation"],
                    "ability": fbm["ability"],
                    "trigger": fbm["trigger"],
                    "fbm_score": fbm["fbm_score"],
                    "sleeping": fbm["sleeping"]
                }
                
                action = 1 if hour["feedback"]["notification_sent"] else 0
                reward = self._calculate_reward(hour["feedback"])
                
                training_data.append({
                    "date": day["date"],
                    "observation": observation,
                    "action": action,
                    "reward": reward,
                    "action_performed": hour["feedback"]["action_performed"]
                })
        
        return training_data
    
    def _calculate_reward(self, feedback: dict) -> float:
        """
        Rewards: VP=+20, VN=-15, FP=-10, FN=0
        
        Histórico:
        - v1 (VN=-15, FP=0): Spam (precision 29%)
        - v2 (VN=-25, FP=0): Conservador (recall 30%)
        - v3 (VN=-20, FP=0): Paralisia total
        - v4 (VN=-15, FP=-10): Balanço atual
        """
        notified = feedback["notification_sent"]
        acted = feedback["action_performed"]
        
        if notified and acted:
            return 20.0
        elif notified and not acted:
            return -15.0
        elif not notified and acted:
            return -10.0
        else:
            return 0.0
    
    def list_users(self) -> List[str]:
        users = []
        for file in self.data_dir.glob("*.json"):
            users.append(file.stem)
        return users
    
    def delete_user_history(self, user_id: str) -> bool:
        user_file = self._get_user_file(user_id)
        if user_file.exists():
            user_file.unlink()
            return True
        return False
