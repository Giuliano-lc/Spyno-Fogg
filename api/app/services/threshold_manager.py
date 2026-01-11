"""Gerenciador de Threshold Dinâmico para FBM."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class FeedbackType(str, Enum):
    VP = "VP"
    VN = "VN"
    FP = "FP"
    FN = "FN"


@dataclass
class ThresholdAdjustment:
    """Registro de ajuste de threshold."""
    timestamp: str
    hour: int
    feedback_type: str
    notified: bool
    executed: bool
    fbm_score: float
    old_threshold: float
    new_threshold: float
    adjustment: float


@dataclass
class ThresholdState:
    """Estado atual do threshold de um usuário."""
    user_id: str
    current_threshold: float
    initial_threshold: float
    min_threshold: float
    max_threshold: float
    adjustment_step: float
    total_adjustments: int
    vp_count: int
    vn_count: int
    fp_count: int
    fn_count: int
    last_updated: str
    history: List[Dict]


class ThresholdManager:
    """Gerencia threshold dinâmico para decisões de notificação."""
    
    DEFAULT_INITIAL_THRESHOLD = 40.0
    DEFAULT_MIN_THRESHOLD = 5.0
    DEFAULT_MAX_THRESHOLD = 80.0
    DEFAULT_ADJUSTMENT_STEP = 2.0
    
    def __init__(self, data_dir: str = "data/thresholds"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, ThresholdState] = {}
    
    def _get_state_path(self, user_id: str) -> Path:
        return self.data_dir / f"{user_id}_threshold.json"
    
    def _get_history_path(self, user_id: str) -> Path:
        return self.data_dir / f"{user_id}_threshold_history.json"
    
    def get_state(self, user_id: str) -> ThresholdState:
        """Obtém o estado atual do threshold."""
        
        if user_id in self._cache:
            return self._cache[user_id]
        
        state_path = self._get_state_path(user_id)
        
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                state = ThresholdState(**data)
        else:
            state = ThresholdState(
                user_id=user_id,
                current_threshold=self.DEFAULT_INITIAL_THRESHOLD,
                initial_threshold=self.DEFAULT_INITIAL_THRESHOLD,
                min_threshold=self.DEFAULT_MIN_THRESHOLD,
                max_threshold=self.DEFAULT_MAX_THRESHOLD,
                adjustment_step=self.DEFAULT_ADJUSTMENT_STEP,
                total_adjustments=0,
                vp_count=0,
                vn_count=0,
                fp_count=0,
                fn_count=0,
                last_updated=datetime.now().isoformat(),
                history=[]
            )
            self._save_state(state)
        
        self._cache[user_id] = state
        return state
    
    def _save_state(self, state: ThresholdState) -> None:
        state_path = self._get_state_path(state.user_id)
        
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2, ensure_ascii=False)
        
        self._cache[state.user_id] = state
    
    def _save_history(self, user_id: str, adjustment: ThresholdAdjustment) -> None:
        history_path = self._get_history_path(user_id)
        
        history = []
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        
        history.append(asdict(adjustment))
        
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def get_threshold(self, user_id: str) -> float:
        state = self.get_state(user_id)
        return state.current_threshold
    
    def classify_feedback(self, notified: bool, executed: bool) -> FeedbackType:
        if notified and executed:
            return FeedbackType.VP
        elif notified and not executed:
            return FeedbackType.VN
        elif not notified and executed:
            return FeedbackType.FP
        else:
            return FeedbackType.FN
    
    def calculate_adjustment(
        self, 
        feedback_type: FeedbackType, 
        fbm_score: float,
        current_threshold: float,
        step: float
    ) -> float:
        """
        Calcula o ajuste PROPORCIONAL à distância entre FBM e threshold.
        
        VP (notificou e respondeu): Threshold SOBE levemente
        VN (notificou mas ignorou): Threshold SOBE (penalização)
        FP (não notificou mas agiu): Threshold DESCE
        FN (não notificou e não agiu): Mantém
        """
        margin = fbm_score - current_threshold
        
        if feedback_type == FeedbackType.VP:
            if margin <= 3:
                return 0.0
            elif margin <= 8:
                return margin * 0.1
            elif margin <= 15:
                return margin * 0.15
            else:
                return min(margin * 0.2, 5.0)
                
        elif feedback_type == FeedbackType.VN:
            if margin <= 5:
                return step * 1.0
            elif margin <= 15:
                return margin * 0.12
            else:
                return min(margin * 0.18, 5.0)
            
        elif feedback_type == FeedbackType.FP:
            return -step * 2
            
        else:
            return 0.0
    
    def update_threshold(
        self,
        user_id: str,
        hour: int,
        notified: bool,
        executed: bool,
        fbm_score: float
    ) -> Dict[str, Any]:
        """Atualiza threshold baseado no feedback."""
        state = self.get_state(user_id)
        old_threshold = state.current_threshold
        
        # Classifica feedback
        feedback_type = self.classify_feedback(notified, executed)
        
        # Calcula ajuste (agora considera FBM e threshold atual)
        adjustment = self.calculate_adjustment(
            feedback_type, 
            fbm_score,
            state.current_threshold,
            state.adjustment_step
        )
        
        # Aplica ajuste com limites
        new_threshold = old_threshold + adjustment
        new_threshold = max(state.min_threshold, min(state.max_threshold, new_threshold))
        
        # Atualiza contadores
        if feedback_type == FeedbackType.VP:
            state.vp_count += 1
        elif feedback_type == FeedbackType.VN:
            state.vn_count += 1
        elif feedback_type == FeedbackType.FP:
            state.fp_count += 1
        else:
            state.fn_count += 1
        
        # Atualiza estado
        state.current_threshold = new_threshold
        state.total_adjustments += 1
        state.last_updated = datetime.now().isoformat()
        
        # Cria registro do ajuste
        adjustment_record = ThresholdAdjustment(
            timestamp=datetime.now().isoformat(),
            hour=hour,
            feedback_type=feedback_type.value,
            notified=notified,
            executed=executed,
            fbm_score=fbm_score,
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            adjustment=adjustment
        )
        
        # Adiciona ao histórico resumido (últimos 100)
        state.history.append({
            "timestamp": adjustment_record.timestamp,
            "feedback": feedback_type.value,
            "threshold": new_threshold
        })
        if len(state.history) > 100:
            state.history = state.history[-100:]
        
        # Salva
        self._save_state(state)
        self._save_history(user_id, adjustment_record)
        
        return {
            "feedback_type": feedback_type.value,
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "adjustment": adjustment,
            "fbm_score": fbm_score,
            "stats": {
                "vp": state.vp_count,
                "vn": state.vn_count,
                "fp": state.fp_count,
                "fn": state.fn_count,
                "total": state.total_adjustments
            }
        }
    
    def should_notify(self, user_id: str, fbm_score: float) -> Dict[str, Any]:
        """
        Decide se deve notificar baseado no FBM score e threshold.
        
        Args:
            user_id: ID do usuário
            fbm_score: Score FBM calculado
            
        Returns:
            Dict com decisão e informações
        """
        threshold = self.get_threshold(user_id)
        should = fbm_score >= threshold
        
        return {
            "should_notify": should,
            "fbm_score": fbm_score,
            "threshold": threshold,
            "margin": fbm_score - threshold,
            "confidence": min(1.0, abs(fbm_score - threshold) / threshold) if threshold > 0 else 1.0
        }
    
    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        """Retorna estatísticas do threshold para um usuário."""
        state = self.get_state(user_id)
        
        total = state.vp_count + state.vn_count + state.fp_count + state.fn_count
        
        # Calcula taxas
        precision = state.vp_count / (state.vp_count + state.vn_count) if (state.vp_count + state.vn_count) > 0 else 0
        recall = state.vp_count / (state.vp_count + state.fp_count) if (state.vp_count + state.fp_count) > 0 else 0
        accuracy = (state.vp_count + state.fn_count) / total if total > 0 else 0
        
        return {
            "user_id": user_id,
            "current_threshold": state.current_threshold,
            "initial_threshold": state.initial_threshold,
            "threshold_change": state.current_threshold - state.initial_threshold,
            "total_events": total,
            "counts": {
                "vp": state.vp_count,
                "vn": state.vn_count,
                "fp": state.fp_count,
                "fn": state.fn_count
            },
            "rates": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "accuracy": round(accuracy, 3)
            },
            "last_updated": state.last_updated
        }
    
    def get_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Retorna o histórico de ajustes."""
        history_path = self._get_history_path(user_id)
        
        if not history_path.exists():
            return []
        
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        return history[-limit:]
    
    def reset_threshold(self, user_id: str) -> Dict[str, Any]:
        """Reseta o threshold para o valor inicial."""
        state = self.get_state(user_id)
        old_threshold = state.current_threshold
        
        state.current_threshold = state.initial_threshold
        state.last_updated = datetime.now().isoformat()
        
        self._save_state(state)
        
        return {
            "old_threshold": old_threshold,
            "new_threshold": state.initial_threshold,
            "reset": True
        }
    
    def set_threshold(
        self, 
        user_id: str, 
        threshold: float,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
        adjustment_step: Optional[float] = None
    ) -> Dict[str, Any]:
        """Define manualmente o threshold e parâmetros."""
        state = self.get_state(user_id)
        old_threshold = state.current_threshold
        
        state.current_threshold = threshold
        
        if min_threshold is not None:
            state.min_threshold = min_threshold
        if max_threshold is not None:
            state.max_threshold = max_threshold
        if adjustment_step is not None:
            state.adjustment_step = adjustment_step
        
        state.last_updated = datetime.now().isoformat()
        self._save_state(state)
        
        return {
            "old_threshold": old_threshold,
            "new_threshold": threshold,
            "min": state.min_threshold,
            "max": state.max_threshold,
            "step": state.adjustment_step
        }
