"""
Modelos Pydantic para validação do JSON diário.
Baseado no Fogg Behavior Model (FBM) conforme paper CAPABLE.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date as date_type


class TrainingFeedback(BaseModel):
    difficulty_level: int = Field(..., ge=1, le=5)
    familiarity_level: int = Field(..., ge=1, le=5)
    completed_fully: Optional[bool] = None
    duration_minutes: Optional[int] = Field(None, ge=0)


class MotivationFactors(BaseModel):
    """Fatores de motivação (M)."""
    valence: int = Field(..., ge=0, le=10)
    last_activity_score: Optional[int] = Field(None, ge=0, le=10)
    hours_slept_last_night: int = Field(..., ge=0, le=12)


class AbilityFactors(BaseModel):
    """Fatores de habilidade (A)."""
    cognitive_load: int = Field(..., ge=0, le=10)
    activities_performed_today: int = Field(..., ge=0)
    time_since_last_activity_hours: int = Field(..., ge=0, le=48)
    confidence_score: Optional[int] = Field(None, ge=0, le=10)


class TriggerFactors(BaseModel):
    """Fatores de gatilho (T)."""
    sleeping: bool
    arousal: int = Field(..., ge=0, le=10)
    location: Literal["home", "work", "other"]
    motion_activity: Literal["stationary", "walking", "running"]


class Context(BaseModel):
    day_period: int = Field(..., ge=0, le=3)  # 0=manhã, 1=meio-dia, 2=noite, 3=madrugada
    is_weekend: bool


class Feedback(BaseModel):
    notification_sent: bool
    action_performed: bool
    training_feedback: Optional[TrainingFeedback] = None
    
    @model_validator(mode='after')
    def validate_training_feedback(self):
        if self.training_feedback is not None and not self.action_performed:
            raise ValueError("training_feedback requer action_performed=true")
        return self


class HourData(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    motivation_factors: MotivationFactors
    ability_factors: AbilityFactors
    trigger_factors: TriggerFactors
    context: Context
    feedback: Feedback


class UserProfile(BaseModel):
    has_family: bool


class DailyData(BaseModel):
    """Schema do JSON diário."""
    user_id: str = Field(..., min_length=1)
    date: date_type
    timezone: str
    user_profile: UserProfile
    hours: List[HourData] = Field(..., min_length=24, max_length=24)
    
    @field_validator('hours')
    @classmethod
    def validate_hours_sequence(cls, v: List[HourData]) -> List[HourData]:
        hours_found = [h.hour for h in v]
        expected = list(range(24))
        if sorted(hours_found) != expected:
            missing = set(expected) - set(hours_found)
            raise ValueError(f"Horas faltando: {missing}")
        return v
    
    def calculate_fbm_scores(self) -> List[dict]:
        """Calcula scores FBM (M, A, T) para cada hora."""
        results = []
        
        for hour_data in self.hours:
            mf = hour_data.motivation_factors
            af = hour_data.ability_factors
            tf = hour_data.trigger_factors
            ctx = hour_data.context
            
            # M: 0-4
            valence_score = 1 if mf.valence == 1 else 0
            family_score = 1 if self.user_profile.has_family else 0
            benefit_score = 1 if (mf.last_activity_score is not None and mf.last_activity_score == 1) else 0
            sleep_score = 1 if mf.hours_slept_last_night >= 7 else 0
            motivation = valence_score + family_score + benefit_score + sleep_score
            
            # A: 0-4
            load_score = 1 if af.cognitive_load == 0 else 0
            strain_score = 1 if af.activities_performed_today <= 1 else 0
            ready_score = 1 if af.time_since_last_activity_hours >= 1 else 0
            conf_score = 1 if (af.confidence_score is not None and af.confidence_score >= 4) else 0
            ability = load_score + strain_score + ready_score + conf_score
            
            # T: 0-6
            if tf.sleeping:
                trigger = 0
            else:
                awake_score = 1
                arousal_score = 1 if tf.arousal == 1 else 0
                location_score = 1 if tf.location == "home" else 0
                motion_score = 1 if tf.motion_activity == "stationary" else 0
                time_score = 1 if ctx.day_period == 1 else 0
                day_score = 1 if ctx.is_weekend else 0
                trigger = (awake_score + arousal_score + location_score + 
                          motion_score + time_score + day_score)
            
            fbm_score = motivation * ability * trigger
            
            results.append({
                "hour": hour_data.hour,
                "motivation": motivation,
                "ability": ability,
                "trigger": trigger,
                "fbm_score": fbm_score,
                "sleeping": tf.sleeping,
                "notification_sent": hour_data.feedback.notification_sent,
                "action_performed": hour_data.feedback.action_performed
            })
        
        return results
