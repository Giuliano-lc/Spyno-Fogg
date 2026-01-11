"""
Gerador de dados sintéticos para testes.
Cria perfis de usuário realistas com variação nos componentes FBM.
"""

import random
from datetime import date, timedelta
from typing import Dict, List, Any, Optional
import json


class UserProfile:
    """Perfil base de usuário com preferências e padrões comportamentais."""
    
    def __init__(
        self,
        name: str,
        preferred_hours: List[int],        # Horas preferidas para treino
        sleep_start: int = 23,              # Hora de dormir
        sleep_end: int = 7,                 # Hora de acordar
        work_start: int = 9,                # Início trabalho
        work_end: int = 18,                 # Fim trabalho
        base_motivation: float = 0.7,       # Motivação base (0-1)
        base_confidence: int = 6,           # Confiança base (0-10)
        exercise_probability: float = 0.8,  # Probabilidade de treinar se notificado na hora certa
        has_family: bool = True,
        description: str = ""
    ):
        self.name = name
        self.preferred_hours = preferred_hours
        self.sleep_start = sleep_start
        self.sleep_end = sleep_end
        self.work_start = work_start
        self.work_end = work_end
        self.base_motivation = base_motivation
        self.base_confidence = base_confidence
        self.exercise_probability = exercise_probability
        self.has_family = has_family
        self.description = description


# Perfil MATINAL - Pessoa que prefere treinar pela manhã
PERFIL_MATINAL = UserProfile(
    name="matinal",
    preferred_hours=[6, 7, 8],              # Prefere treinar cedo (6h-8h)
    sleep_start=22,                          # Dorme cedo (22h)
    sleep_end=6,                             # Acorda cedo (6h)
    work_start=8,                            # Começa trabalho 8h
    work_end=17,                             # Termina 17h
    base_motivation=0.8,                     # Alta motivação pela manhã
    base_confidence=7,                       # Boa confiança
    exercise_probability=0.85,               # Alta chance de treinar se notificado certo
    has_family=True,
    description="Usuário matinal: acorda cedo, prefere treinar entre 6h-8h, alta motivação pela manhã"
)


class SyntheticDataGenerator:
    """Gera dados sintéticos realistas baseado em perfil de usuário."""
    
    def __init__(self, profile: UserProfile, seed: Optional[int] = None):
        self.profile = profile
        if seed is not None:
            random.seed(seed)
    
    def is_sleeping(self, hour: int) -> bool:
        """Determina se o usuário está dormindo."""
        if self.profile.sleep_start > self.profile.sleep_end:
            # Dorme à noite e acorda de manhã (ex: 22h - 6h)
            return hour >= self.profile.sleep_start or hour < self.profile.sleep_end
        else:
            # Caso atípico
            return self.profile.sleep_start <= hour < self.profile.sleep_end
    
    def is_at_work(self, hour: int, is_weekend: bool) -> bool:
        """Determina se está no trabalho."""
        if is_weekend:
            return False
        return self.profile.work_start <= hour < self.profile.work_end
    
    def get_day_period(self, hour: int) -> int:
        """Retorna período do dia: 0=manhã, 1=meio-dia, 2=noite, 3=madrugada."""
        if 6 <= hour < 10:
            return 0
        elif 10 <= hour < 18:
            return 1
        elif 18 <= hour < 22:
            return 2
        else:
            return 3
    
    def calculate_valence(self, hour: int, hours_slept: int, activities_done: int) -> int:
        """
        Calcula valência (estado emocional).
        Varia baseado em: sono, hora do dia, atividades realizadas.
        """
        base_prob = 0.5
        
        # Sono suficiente aumenta valência
        if hours_slept >= 7:
            base_prob += 0.2
        elif hours_slept < 5:
            base_prob -= 0.2
        
        # Hora preferida do perfil aumenta valência
        if hour in self.profile.preferred_hours:
            base_prob += 0.2
        
        # Ter feito atividade melhora humor
        if activities_done > 0:
            base_prob += 0.15
        
        # Variação aleatória
        base_prob += random.uniform(-0.1, 0.1)
        
        return 1 if random.random() < base_prob else 0
    
    def calculate_cognitive_load(self, hour: int, is_weekend: bool) -> int:
        """
        Calcula carga cognitiva.
        Alta no trabalho, baixa em casa/fim de semana.
        """
        if self.is_sleeping(hour):
            return 0
        
        if is_weekend:
            return 0 if random.random() < 0.8 else 1
        
        if self.is_at_work(hour, is_weekend):
            # No trabalho: 70% chance de carga alta
            return 1 if random.random() < 0.7 else 0
        
        # Em casa: 20% chance de carga alta
        return 1 if random.random() < 0.2 else 0
    
    def calculate_arousal(self, hour: int) -> int:
        """
        Calcula nível de arousal (ativação).
        0=baixo, 1=médio (ideal), 2=alto.
        """
        if self.is_sleeping(hour):
            return 0
        
        # Arousal varia ao longo do dia
        if hour in self.profile.preferred_hours:
            # Nas horas preferidas, arousal tende a ser médio (ideal)
            return 1 if random.random() < 0.7 else random.choice([0, 2])
        elif 12 <= hour <= 14:
            # Pós-almoço: arousal baixo
            return 0 if random.random() < 0.5 else 1
        elif 9 <= hour <= 11:
            # Manhã: pode ser alto (estresse trabalho)
            return random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0]
        else:
            return random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
    
    def calculate_confidence(self, activities_total: int, last_completed: bool) -> int:
        """
        Calcula confiança/autoeficácia.
        Aumenta com sucesso, diminui com falha.
        """
        confidence = self.profile.base_confidence
        
        # Histórico de atividades aumenta confiança
        confidence += min(activities_total // 3, 2)
        
        # Última atividade completada aumenta confiança
        if last_completed:
            confidence += 1
        
        # Variação aleatória
        confidence += random.randint(-1, 1)
        
        return max(0, min(10, confidence))
    
    def should_exercise(self, hour: int, was_notified: bool, fbm_score: int) -> bool:
        """
        Determina se o usuário vai executar o exercício.
        Baseado em: hora preferida, notificação, FBM score.
        """
        if self.is_sleeping(hour):
            return False
        
        base_prob = 0.1  # Chance base sem notificação
        
        if was_notified:
            if hour in self.profile.preferred_hours:
                # Notificado na hora preferida: alta chance
                base_prob = self.profile.exercise_probability
            else:
                # Notificado em outra hora: chance moderada
                base_prob = self.profile.exercise_probability * 0.4
        else:
            # Não notificado mas pode fazer sozinho
            if hour in self.profile.preferred_hours:
                base_prob = 0.3  # Às vezes faz sozinho na hora preferida
        
        # FBM score influencia
        if fbm_score > 60:
            base_prob *= 1.2
        elif fbm_score < 30:
            base_prob *= 0.7
        
        return random.random() < base_prob
    
    def generate_hour_data(
        self,
        hour: int,
        target_date: date,
        hours_slept: int,
        activities_today: int,
        time_since_last: int,
        activities_total: int,
        last_completed: bool,
        notification_strategy: str = "fbm_based"  # "fbm_based", "smart", "random", "always", "never"
    ) -> Dict[str, Any]:
        """Gera dados para uma hora específica."""
        
        sleeping = self.is_sleeping(hour)
        is_weekend = target_date.weekday() >= 5
        
        # Motivation factors
        valence = 0 if sleeping else self.calculate_valence(hour, hours_slept, activities_today)
        
        # Ability factors
        cognitive_load = self.calculate_cognitive_load(hour, is_weekend)
        confidence = self.calculate_confidence(activities_total, last_completed)
        
        # Trigger factors
        arousal = self.calculate_arousal(hour)
        location = "home" if (sleeping or not self.is_at_work(hour, is_weekend)) else "work"
        
        # Motion varia
        if sleeping:
            motion = "stationary"
        elif random.random() < 0.1:
            motion = "walking"
        else:
            motion = "stationary"
        
        # Context
        day_period = self.get_day_period(hour)
        
        # Decide notificação baseado na estratégia
        if notification_strategy == "fbm_based":
            # Sistema decidirá baseado em FBM score e threshold
            notification_sent = None  # Será definido pelo simulador
        elif notification_strategy == "smart":
            # Notifica apenas nas horas preferidas
            notification_sent = hour in self.profile.preferred_hours and not sleeping
        elif notification_strategy == "random":
            notification_sent = random.random() < 0.15 and not sleeping
        elif notification_strategy == "always":
            notification_sent = not sleeping
        else:
            notification_sent = False
        
        # Calcula FBM para decidir se executa ação (simulação simplificada)
        # M
        m_valence = 1 if valence == 1 else 0
        m_family = 1 if self.profile.has_family else 0
        m_benefit = 1  # Assume benefício percebido
        m_sleep = 1 if hours_slept >= 7 else 0
        motivation = m_valence + m_family + m_benefit + m_sleep
        
        # A
        a_load = 1 if cognitive_load == 0 else 0
        a_strain = 1 if activities_today <= 1 else 0
        a_ready = 1 if time_since_last >= 1 else 0
        a_conf = 1 if confidence >= 4 else 0
        ability = a_load + a_strain + a_ready + a_conf
        
        # T
        if sleeping:
            trigger = 0
        else:
            t_awake = 1
            t_arousal = 1 if arousal == 1 else 0
            t_location = 1 if location == "home" else 0
            t_motion = 1 if motion == "stationary" else 0
            # CORREÇÃO: Favorecer horas preferidas do perfil ao invés de meio-dia genérico
            t_time = 1 if hour in self.profile.preferred_hours else 0
            t_day = 1 if is_weekend else 0
            trigger = t_awake + t_arousal + t_location + t_motion + t_time + t_day
        
        fbm_score = motivation * ability * trigger
        
        # Decide se executa ação (se notification_sent for None, será decidido depois)
        if notification_sent is None:
            # Sistema decidirá baseado em FBM
            action_performed = None
        else:
            action_performed = self.should_exercise(hour, notification_sent, fbm_score)
        
        # Training feedback (será gerado pelo simulador se action_performed for True)
        training_feedback = None
        if action_performed is True:
            training_feedback = {
                "difficulty_level": random.randint(2, 4),
                "familiarity_level": min(5, 3 + activities_total // 5),
                "completed_fully": random.random() < 0.85,
                "duration_minutes": random.randint(20, 45)
            }
        
        return {
            "hour": hour,
            "motivation_factors": {
                "valence": valence,
                "last_activity_score": 1 if last_completed else 0,
                "hours_slept_last_night": hours_slept
            },
            "ability_factors": {
                "cognitive_load": cognitive_load,
                "activities_performed_today": activities_today,
                "time_since_last_activity_hours": time_since_last,
                "confidence_score": confidence
            },
            "trigger_factors": {
                "sleeping": sleeping,
                "arousal": arousal,
                "location": location,
                "motion_activity": motion
            },
            "context": {
                "day_period": day_period,
                "is_weekend": is_weekend
            },
            "feedback": {
                "notification_sent": notification_sent,
                "action_performed": action_performed,
                "training_feedback": training_feedback
            },
            "_debug": {
                "fbm_score": fbm_score,
                "motivation": motivation,
                "ability": ability,
                "trigger": trigger
            }
        }
    
    def generate_day(
        self,
        user_id: str,
        target_date: date,
        previous_activities_total: int = 0,
        last_completed: bool = True,
        notification_strategy: str = "smart"
    ) -> Dict[str, Any]:
        """Gera dados completos de um dia."""
        
        hours = []
        activities_today = 0
        time_since_last = 12  # Assume 12h desde última atividade
        hours_slept = random.randint(6, 8)  # Varia sono
        
        for hour in range(24):
            hour_data = self.generate_hour_data(
                hour=hour,
                target_date=target_date,
                hours_slept=min(hour, hours_slept) if hour <= hours_slept else hours_slept,
                activities_today=activities_today,
                time_since_last=time_since_last,
                activities_total=previous_activities_total + activities_today,
                last_completed=last_completed,
                notification_strategy=notification_strategy
            )
            
            # Atualiza contadores
            if hour_data["feedback"]["action_performed"]:
                activities_today += 1
                time_since_last = 0
                if hour_data["feedback"]["training_feedback"]:
                    last_completed = hour_data["feedback"]["training_feedback"]["completed_fully"]
            else:
                time_since_last += 1
            
            # Remove debug info antes de retornar
            del hour_data["_debug"]
            hours.append(hour_data)
        
        return {
            "user_id": user_id,
            "date": target_date.isoformat(),
            "timezone": "America/Sao_Paulo",
            "user_profile": {
                "has_family": self.profile.has_family
            },
            "hours": hours
        }
    
    def generate_week(
        self,
        user_id: str,
        start_date: date,
        notification_strategy: str = "smart"
    ) -> List[Dict[str, Any]]:
        """Gera uma semana de dados."""
        
        days = []
        activities_total = 0
        last_completed = True
        
        for i in range(7):
            target_date = start_date + timedelta(days=i)
            day_data = self.generate_day(
                user_id=user_id,
                target_date=target_date,
                previous_activities_total=activities_total,
                last_completed=last_completed,
                notification_strategy=notification_strategy
            )
            days.append(day_data)
            
            # Conta atividades do dia
            for hour in day_data["hours"]:
                if hour["feedback"]["action_performed"]:
                    activities_total += 1
                    if hour["feedback"]["training_feedback"]:
                        last_completed = hour["feedback"]["training_feedback"]["completed_fully"]
        
        return days


def test_matinal_profile():
    """Testa o gerador com perfil matinal."""
    
    print("=" * 120)
    print(f"🌅 PERFIL: {PERFIL_MATINAL.name.upper()}")
    print(f"   {PERFIL_MATINAL.description}")
    print("=" * 120)
    
    generator = SyntheticDataGenerator(PERFIL_MATINAL, seed=42)
    
    # Gera um dia
    day_data = generator.generate_day(
        user_id="user_matinal_001",
        target_date=date.today(),
        notification_strategy="smart"
    )
    
    print(f"\n📅 Data: {day_data['date']} | Timezone: {day_data['timezone']}")
    print(f"👨‍👩‍👧 Família: {'Sim' if day_data['user_profile']['has_family'] else 'Não'}")
    
    # Cabeçalho detalhado
    print("\n" + "=" * 120)
    print("                    MOTIVAÇÃO              |        HABILIDADE           |       GATILHO        |   CONTEXTO  | FEEDBACK")
    print("-" * 120)
    print(f"{'Hora':^5}| Val | LastAct | Sono | {'CogLoad':^7} | ActHoje | TempUlt | Conf | Sleep | Arous | Loc  | Motion | Per | WE | Notif | Ação")
    print("-" * 120)
    
    notifications = 0
    actions = 0
    vp = fn = fp = vn = 0
    
    for h in day_data["hours"]:
        hour = h["hour"]
        mf = h["motivation_factors"]
        af = h["ability_factors"]
        tf = h["trigger_factors"]
        ctx = h["context"]
        fb = h["feedback"]
        
        # Indicadores visuais
        notif = "📱" if fb["notification_sent"] else "  "
        acao = "✅" if fb["action_performed"] else "  "
        sleeping = "💤" if tf["sleeping"] else "  "
        preferred = "⭐" if hour in PERFIL_MATINAL.preferred_hours else "  "
        weekend = "🏖️" if ctx["is_weekend"] else "  "
        
        # Contadores
        if fb["notification_sent"]:
            notifications += 1
        if fb["action_performed"]:
            actions += 1
        
        # Matriz de confusão
        if fb["notification_sent"] and fb["action_performed"]:
            vp += 1
        elif fb["notification_sent"] and not fb["action_performed"]:
            fn += 1
        elif not fb["notification_sent"] and fb["action_performed"]:
            fp += 1
        else:
            vn += 1
        
        # Período legível
        periodos = ["Manhã", "MeioDia", "Noite", "Madrug"]
        periodo = periodos[ctx["day_period"]]
        
        # Location abreviado
        loc_map = {"home": "Casa", "work": "Trab", "other": "Outr"}
        loc = loc_map.get(tf["location"], tf["location"])
        
        # Motion abreviado
        mot_map = {"stationary": "Parad", "walking": "Andnd", "running": "Crrnd"}
        mot = mot_map.get(tf["motion_activity"], tf["motion_activity"][:5])
        
        print(f" {hour:02d}{preferred}|  {mf['valence']}  |    {mf['last_activity_score']}    |  {mf['hours_slept_last_night']:2d}  |"
              f"    {af['cognitive_load']}    |    {af['activities_performed_today']}    |    {af['time_since_last_activity_hours']:2d}   |  {af['confidence_score']:2d}  |"
              f"  {sleeping}  |   {tf['arousal']}   | {loc:4} | {mot:6} |"
              f" {periodo[:3]:3} | {weekend} |  {notif}  |  {acao}")
        
        # Se teve treino, mostra detalhes
        if fb["training_feedback"]:
            tf_data = fb["training_feedback"]
            print(f"      └── 🏋️ Treino: Dificuldade={tf_data['difficulty_level']}/5, "
                  f"Familiaridade={tf_data['familiarity_level']}/5, "
                  f"Completo={'Sim' if tf_data['completed_fully'] else 'Não'}, "
                  f"Duração={tf_data['duration_minutes']}min")
    
    print("-" * 120)
    
    # Resumo detalhado
    print(f"\n{'='*50}")
    print("📊 RESUMO DETALHADO")
    print(f"{'='*50}")
    print(f"\n📱 Notificações:")
    print(f"   - Enviadas: {notifications}")
    print(f"   - Nas horas preferidas: {PERFIL_MATINAL.preferred_hours}")
    
    print(f"\n✅ Ações Executadas: {actions}")
    
    print(f"\n📈 Matriz de Confusão:")
    print(f"   - VP (notificou + executou): {vp}")
    print(f"   - VN (não notificou + não executou): {vn}")
    print(f"   - FP (não notificou + executou): {fp} <- oportunidades perdidas")
    print(f"   - FN (notificou + não executou): {fn} <- notificações ignoradas")
    
    if notifications > 0:
        print(f"\n📊 Métricas:")
        print(f"   - Taxa de sucesso: {vp/notifications*100:.1f}%")
        print(f"   - Precisão: {vp/(vp+fn)*100:.1f}%" if (vp+fn) > 0 else "")
    
    # Análise por período
    print(f"\n⏰ Ações por Período:")
    periodos_count = {"Manhã": 0, "MeioDia": 0, "Noite": 0, "Madrug": 0}
    periodos_map = {0: "Manhã", 1: "MeioDia", 2: "Noite", 3: "Madrug"}
    for h in day_data["hours"]:
        if h["feedback"]["action_performed"]:
            periodos_count[periodos_map[h["context"]["day_period"]]] += 1
    for p, c in periodos_count.items():
        if c > 0:
            print(f"   - {p}: {c} {'⭐' if p == 'Manhã' else ''}")
    
    return day_data


if __name__ == "__main__":
    day = test_matinal_profile()
    
    # Salva exemplo
    with open("tests/example_matinal_day.json", "w", encoding="utf-8") as f:
        json.dump(day, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Exemplo salvo em: tests/example_matinal_day.json")
