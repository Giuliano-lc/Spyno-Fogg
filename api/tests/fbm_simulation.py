"""
Simulador de Notificações baseado em FBM e Threshold Dinâmico.

Este módulo:
1. Recebe dados sintéticos com FBM calculado
2. Usa ThresholdManager para decidir quando notificar
3. Simula resposta do usuário baseada nos níveis de FBM
4. Gera análises e métricas do comportamento
"""

import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.services.threshold_manager import ThresholdManager


@dataclass
class SimulationResult:
    """Resultado de uma simulação."""
    user_id: str
    total_hours: int
    total_days: int
    notifications_sent: int
    actions_performed: int
    vp_count: int
    vn_count: int
    fp_count: int
    fn_count: int
    initial_threshold: float
    final_threshold: float
    threshold_change: float
    precision: float
    recall: float
    accuracy: float
    f1_score: float
    avg_fbm_when_notified: float
    avg_fbm_when_responded: float
    avg_fbm_when_ignored: float
    hourly_stats: Dict[int, Dict[str, int]]
    threshold_evolution: List[Dict[str, Any]]


class FBMSimulator:
    """
    Simula notificações e respostas do usuário baseado em FBM.
    
    O sistema decide quando notificar comparando FBM score com threshold dinâmico.
    A resposta do usuário é probabilística baseada no FBM score.
    """
    
    def __init__(
        self, 
        user_id: str,
        initial_threshold: float = 15.0,
        seed: Optional[int] = None
    ):
        self.user_id = user_id
        self.threshold_manager = ThresholdManager()
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        # Define threshold inicial
        self.threshold_manager.set_threshold(user_id, initial_threshold)
        
        # Estatísticas
        self.notifications_sent = 0
        self.actions_performed = 0
        self.vp_count = 0
        self.vn_count = 0
        self.fp_count = 0
        self.fn_count = 0
        
        self.fbm_scores_notified = []
        self.fbm_scores_responded = []
        self.fbm_scores_ignored = []
        
        self.hourly_stats = {h: {"notified": 0, "responded": 0} for h in range(24)}
        self.threshold_evolution = []
    
    def calculate_fbm_from_hour(self, hour_data: Dict) -> int:
        """Calcula FBM score a partir dos dados de uma hora."""
        mf = hour_data.get("motivation_factors", {})
        af = hour_data.get("ability_factors", {})
        tf = hour_data.get("trigger_factors", {})
        ctx = hour_data.get("context", {})
        
        # Motivação (M)
        m_valence = 1 if mf.get("valence", 0) == 1 else 0
        m_family = 1  # Assumindo has_family=True
        m_benefit = 1 if mf.get("last_activity_score", 0) == 1 else 0
        m_sleep = 1 if mf.get("hours_slept_last_night", 0) >= 7 else 0
        motivation = m_valence + m_family + m_benefit + m_sleep
        
        # Habilidade (A)
        a_load = 1 if af.get("cognitive_load", 1) == 0 else 0
        a_strain = 1 if af.get("activities_performed_today", 0) <= 1 else 0
        a_ready = 1 if af.get("time_since_last_activity_hours", 0) >= 1 else 0
        a_conf = 1 if af.get("confidence_score", 0) >= 4 else 0
        ability = a_load + a_strain + a_ready + a_conf
        
        # Gatilho (T)
        sleeping = tf.get("sleeping", False)
        if sleeping:
            trigger = 0
        else:
            t_awake = 1
            t_arousal = 1 if tf.get("arousal", 0) == 1 else 0
            t_location = 1 if tf.get("location", "") == "home" else 0
            t_motion = 1 if tf.get("motion_activity", "") == "stationary" else 0
            # CORREÇÃO: Usar horas preferidas (6,7,8 para perfil matinal)
            hour = hour_data.get("hour", 0)
            t_time = 1 if hour in [6, 7, 8] else 0  # Perfil matinal
            t_day = 1 if ctx.get("is_weekend", False) else 0
            trigger = t_awake + t_arousal + t_location + t_motion + t_time + t_day
        
        fbm_score = motivation * ability * trigger
        
        return fbm_score
    
    def should_user_respond(self, fbm_score: int, was_notified: bool, hour: int) -> bool:
        """
        Simula se usuário responde (PERFIL MATINAL RÍGIDO).
        
        MODIFICAÇÃO PARA TESTE:
        - Usuário matinal responde MUITO BEM em 6-8h (pico matinal)
        - Usuário matinal responde MODERADAMENTE em 9-12h (manhã tardia)
        - Usuário matinal IGNORA QUASE SEMPRE fora de 6-12h
        - Objetivo: Forçar RL a aprender a focar em 6-8h
        
        Lógica:
        1. Pico matinal: 6-8h (95% resposta)
        2. Manhã tardia: 9-12h (45-60% resposta)
        3. Fora de 6-12h: ignora quase sempre (feedback negativo forte)
        """
        # Janela matinal com pico
        PEAK_HOURS = [6, 7, 8]  # Pico matinal (alta probabilidade)
        ACCEPTABLE_HOURS = [9, 10, 11, 12]  # Manhã tardia (média probabilidade)
        
        if not was_notified:
            # Usuário pode agir espontaneamente
            if hour in PEAK_HOURS:
                # Pico matinal: age muito espontaneamente
                if fbm_score >= 40:
                    return random.random() < 0.6
                else:
                    return random.random() < 0.3
            elif hour in ACCEPTABLE_HOURS:
                # Manhã tardia: age moderadamente
                if fbm_score >= 40:
                    return random.random() < 0.3
                else:
                    return random.random() < 0.1
            else:
                # Fora de 6-12h: quase nunca age
                if fbm_score >= 60:
                    return random.random() < 0.08
                else:
                    return random.random() < 0.02
        
        # Quando notificado: PERFIL MATINAL COM PICO 6-8h
        if hour in PEAK_HOURS:
            # PICO MATINAL (6-8h): Probabilidade MUITO ALTA
            if fbm_score >= 40:
                probability = 0.95  # Quase certeza
            elif fbm_score >= 30:
                probability = 0.85
            elif fbm_score >= 20:
                probability = 0.75
            else:
                probability = 0.60
        elif hour in ACCEPTABLE_HOURS:
            # MANHÃ TARDIA (9-12h): Probabilidade MÉDIA
            # RL deve preferir 6-8h, mas 9-12h ainda funciona
            if fbm_score >= 50:
                probability = 0.60  # Menos que 6-8h
            elif fbm_score >= 40:
                probability = 0.45
            elif fbm_score >= 30:
                probability = 0.30
            else:
                probability = 0.15
        else:
            # FORA DA MANHÃ (13-22h): QUASE SEMPRE IGNORA
            # Forçar feedback negativo (VN) para RL aprender
            if fbm_score >= 70:
                probability = 0.10  # Mesmo com FBM altíssimo
            elif fbm_score >= 60:
                probability = 0.07
            elif fbm_score >= 50:
                probability = 0.05
            elif fbm_score >= 40:
                probability = 0.03
            else:
                probability = 0.01  # Praticamente ignora
        
        return random.random() < probability
    
    def simulate_hour(self, hour_data: Dict) -> Dict[str, Any]:
        """
        Simula uma hora: decide notificação, calcula resposta, atualiza threshold.
        
        Returns:
            Dict com resultados da simulação
        """
        hour = hour_data["hour"]
        
        # Calcula FBM score
        fbm_score = self.calculate_fbm_from_hour(hour_data)
        
        # Pula horas dormindo
        sleeping = hour_data.get("trigger_factors", {}).get("sleeping", False)
        if sleeping:
            return {
                "hour": hour,
                "fbm_score": fbm_score,
                "sleeping": True,
                "notified": False,
                "responded": False,
                "threshold": self.threshold_manager.get_threshold(self.user_id)
            }
        
        # Decide se notifica baseado em threshold
        current_threshold = self.threshold_manager.get_threshold(self.user_id)
        should_notify = fbm_score >= current_threshold
        
        # Simula resposta do usuário (passa hora para considerar preferências)
        user_responded = self.should_user_respond(fbm_score, should_notify, hour)
        
        # Atualiza estatísticas
        if should_notify:
            self.notifications_sent += 1
            self.fbm_scores_notified.append(fbm_score)
            self.hourly_stats[hour]["notified"] += 1
        
        if user_responded:
            self.actions_performed += 1
            self.fbm_scores_responded.append(fbm_score)
            self.hourly_stats[hour]["responded"] += 1
        
        # Registra FBM quando ignorou
        if should_notify and not user_responded:
            self.fbm_scores_ignored.append(fbm_score)
        
        # Classifica feedback
        if should_notify and user_responded:
            self.vp_count += 1
            feedback_type = "VP"
        elif should_notify and not user_responded:
            self.vn_count += 1
            feedback_type = "VN"
        elif not should_notify and user_responded:
            self.fp_count += 1
            feedback_type = "FP"
        else:
            self.fn_count += 1
            feedback_type = "FN"
        
        # Atualiza threshold baseado no feedback
        threshold_update = self.threshold_manager.update_threshold(
            user_id=self.user_id,
            hour=hour,
            notified=should_notify,
            executed=user_responded,
            fbm_score=fbm_score
        )
        
        # Registra evolução do threshold
        self.threshold_evolution.append({
            "hour": hour,
            "fbm_score": fbm_score,
            "threshold": threshold_update["new_threshold"],
            "feedback": feedback_type,
            "adjustment": threshold_update["adjustment"]
        })
        
        return {
            "hour": hour,
            "fbm_score": fbm_score,
            "sleeping": False,
            "notified": should_notify,
            "responded": user_responded,
            "feedback_type": feedback_type,
            "threshold": threshold_update["new_threshold"],
            "threshold_adjustment": threshold_update["adjustment"]
        }
    
    def simulate_day(self, day_data: Dict) -> Dict[str, Any]:
        """Simula um dia completo."""
        day_results = []
        
        for hour_data in day_data["hours"]:
            result = self.simulate_hour(hour_data)
            day_results.append(result)
        
        return {
            "date": day_data["date"],
            "hours": day_results,
            "daily_stats": {
                "notifications": sum(1 for h in day_results if h["notified"]),
                "responses": sum(1 for h in day_results if h["responded"]),
                "vp": sum(1 for h in day_results if h.get("feedback_type") == "VP"),
                "vn": sum(1 for h in day_results if h.get("feedback_type") == "VN"),
                "fp": sum(1 for h in day_results if h.get("feedback_type") == "FP"),
                "fn": sum(1 for h in day_results if h.get("feedback_type") == "FN"),
            }
        }
    
    def simulate_multiple_days(self, days_data: List[Dict]) -> SimulationResult:
        """Simula múltiplos dias e retorna análise completa."""
        
        initial_threshold = self.threshold_manager.get_threshold(self.user_id)
        
        print(f"{'='*100}")
        print(f"🎯 SIMULAÇÃO FBM-BASED - Usuário: {self.user_id}")
        print(f"📊 Threshold Inicial: {initial_threshold:.2f}")
        print(f"📅 Total de dias: {len(days_data)}")
        print(f"{'='*100}\n")
        
        all_day_results = []
        
        for i, day_data in enumerate(days_data, 1):
            day_result = self.simulate_day(day_data)
            all_day_results.append(day_result)
            
            stats = day_result["daily_stats"]
            print(f"Dia {i:2d} ({day_data['date']}): "
                  f"Notif={stats['notifications']:2d}, Resp={stats['responses']:2d}, "
                  f"VP={stats['vp']:2d}, VN={stats['vn']:2d}, FP={stats['fp']:2d}, "
                  f"Threshold={self.threshold_manager.get_threshold(self.user_id):.2f}")
        
        final_threshold = self.threshold_manager.get_threshold(self.user_id)
        
        # Calcula métricas finais
        precision = self.vp_count / (self.vp_count + self.vn_count) if (self.vp_count + self.vn_count) > 0 else 0
        recall = self.vp_count / (self.vp_count + self.fp_count) if (self.vp_count + self.fp_count) > 0 else 0
        accuracy = (self.vp_count + self.fn_count) / (self.vp_count + self.vn_count + self.fp_count + self.fn_count)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_fbm_notified = sum(self.fbm_scores_notified) / len(self.fbm_scores_notified) if self.fbm_scores_notified else 0
        avg_fbm_responded = sum(self.fbm_scores_responded) / len(self.fbm_scores_responded) if self.fbm_scores_responded else 0
        avg_fbm_ignored = sum(self.fbm_scores_ignored) / len(self.fbm_scores_ignored) if self.fbm_scores_ignored else 0
        
        result = SimulationResult(
            user_id=self.user_id,
            total_hours=len(days_data) * 24,
            total_days=len(days_data),
            notifications_sent=self.notifications_sent,
            actions_performed=self.actions_performed,
            vp_count=self.vp_count,
            vn_count=self.vn_count,
            fp_count=self.fp_count,
            fn_count=self.fn_count,
            initial_threshold=initial_threshold,
            final_threshold=final_threshold,
            threshold_change=final_threshold - initial_threshold,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1_score=f1_score,
            avg_fbm_when_notified=avg_fbm_notified,
            avg_fbm_when_responded=avg_fbm_responded,
            avg_fbm_when_ignored=avg_fbm_ignored,
            hourly_stats=self.hourly_stats,
            threshold_evolution=self.threshold_evolution
        )
        
        self._print_analysis(result, all_day_results)
        
        return result
    
    def _print_analysis(self, result: SimulationResult, day_results: List[Dict]):
        """Imprime análise detalhada da simulação."""
        
        print(f"\n{'='*100}")
        print("📊 ANÁLISE FINAL DA SIMULAÇÃO")
        print(f"{'='*100}\n")
        
        print("🎯 Threshold:")
        print(f"   Inicial: {result.initial_threshold:.2f}")
        print(f"   Final:   {result.final_threshold:.2f}")
        print(f"   Mudança: {result.threshold_change:+.2f} ({result.threshold_change/result.initial_threshold*100:+.1f}%)")
        
        print(f"\n📱 Notificações e Ações:")
        print(f"   Total de horas simuladas: {result.total_hours}")
        print(f"   Notificações enviadas: {result.notifications_sent}")
        print(f"   Ações executadas: {result.actions_performed}")
        print(f"   Taxa de notificação: {result.notifications_sent/result.total_hours*100:.1f}%")
        print(f"   Taxa de resposta: {result.actions_performed/result.notifications_sent*100:.1f}%" if result.notifications_sent > 0 else "")
        
        print(f"\n📈 Matriz de Confusão:")
        print(f"   VP (Verdadeiro Positivo):  {result.vp_count:3d} - Notificou e usuário respondeu ✅")
        print(f"   VN (Verdadeiro Negativo):  {result.vn_count:3d} - Notificou mas usuário ignorou ❌")
        print(f"   FP (Falso Positivo):       {result.fp_count:3d} - Não notificou mas usuário agiu 🎯")
        print(f"   FN (Falso Negativo):       {result.fn_count:3d} - Não notificou e não agiu ⚪")
        
        print(f"\n📊 Métricas de Performance:")
        print(f"   Precisão (Precision): {result.precision*100:.1f}% - Quando notifica, qual % responde")
        print(f"   Recall:               {result.recall*100:.1f}% - De todas ações, qual % foi notificada")
        print(f"   Acurácia (Accuracy):  {result.accuracy*100:.1f}% - % de decisões corretas")
        print(f"   F1-Score:             {result.f1_score*100:.1f}% - Harmonia precisão/recall")
        
        print(f"\n🎲 FBM Scores Médios:")
        print(f"   Quando notificado: {result.avg_fbm_when_notified:.1f}")
        print(f"   Quando respondeu:  {result.avg_fbm_when_responded:.1f}")
        print(f"   Quando ignorou:    {result.avg_fbm_when_ignored:.1f}")
        
        # Top 5 horas com mais respostas
        top_hours = sorted(result.hourly_stats.items(), key=lambda x: x[1]["responded"], reverse=True)[:5]
        print(f"\n⏰ Top 5 Horas com Mais Respostas:")
        for hour, stats in top_hours:
            if stats["responded"] > 0:
                response_rate = stats["responded"] / stats["notified"] * 100 if stats["notified"] > 0 else 0
                print(f"   {hour:02d}h: {stats['responded']:2d} respostas "
                      f"({stats['notified']:2d} notif, {response_rate:.0f}% taxa)")
        
        # Evolução do threshold ao longo do tempo
        print(f"\n📈 Evolução do Threshold:")
        checkpoints = [0, len(result.threshold_evolution)//4, len(result.threshold_evolution)//2, 
                      3*len(result.threshold_evolution)//4, len(result.threshold_evolution)-1]
        for i in checkpoints:
            if i < len(result.threshold_evolution):
                ev = result.threshold_evolution[i]
                print(f"   Hora {i:3d}: Threshold={ev['threshold']:.2f}, "
                      f"FBM={ev['fbm_score']:2d}, Feedback={ev['feedback']}, "
                      f"Ajuste={ev['adjustment']:+.2f}")
        
        print(f"\n{'='*100}")
        print("✅ Simulação concluída!")
        print(f"{'='*100}\n")


def save_simulation_results(result: SimulationResult, output_dir: str = "data/simulation"):
    """Salva resultados da simulação em JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.user_id}_simulation_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    
    print(f"💾 Resultados salvos em: {filepath}")
    
    return str(filepath)


if __name__ == "__main__":
    # Teste básico
    print("🧪 Teste do FBMSimulator")
    print("Execute run_simulation.py para simulação completa")
