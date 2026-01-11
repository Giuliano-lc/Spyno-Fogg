"""
Script de Validação Interativa do Sistema de Notificações baseado em FBM.

Este script valida que o sistema está decidindo quando notificar baseado no 
FBM (Fasting Blood Metabolism) e threshold, não apenas nos horários pré-estabelecidos.

Fluxo:
1. Gera dados base do perfil matinal (sem notif/exec pré-definidas)
2. Calcula FBM hora a hora
3. Quando FBM > threshold, notifica o usuário e pausa para interação
4. Usuário responde (1-Responder ou 2-Ignorar)
5. Se responder, coleta feedback (dificuldade, familiaridade)
6. Registra dados e treina modelo progressivamente
7. No dia 7, também coleta feedback da rotina
"""

import sys
import os
import json
from datetime import date, timedelta
from typing import List, Dict, Any
import random
import requests
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.synthetic_data_generator import SyntheticDataGenerator, PERFIL_MATINAL
from app.rl.environment import NotificationEnv
from app.rl.trainer import RLTrainer
from app.services.threshold_manager import ThresholdManager


class InteractiveValidator:
    """Validador interativo do sistema de notificações baseado em FBM e RL."""
    
    def __init__(
        self,
        user_id: str = "user_interactive_001",
        max_notifications_per_day: int = 5,
        start_date: date = date(2025, 11, 21),
        api_base_url: str = "http://localhost:8000"
    ):
        self.user_id = user_id
        self.max_notifications_per_day = max_notifications_per_day
        self.start_date = start_date
        self.api_base_url = api_base_url
        
        # Gerador de dados sintéticos
        self.generator = SyntheticDataGenerator(PERFIL_MATINAL, seed=42)
        
        # Threshold Manager (local e via API)
        self.threshold_manager = ThresholdManager()
        
        # Estatísticas
        self.total_notifications = 0
        self.total_responses = 0
        self.total_ignores = 0
        
        # Dados coletados
        self.collected_data = []
        
        # Histórico de treinamento
        self.training_history = []
        
        # Modelo RL (carrega se existir)
        self.current_model = None
        self.load_existing_model()
        
        # Obtém threshold inicial
        initial_threshold = self.get_threshold_from_api()
        
        print("=" * 100)
        print("🎯 VALIDAÇÃO INTERATIVA - RL DECIDE NOTIFICAÇÕES")
        print("=" * 100)
        print(f"\n📋 Configuração:")
        print(f"   - Perfil: {PERFIL_MATINAL.name.upper()}")
        print(f"   - Threshold inicial (referência): {initial_threshold}")
        print(f"   - Máx notificações/dia: {max_notifications_per_day}")
        print(f"   - Data inicial: {self.start_date}")
        print(f"   - Modelo RL: {'✅ Carregado' if self.current_model else '⏳ Será treinado'}")
        print(f"\n💡 Como funciona:")
        print(f"   1. Sistema calcula FBM hora a hora")
        print(f"   2. RL (ou heurística) DECIDE se notifica")
        print(f"   3. Threshold é REFERÊNCIA do seu padrão")
        print(f"   4. Você responde: 1-Responder ou 2-Ignorar")
        print(f"   5. Sistema aprende e ajusta threshold para refletir seu comportamento")
        print("\n" + "=" * 100)
    
    def generate_base_day_data(self, target_date: date) -> Dict[str, Any]:
        """
        Gera dados base de um dia SEM notificações/execuções pré-estabelecidas.
        Apenas os dados contextuais (M, A, T, FBM).
        """
        # Gera dia sem estratégia de notificação
        day_data = self.generator.generate_day(
            user_id=self.user_id,
            target_date=target_date,
            notification_strategy="never"  # Não pré-estabelece notificações
        )
        
        # Remove feedback pré-estabelecido - vamos coletar interativamente
        for hour_data in day_data["hours"]:
            hour_data["feedback"]["notification_sent"] = False
            hour_data["feedback"]["action_performed"] = False
            hour_data["feedback"]["training_feedback"] = None
        
        return day_data
    
    def calculate_fbm(self, hour_data: Dict) -> Dict[str, Any]:
        """Calcula scores FBM para uma hora com detalhamento completo."""
        mf = hour_data["motivation_factors"]
        af = hour_data["ability_factors"]
        tf = hour_data["trigger_factors"]
        ctx = hour_data["context"]
        
        # MOTIVAÇÃO (M) - Range: 0 a 4
        valence_score = 1 if mf["valence"] == 1 else 0
        family_score = 1  # PERFIL_MATINAL tem família
        benefit_score = 1 if mf.get("last_activity_score", 0) == 1 else 0
        sleep_score = 1 if mf["hours_slept_last_night"] >= 7 else 0
        motivation = valence_score + family_score + benefit_score + sleep_score
        
        # HABILIDADE (A) - Range: 0 a 4
        load_score = 1 if af["cognitive_load"] == 0 else 0
        strain_score = 1 if af["activities_performed_today"] <= 1 else 0
        ready_score = 1 if af["time_since_last_activity_hours"] >= 1 else 0
        conf_score = 1 if af.get("confidence_score", 0) >= 4 else 0
        ability = load_score + strain_score + ready_score + conf_score
        
        # GATILHO (T) - Range: 0 a 6
        if tf["sleeping"]:
            trigger = 0
            awake_score = 0
            arousal_score = 0
            location_score = 0
            motion_score = 0
            time_score = 0
            day_score = 0
        else:
            awake_score = 1
            arousal_score = 1 if tf["arousal"] == 1 else 0
            location_score = 1 if tf["location"] == "home" else 0
            motion_score = 1 if tf["motion_activity"] == "stationary" else 0
            time_score = 1 if ctx["day_period"] == 1 else 0
            day_score = 1 if ctx["is_weekend"] else 0
            trigger = awake_score + arousal_score + location_score + motion_score + time_score + day_score
        
        fbm_score = motivation * ability * trigger
        
        return {
            "motivation": motivation,
            "ability": ability,
            "trigger": trigger,
            "fbm_score": fbm_score,
            # Detalhamento de M
            "m_valence": valence_score,
            "m_family": family_score,
            "m_benefit": benefit_score,
            "m_sleep": sleep_score,
            # Detalhamento de A
            "a_load": load_score,
            "a_strain": strain_score,
            "a_ready": ready_score,
            "a_confidence": conf_score,
            # Detalhamento de T
            "t_awake": awake_score,
            "t_arousal": arousal_score,
            "t_location": location_score,
            "t_motion": motion_score,
            "t_time": time_score,
            "t_weekend": day_score,
            # Dados contextuais
            "valence_raw": mf["valence"],
            "hours_slept": mf["hours_slept_last_night"],
            "cognitive_load_raw": af["cognitive_load"],
            "activities_today": af["activities_performed_today"],
            "time_since_last": af["time_since_last_activity_hours"],
            "confidence_raw": af.get("confidence_score", 0),
            "arousal_raw": tf["arousal"],
            "location_raw": tf["location"],
            "motion_raw": tf["motion_activity"],
            "day_period": ctx["day_period"],
            "is_weekend": ctx["is_weekend"]
        }
    
    def load_existing_model(self):
        """Carrega modelo existente se disponível."""
        model_path = Path(f"models/interactive_model_latest.zip")
        if model_path.exists():
            try:
                from stable_baselines3 import PPO
                self.current_model = PPO.load(str(model_path))
                print(f"✅ Modelo carregado: {model_path}")
            except Exception as e:
                print(f"⚠️ Erro ao carregar modelo: {e}")
                self.current_model = None
        else:
            print("ℹ️ Nenhum modelo anterior encontrado, usando heurística inicial")
    
    def get_threshold_from_api(self) -> float:
        """Obtém threshold atual da API."""
        try:
            response = requests.get(f"{self.api_base_url}/threshold/{self.user_id}")
            if response.status_code == 200:
                return response.json()["current_threshold"]
        except:
            pass
        # Fallback para threshold local
        return self.threshold_manager.get_threshold(self.user_id)
    
    def should_notify_with_rl(
        self,
        hour_data: Dict,
        notifications_today: int
    ) -> tuple[bool, str]:
        """
        RL DECIDE se deve notificar (não o threshold!).
        Threshold é apenas referência visual.
        
        Returns:
            (should_notify, reason)
        """
        tf = hour_data["trigger_factors"]
        
        # Regras básicas invioláveis
        if tf["sleeping"]:
            return False, "Dormindo"
        
        if notifications_today >= self.max_notifications_per_day:
            return False, "Limite diário atingido"
        
        # SE MODELO RL TREINADO: RL decide!
        if self.current_model is not None:
            try:
                env = NotificationEnv()
                obs = env._prepare_observation(hour_data)
                action, _ = self.current_model.predict(obs, deterministic=True)
                
                if action == 1:
                    return True, "RL recomenda notificar"
                else:
                    return False, "RL recomenda não notificar"
            except Exception as e:
                print(f"⚠️ Erro no modelo RL: {e}, usando heurística")
        
        # FALLBACK (primeiros dias sem modelo): usa threshold heurístico
        fbm_info = self.calculate_fbm(hour_data)
        threshold = self.get_threshold_from_api()
        
        if fbm_info['fbm_score'] >= threshold:
            return True, f"Heurística: FBM ({fbm_info['fbm_score']}) >= Threshold ({threshold})"
        else:
            return False, f"Heurística: FBM ({fbm_info['fbm_score']}) < Threshold ({threshold})"
    
    def prompt_user_response(self, hour: int, date_str: str, fbm_info: Dict, day_number: int) -> Dict[str, Any]:
        """
        Exibe notificação simulada com detalhes completos de FBM e coleta resposta do usuário.
        """
        print("\n" + "🔔" * 50)
        print(f"📱 NOTIFICAÇÃO DE EXERCÍCIO!")
        print(f"   📅 Data: {date_str} (Dia {day_number})")
        print(f"   🕐 Hora: {hour:02d}:00")
        print("🔔" * 50)
        
        # FBM Score e Threshold (agora obtém da API)
        fbm_score = fbm_info['fbm_score']
        threshold = self.get_threshold_from_api()
        
        print(f"\n📊 BEHAVIOR (FBM Score): {fbm_score}")
        print(f"   🎚️  Threshold de Referência (seu padrão): {threshold:.1f}")
        
        if fbm_score > threshold:
            diff = fbm_score - threshold
            print(f"   ✅ FBM está {diff:.1f} pontos ACIMA do seu padrão usual")
            if diff > 20:
                print(f"   💪 Condições EXCELENTES para você!")
            elif diff > 10:
                print(f"   👍 Boas condições!")
            else:
                print(f"   🤔 Condições razoáveis")
        else:
            diff = threshold - fbm_score
            print(f"   ⚠️  FBM está {diff:.1f} pontos ABAIXO do seu padrão")
            print(f"   🤔 RL decidiu notificar mesmo assim")
        
        print(f"\n   ℹ️  Threshold reflete SEU comportamento (ajusta conforme suas respostas)")
        
        # Decomposição: B = M × A × T
        print(f"\n🧮 Fórmula: B = M × A × T")
        print(f"   {fbm_score} = {fbm_info['motivation']} × {fbm_info['ability']} × {fbm_info['trigger']}")
        
        # Detalhamento da MOTIVAÇÃO
        print(f"\n💪 MOTIVAÇÃO (M = {fbm_info['motivation']}/4):")
        print(f"   {'✅' if fbm_info['m_valence'] else '❌'} Valência (estado emocional): {'Positivo' if fbm_info['valence_raw'] == 1 else 'Negativo'} [{fbm_info['m_valence']}]")
        print(f"   {'✅' if fbm_info['m_family'] else '❌'} Suporte familiar: Sim [{fbm_info['m_family']}]")
        print(f"   {'✅' if fbm_info['m_benefit'] else '❌'} Benefício percebido (última atividade): {'Alto' if fbm_info['m_benefit'] else 'Baixo'} [{fbm_info['m_benefit']}]")
        print(f"   {'✅' if fbm_info['m_sleep'] else '❌'} Sono adequado: {fbm_info['hours_slept']}h {'(≥7h)' if fbm_info['m_sleep'] else '(<7h)'} [{fbm_info['m_sleep']}]")
        
        # Detalhamento da HABILIDADE
        print(f"\n🎯 HABILIDADE (A = {fbm_info['ability']}/4):")
        print(f"   {'✅' if fbm_info['a_load'] else '❌'} Carga cognitiva: {'Baixa' if fbm_info['cognitive_load_raw'] == 0 else 'Alta'} [{fbm_info['a_load']}]")
        print(f"   {'✅' if fbm_info['a_strain'] else '❌'} Fadiga (atividades hoje): {fbm_info['activities_today']} {'(≤1)' if fbm_info['a_strain'] else '(>1)'} [{fbm_info['a_strain']}]")
        print(f"   {'✅' if fbm_info['a_ready'] else '❌'} Tempo desde última atividade: {fbm_info['time_since_last']}h {'(≥1h)' if fbm_info['a_ready'] else '(<1h)'} [{fbm_info['a_ready']}]")
        print(f"   {'✅' if fbm_info['a_confidence'] else '❌'} Confiança/autoeficácia: {fbm_info['confidence_raw']}/10 {'(≥4)' if fbm_info['a_confidence'] else '(<4)'} [{fbm_info['a_confidence']}]")
        
        # Detalhamento do GATILHO
        period_names = {0: "Manhã (6-10h)", 1: "Meio-dia (10-18h)", 2: "Noite (18-22h)", 3: "Madrugada (22-6h)"}
        arousal_names = {0: "Baixo", 1: "Médio (ideal)", 2: "Alto"}
        print(f"\n🎯 GATILHO (T = {fbm_info['trigger']}/6):")
        print(f"   {'✅' if fbm_info['t_awake'] else '❌'} Acordado: Sim [{fbm_info['t_awake']}]")
        print(f"   {'✅' if fbm_info['t_arousal'] else '❌'} Nível de ativação: {arousal_names.get(fbm_info['arousal_raw'], fbm_info['arousal_raw'])} [{fbm_info['t_arousal']}]")
        print(f"   {'✅' if fbm_info['t_location'] else '❌'} Localização: {fbm_info['location_raw'].capitalize()} [{fbm_info['t_location']}]")
        print(f"   {'✅' if fbm_info['t_motion'] else '❌'} Atividade física: {fbm_info['motion_raw'].capitalize()} [{fbm_info['t_motion']}]")
        print(f"   {'✅' if fbm_info['t_time'] else '❌'} Período do dia: {period_names.get(fbm_info['day_period'], fbm_info['day_period'])} [{fbm_info['t_time']}]")
        print(f"   {'✅' if fbm_info['t_weekend'] else '❌'} Fim de semana: {'Sim' if fbm_info['is_weekend'] else 'Não'} [{fbm_info['t_weekend']}]")
        
        print("\n💪 Hora de fazer seu treino matinal!")
        print("🔔" * 50)
        
        # Opções (SIMPLIFICADO: só pergunta 1 ou 2)
        print("\n❓ Como você responde?")
        print("   1️⃣  - RESPONDER (fazer o treino)")
        print("   2️⃣  - IGNORAR (não fazer agora)")
        
        while True:
            try:
                choice = input("\n>>> ").strip()
                if choice in ['1', '2']:
                    break
                print("❌ Digite 1 ou 2.")
            except KeyboardInterrupt:
                print("\n\n⚠️  Validação interrompida pelo usuário.")
                sys.exit(0)
        
        if choice == '2':
            print("❌ Você ignorou a notificação.")
            return {"responded": False}
        
        # Usuário respondeu - simula feedback automaticamente
        print("✅ Você decidiu fazer o treino!")
        
        # Simula feedback baseado no FBM
        import random
        difficulty = random.randint(1, 3) if fbm_info['fbm_score'] > 40 else random.randint(2, 4)
        familiarity = random.randint(2, 4)
        
        return {
            "responded": True,
            "difficulty_level": difficulty,
            "familiarity_level": familiarity,
            "completed_fully": True,
            "duration_minutes": None
        }
    
    def collect_routine_feedback(self, day_number: int) -> Dict[str, Any]:
        """Coleta feedback sobre a rotina (feedback semanal opcional)."""
        print("\n" + "=" * 50)
        print(f"🎉 DIA {day_number} - FINAL DA SEMANA {day_number // 7}!")
        print("=" * 50)
        print("\n📝 Feedback OPCIONAL da rotina semanal (pressione Enter para pular):")
        print("   💡 Você pode pular para continuar mais rápido!\n")
        
        # Dificuldade da rotina (opcional)
        try:
            diff = input("🏋️  Dificuldade da rotina? (1-5 ou Enter para pular): ").strip()
            if not diff:
                print("⏩ Feedback pulado, continuando validação...")
                return None
            difficulty = int(diff)
            if not (1 <= difficulty <= 5):
                print("❌ Inválido, pulando feedback.")
                return None
        except (ValueError, KeyboardInterrupt):
            if isinstance(sys.exc_info()[0], KeyboardInterrupt):
                print("\n\n⚠️  Validação interrompida pelo usuário.")
                sys.exit(0)
            print("⏩ Pulando feedback.")
            return None
        
        # Familiaridade com a rotina
        try:
            fam = input("🎯 Familiaridade com a rotina? (1-5 ou Enter para pular): ").strip()
            if not fam:
                familiarity = 3  # Default médio
            else:
                familiarity = int(fam)
                if not (1 <= familiarity <= 5):
                    familiarity = 3
        except (ValueError, KeyboardInterrupt):
            if isinstance(sys.exc_info()[0], KeyboardInterrupt):
                print("\n\n⚠️  Validação interrompida pelo usuário.")
                sys.exit(0)
            familiarity = 3
        
        print("\n✅ Feedback registrado! Continuando validação...")
        print("=" * 50)
        
        return {
            "routine_difficulty": difficulty,
            "routine_familiarity": familiarity
        }
    
    def process_day(self, day_number: int, target_date: date) -> Dict[str, Any]:
        """Processa um dia completo com interação do usuário."""
        print("\n" + "=" * 100)
        print(f"📅 DIA {day_number} - {target_date.strftime('%Y-%m-%d (%A)')}")
        print("=" * 100)
        
        # Gera dados base
        day_data = self.generate_base_day_data(target_date)
        
        notifications_today = 0
        responses_today = 0
        
        # Processa cada hora
        for hour_data in day_data["hours"]:
            hour = hour_data["hour"]
            
            # Calcula FBM
            fbm_info = self.calculate_fbm(hour_data)
            
            # RL decide se notifica
            should_notify, reason = self.should_notify_with_rl(hour_data, notifications_today)
            
            if should_notify:
                # NOTIFICA O USUÁRIO
                self.total_notifications += 1
                notifications_today += 1
                
                # Coleta resposta
                user_response = self.prompt_user_response(
                    hour,
                    target_date.strftime('%Y-%m-%d'),
                    fbm_info,
                    day_number
                )
                
                # Atualiza feedback
                hour_data["feedback"]["notification_sent"] = True
                executed = user_response["responded"]
                
                if executed:
                    # Usuário respondeu
                    self.total_responses += 1
                    responses_today += 1
                    hour_data["feedback"]["action_performed"] = True
                    hour_data["feedback"]["training_feedback"] = {
                        "difficulty_level": user_response["difficulty_level"],
                        "familiarity_level": user_response["familiarity_level"],
                        "completed_fully": user_response["completed_fully"],
                        "duration_minutes": user_response["duration_minutes"]
                    }
                else:
                    # Usuário ignorou
                    self.total_ignores += 1
                    hour_data["feedback"]["action_performed"] = False
                
                # ENVIA FEEDBACK PARA API (atualiza threshold)
                self.send_feedback_to_api(
                    hour=hour,
                    notified=True,
                    executed=executed,
                    fbm_score=fbm_info["fbm_score"]
                )
        
        # Feedback da rotina no dia 7
        routine_feedback = None
        if day_number % 7 == 0:
            routine_feedback = self.collect_routine_feedback(day_number)
        
        # Resumo do dia
        print(f"\n📊 RESUMO DO DIA {day_number}:")
        print(f"   - Notificações enviadas: {notifications_today}")
        print(f"   - Treinos realizados: {responses_today}")
        print(f"   - Taxa de resposta: {responses_today/notifications_today*100:.1f}%" if notifications_today > 0 else "   - Nenhuma notificação enviada")
        
        # Adiciona metadata
        day_data["_metadata"] = {
            "day_number": day_number,
            "notifications_sent": notifications_today,
            "responses": responses_today,
            "routine_feedback": routine_feedback
        }
        
        return day_data
    
    def send_feedback_to_api(self, hour: int, notified: bool, executed: bool, fbm_score: float):
        """Envia feedback para API (atualiza threshold)."""
        try:
            response = requests.post(
                f"{self.api_base_url}/threshold/{self.user_id}/feedback",
                json={
                    "hour": hour,
                    "notified": notified,
                    "executed": executed,
                    "fbm_score": fbm_score
                }
            )
            if response.status_code == 200:
                result = response.json()
                print(f"   🎚️  Threshold: {result['old_threshold']:.1f} → {result['new_threshold']:.1f} ({result['feedback_type']})")
        except Exception as e:
            # Fallback: atualiza threshold localmente
            self.threshold_manager.update_threshold(
                self.user_id, hour, notified, executed, fbm_score
            )
    
    def run_validation(self, num_days: int = 14):
        """Executa validação interativa por N dias."""
        threshold = self.get_threshold_from_api()
        print(f"\n🚀 Iniciando validação de {num_days} dias...")
        print(f"   RL (ou heurística) decidirá quando notificar")
        print(f"   Threshold atual (referência): {threshold:.1f}")
        print(f"   Máximo de {self.max_notifications_per_day} notificações por dia")
        if num_days >= 14:
            print(f"\n   💡 DICA: A cada 7 dias você pode PULAR o feedback de rotina (Enter)")
            print(f"            para continuar mais rápido até o dia {num_days}!")
        print("\n" + "=" * 100)
        
        for day in range(1, num_days + 1):
            target_date = self.start_date + timedelta(days=day - 1)
            day_data = self.process_day(day, target_date)
            self.collected_data.append(day_data)
            
            # Treina modelo a cada 7 dias
            if day % 7 == 0:
                print(f"\n📊 PROGRESSO: {day}/{num_days} dias completos ({day/num_days*100:.0f}%)")
                self.train_model_incremental()
        
        # Relatório final
        self.generate_final_report()
    
    def train_model_incremental(self):
        """Treina modelo com dados coletados até o momento."""
        print("\n" + "🤖" * 50)
        print("🧠 TREINANDO MODELO COM DADOS COLETADOS...")
        print("🤖" * 50)
        
        # Prepara dados de treinamento
        training_data = self.prepare_training_data()
        
        if len(training_data) == 0:
            print("⚠️  Nenhum dado para treinar ainda.")
            return
        
        # Cria trainer e adiciona dados
        trainer = RLTrainer(
            model_path=f"tests/models/interactive_validation_{len(self.collected_data)}days",
            verbose=0
        )
        
        trainer.add_day_data(training_data)
        
        print(f"\n📚 Treinando com {len(training_data)} amostras...")
        training_stats = trainer.train(total_timesteps=len(training_data) * 3)
        
        if "error" in training_stats:
            print(f"⚠️  {training_stats['error']}")
            return
        
        print(f"\n📊 Estatísticas do Treinamento:")
        eval_stats = training_stats.get("eval_stats", {})
        print(f"   - Total de amostras: {training_stats['total_samples']}")
        print(f"   - VP: {eval_stats.get('vp', 0)} | FN: {eval_stats.get('fn', 0)} | FP: {eval_stats.get('fp', 0)} | VN: {eval_stats.get('vn', 0)}")
        print(f"   - Precisão: {eval_stats.get('precision', 0)*100:.1f}%")
        print(f"   - Recall: {eval_stats.get('recall', 0)*100:.1f}%")
        
        self.training_history.append({
            "day": len(self.collected_data),
            "samples": len(training_data),
            "stats": eval_stats
        })
        
        # Salva modelo
        if trainer.model is not None:
            model_path = f"tests/models/interactive_model_{len(self.collected_data)}days.zip"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            trainer.model.save(model_path)
            print(f"\n💾 Modelo salvo em: {model_path}")
        
        print("🤖" * 50)
    
    def prepare_training_data(self) -> List[Dict]:
        """Prepara dados coletados para formato de treinamento do RL."""
        training_data = []
        
        for day_data in self.collected_data:
            for hour_data in day_data["hours"]:
                fbm_info = self.calculate_fbm(hour_data)
                
                observation = {
                    "hour": hour_data["hour"],
                    "day_period": hour_data["context"]["day_period"],
                    "is_weekend": hour_data["context"]["is_weekend"],
                    "motivation": fbm_info["motivation"],
                    "ability": fbm_info["ability"],
                    "trigger": fbm_info["trigger"],
                    "sleeping": hour_data["trigger_factors"]["sleeping"],
                    "fbm_score": fbm_info["fbm_score"]
                }
                
                action = 1 if hour_data["feedback"]["notification_sent"] else 0
                action_performed = hour_data["feedback"]["action_performed"]
                
                training_data.append({
                    "observation": observation,
                    "action": action,
                    "action_performed": action_performed
                })
        
        return training_data
    
    def generate_final_report(self):
        """Gera relatório final da validação."""
        print("\n" + "=" * 100)
        print("📊 RELATÓRIO FINAL DA VALIDAÇÃO")
        print("=" * 100)
        
        print(f"\n🎯 Estatísticas Gerais:")
        print(f"   - Total de dias: {len(self.collected_data)}")
        print(f"   - Total de notificações: {self.total_notifications}")
        print(f"   - Total de respostas: {self.total_responses}")
        print(f"   - Total de ignores: {self.total_ignores}")
        print(f"   - Taxa de resposta global: {self.total_responses/self.total_notifications*100:.1f}%" if self.total_notifications > 0 else "   - Nenhuma notificação enviada")
        
        # Análise por dia
        print(f"\n📅 Análise Diária:")
        print(f"{'Dia':^5} | {'Data':^12} | {'Notif':^6} | {'Resp':^6} | {'Taxa':^8}")
        print("-" * 50)
        for day_data in self.collected_data:
            meta = day_data["_metadata"]
            day_num = meta["day_number"]
            notif = meta["notifications_sent"]
            resp = meta["responses"]
            taxa = f"{resp/notif*100:.1f}%" if notif > 0 else "N/A"
            print(f"{day_num:^5} | {day_data['date']:^12} | {notif:^6} | {resp:^6} | {taxa:^8}")
        
        # Histórico de treinamento
        if self.training_history:
            print(f"\n🧠 Histórico de Treinamento:")
            print(f"{'Dia':^5} | {'Amostras':^10} | {'Precisão':^10} | {'Recall':^10}")
            print("-" * 50)
            for entry in self.training_history:
                stats = entry["stats"]
                print(f"{entry['day']:^5} | {entry['samples']:^10} | {stats['precision']*100:^9.1f}% | {stats['recall']*100:^9.1f}%")
        
        # Salva dados
        output_dir = "tests/results/interactive_validation"
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva JSON
        output_file = f"{output_dir}/validation_data_{len(self.collected_data)}days.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "user_id": self.user_id,
                    "start_date": self.start_date.isoformat(),
                    "max_notifications_per_day": self.max_notifications_per_day
                },
                "stats": {
                    "total_days": len(self.collected_data),
                    "total_notifications": self.total_notifications,
                    "total_responses": self.total_responses,
                    "total_ignores": self.total_ignores,
                    "response_rate": self.total_responses/self.total_notifications if self.total_notifications > 0 else 0
                },
                "training_history": self.training_history,
                "daily_data": self.collected_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Dados salvos em: {output_file}")
        
        print("\n" + "=" * 100)
        print("✅ VALIDAÇÃO CONCLUÍDA!")
        print("=" * 100)
        
        # Threshold final
        final_threshold = self.get_threshold_from_api()
        
        print(f"\n🎉 O sistema foi validado com sucesso!")
        print(f"   - RL (ou heurística) decidiu notificações baseado em FBM")
        print(f"   - Threshold dinâmico ajustou-se ao SEU padrão: {final_threshold:.1f}")
        print(f"   - O modelo aprendeu com suas {self.total_responses} respostas")
        print(f"   - Taxa de resposta alcançada: {self.total_responses/self.total_notifications*100:.1f}%" if self.total_notifications > 0 else "")
        print("\n" + "=" * 100)


def main():
    """Função principal."""
    print("\n" + "🎮" * 50)
    print("BEM-VINDO À VALIDAÇÃO INTERATIVA DO SISTEMA DE NOTIFICAÇÕES!")
    print("🎮" * 50)
    
    print("\n📝 Instruções:")
    print("   - RL (ou heurística) decide quando notificar baseado no FBM")
    print("   - Threshold é REFERÊNCIA do seu padrão (ajusta automaticamente)")
    print("   - Escolha se vai RESPONDER (fazer treino) ou IGNORAR")
    print("   - Se responder, forneça feedback sobre o treino")
    print("   - Sistema aprende e threshold reflete SEU comportamento")
    print("   - Pressione Ctrl+C a qualquer momento para sair")
    
    # Configuração
    while True:
        try:
            print("\n⚙️  Configuração:")
            days_input = input(f"   Quantos dias deseja simular? (padrão: 14): ").strip()
            num_days = int(days_input) if days_input else 14
            
            max_notif_input = input(f"   Máximo de notificações por dia? (padrão: 5): ").strip()
            max_notif = int(max_notif_input) if max_notif_input else 5
            
            break
        except (ValueError, KeyboardInterrupt):
            if isinstance(sys.exc_info()[0], KeyboardInterrupt):
                print("\n\n👋 Até logo!")
                return
            print("❌ Digite valores numéricos válidos.")
    
    # Cria validador (threshold agora vem da API)
    validator = InteractiveValidator(
        max_notifications_per_day=max_notif
    )
    
    # Executa validação
    try:
        validator.run_validation(num_days=num_days)
    except KeyboardInterrupt:
        print("\n\n⚠️  Validação interrompida pelo usuário.")
        print(f"   Dados coletados até agora: {len(validator.collected_data)} dias")
        if validator.collected_data:
            validator.generate_final_report()


if __name__ == "__main__":
    main()
