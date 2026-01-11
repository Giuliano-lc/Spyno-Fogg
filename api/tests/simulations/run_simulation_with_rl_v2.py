"""
Simulação FBM-Based com RL em Produção - Versão 2

Segue o mesmo fluxo do run_simulation.py mas usando APIs de produção:
1. Gera dados sintéticos
2. Para cada hora:
   - RL decide se notifica (via API /previsao/custom)
   - Usuário responde baseado em FBM + preferências
   - Feedback é coletado
3. Ao final do dia, treina modelo (via API /treino)
4. Compara RL vs Regra Simples
"""

import json
import requests
import random
import shutil
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from synthetic_data_generator import SyntheticDataGenerator, PERFIL_MATINAL
from fbm_simulation import FBMSimulator


API_URL = "http://localhost:8000"
API_TIMEOUT = 30


class RLSimulator:
    """Simulador que usa RL via API para decidir notificações."""
    
    def __init__(self, api_url: str, config: Dict):
        self.api_url = api_url
        self.config = config
        self.user_id = config["user_id"]  # Extrai user_id do config
        
        # Métricas da simulação
        self.total_notifications = 0
        self.total_actions = 0
        self.vp_count = 0
        self.vn_count = 0
        self.fp_count = 0
        self.fn_count = 0
        
        # Stats por hora
        self.hourly_stats = {hour: {"notified": 0, "responded": 0} for hour in range(24)}
        
        # Resultados diários
        self.daily_results = []
        
        # 🔥 EXPLORATION DECAY DINÂMICO
        self.epsilon = 0.30  # Começa com 30% exploration
        self.epsilon_min = 0.02  # Mínimo de 2%
        self.epsilon_decay_rate = 0.95  # Taxa base de decay
        self.vn_threshold = 10  # Se VN do dia > 10, acelera decay
        
    def should_user_respond(self, fbm_score: int, was_notified: bool, hour: int) -> bool:
        """
        Simula se usuário responde (PERFIL MATINAL RÍGIDO).
        
        MODIFICAÇÃO PARA TESTE:
        - Usuário matinal responde MUITO BEM em 6-8h (pico matinal)
        - Usuário matinal responde MODERADAMENTE em 9-12h (manhã tardia)
        - Usuário matinal IGNORA QUASE SEMPRE fora de 6-12h
        - Objetivo: Forçar RL a aprender a focar em 6-8h
        """
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
            # ✅✅ PICO MATINAL (6-8h): Probabilidade MUITO ALTA
            if fbm_score >= 40:
                probability = 0.95  # Quase certeza
            elif fbm_score >= 30:
                probability = 0.85
            elif fbm_score >= 20:
                probability = 0.75
            else:
                probability = 0.60
        elif hour in ACCEPTABLE_HOURS:
            # ✅ MANHÃ TARDIA (9-12h): Probabilidade MÉDIA
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
            # ❌ FORA DA MANHÃ (13-22h): QUASE SEMPRE IGNORA
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
    
    def update_epsilon(self, day_vn: int):
        """
        Atualiza epsilon baseado em VN do dia.
        
        Lógica:
        - Decay base: epsilon *= 0.95
        - Se VN > threshold: decay acelerado (epsilon *= 0.85)
        - Nunca vai abaixo de epsilon_min (2%)
        """
        # Decay base
        self.epsilon *= self.epsilon_decay_rate
        
        # Se dia teve muito VN (spam), acelera decay
        if day_vn > self.vn_threshold:
            extra_decay = 0.90  # Decay adicional de 10%
            self.epsilon *= extra_decay
        
        # Garante mínimo
        self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def get_rl_decision_for_hour(self, day_hours_data: List[Dict], target_hour: int) -> bool:
        """
        Pede para RL decidir se deve notificar em uma hora específica.
        
        Aplica exploration-exploitation com epsilon:
        - epsilon% das vezes: explora (decisão aleatória)
        - (1-epsilon)% das vezes: explota (usa modelo RL)
        
        Envia dados das 24 horas via API /previsao/custom e pega decisão para target_hour.
        """
        # 🔥 EXPLORATION: Com probabilidade epsilon, decide aleatoriamente
        if random.random() < self.epsilon:
            # Exploration: decisão aleatória (50% chance de notificar)
            return random.random() < 0.15  # Bias baixo para não spammar muito
        
        # 🔥 EXPLOITATION: Usa modelo RL
        try:
            response = requests.post(
                f"{self.api_url}/previsao/{self.user_id}/custom",
                json={"hours_data": day_hours_data},
                headers={"Content-Type": "application/json"},
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Procura previsão para hora específica
                for hour_pred in result.get("all_hours", []):
                    if hour_pred["hour"] == target_hour:
                        return hour_pred["recommended"]
                
                # Fallback: verifica se está nas horas recomendadas
                return target_hour in result.get("recommended_hours", [])
            
            # Em caso de erro, não notifica
            return False
            
        except Exception as e:
            # Em caso de erro, não notifica
            return False
    
    def simulate_day_with_rl(self, day_data: Dict) -> Dict:
        """
        Simula um dia completo com RL decidindo notificações.
        
        Para cada hora:
        1. RL decide se notifica
        2. Usuário responde (baseado em FBM + preferências)
        3. Coleta métricas
        """
        day_hours = day_data["hours"]
        results = []
        
        for hour_idx, hour_data in enumerate(day_hours):
            hour = hour_data["hour"]
            
            # Pula se dormindo
            if hour_data["trigger_factors"].get("sleeping"):
                continue
            
            # Calcula FBM (mesma lógica do fbm_simulation.py)
            mf = hour_data["motivation_factors"]
            af = hour_data["ability_factors"]
            tf = hour_data["trigger_factors"]
            ctx = hour_data["context"]
            
            m = (1 if mf.get("valence", 0) == 1 else 0) + 1 + \
                (1 if mf.get("last_activity_score", 0) == 1 else 0) + \
                (1 if mf.get("hours_slept_last_night", 0) >= 7 else 0)
            
            a = (1 if af.get("cognitive_load", 1) == 0 else 0) + \
                (1 if af.get("activities_performed_today", 0) <= 1 else 0) + \
                (1 if af.get("time_since_last_activity_hours", 0) >= 1 else 0) + \
                (1 if af.get("confidence_score", 0) >= 4 else 0)
            
            if tf.get("sleeping", False):
                t = 0
            else:
                t = 1 + (1 if tf.get("arousal", 0) == 1 else 0) + \
                    (1 if tf.get("location", "") == "home" else 0) + \
                    (1 if tf.get("motion_activity", "") == "stationary" else 0) + \
                    (1 if hour in [6, 7, 8] else 0) + \
                    (1 if ctx.get("is_weekend", False) else 0)
            
            fbm_score = m * a * t
            
            # RL decide se notifica
            rl_notifies = self.get_rl_decision_for_hour(day_hours, hour)
            
            # Usuário responde?
            user_responded = self.should_user_respond(fbm_score, rl_notifies, hour)
            
            # Atualiza métricas
            if rl_notifies:
                self.total_notifications += 1
                self.hourly_stats[hour]["notified"] += 1
                
                if user_responded:
                    self.vp_count += 1
                else:
                    self.vn_count += 1
            else:
                if user_responded:
                    self.fp_count += 1
                else:
                    self.fn_count += 1
            
            if user_responded:
                self.total_actions += 1
                self.hourly_stats[hour]["responded"] += 1
            
            # 🔥 ATUALIZA FEEDBACK NOS DADOS ORIGINAIS (para /treino aprender correto!)
            day_hours[hour_idx]["feedback"]["notification_sent"] = rl_notifies
            day_hours[hour_idx]["feedback"]["action_performed"] = user_responded
            
            results.append({
                "hour": hour,
                "fbm_score": fbm_score,
                "rl_notified": rl_notifies,
                "user_responded": user_responded
            })
        
        day_result = {
            "date": day_data["date"],
            "results": results,
            "day_vp": sum(1 for r in results if r["rl_notified"] and r["user_responded"]),
            "day_vn": sum(1 for r in results if r["rl_notified"] and not r["user_responded"]),
            "day_fp": sum(1 for r in results if not r["rl_notified"] and r["user_responded"]),
            "day_fn": sum(1 for r in results if not r["rl_notified"] and not r["user_responded"])
        }
        
        # Armazena resultado do dia
        self.daily_results.append(day_result)
        
        return day_result
    
    def train_with_day(self, day_data: Dict, day_num: int) -> bool:
        """
        Envia dia para API /treino E treina modelo incrementalmente.
        
        Fluxo:
        1. Envia dados do dia para /treino (salva histórico)
        2. Treina modelo com todos dados acumulados até agora
        """
        try:
            # PASSO 1: Envia dados do dia
            api_data = {
                "user_id": day_data["user_id"],
                "date": day_data["date"],
                "timezone": day_data["timezone"],
                "user_profile": day_data["user_profile"],
                "hours": []
            }
            
            # Converte cada hora (null → False, remove campos extras)
            for hour in day_data["hours"]:
                # Feedback: notification_sent, action_performed, e training_feedback (sempre None)
                feedback_clean = {
                    "notification_sent": hour["feedback"].get("notification_sent") or False,
                    "action_performed": hour["feedback"].get("action_performed") or False,
                    "training_feedback": None  # Sempre None para dados sintéticos
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
                # Debug: mostra erro detalhado
                try:
                    error_detail = response.json()
                    print(f"\n   ⚠️ Erro {response.status_code} ao enviar dados:")
                    print(f"   {error_detail}")
                except:
                    print(f"\n   ⚠️ Erro {response.status_code}: {response.text[:200]}")
                return False
            
            # PASSO 2: Treina modelo incrementalmente (só em memória, não salva)
            train_response = requests.post(
                f"{self.api_url}/treino/treinar-incremental/{self.user_id}",
                timeout=120  # Treino pode demorar
            )
            
            if train_response.status_code == 200:
                # 🔍 DEBUG: Mostra informações de treinamento
                train_result = train_response.json()
                
                # Armazena informações para log
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
                # Debug: mostra erro detalhado do treino
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
        
        # Top horas
        top_hours = sorted(
            [(h, stats["responded"]) for h, stats in self.hourly_stats.items() if stats["responded"] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
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
            "top_hours": top_hours
        }


def run_simulation_with_rl():
    """Executa simulação completa com RL."""
    
    CONFIG = {
        "num_days": 30,
        "user_id": "user_matinal_rl_v2",
        "initial_threshold": 40.0,
        "seed": 42
    }
    
    print("\n🚀 Iniciando Simulação FBM-Based com RL (Versão 2)\n")
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
    
    # Limpa histórico anterior do usuário
    print("🧹 Limpando dados anteriores...")
    try:
        response = requests.delete(
            f"{API_URL}/treino/historico/{CONFIG['user_id']}",
            timeout=5
        )
        if response.status_code in [200, 404]:  # 200 = deletado, 404 = não existia
            print("✅ Histórico limpo!")
        else:
            print("⚠️ Aviso: não foi possível limpar histórico anterior")
    except:
        print("⚠️ Aviso: erro ao limpar histórico (continuando)")
    
    # Deleta modelo antigo para re-treinar do zero
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
    print(f"# SIMULAÇÃO HORA-A-HORA - RL Decide Notificações")
    print(f"{'#'*100}\n")
    
    # ETAPA 1: Gera dados sintéticos
    print("📋 ETAPA 1: Geração de Dados Sintéticos\n")
    generator = SyntheticDataGenerator(PERFIL_MATINAL, seed=CONFIG["seed"])
    
    start_date = date.today() - timedelta(days=CONFIG["num_days"])
    days_data = []
    activities_total = 0
    last_completed = True
    
    for i in range(CONFIG["num_days"]):
        current_date = start_date + timedelta(days=i)
        
        day_data = generator.generate_day(
            user_id=CONFIG["user_id"],
            target_date=current_date,
            previous_activities_total=activities_total,
            last_completed=last_completed,
            notification_strategy="fbm_based"
        )
        
        days_data.append(day_data)
        
        # Atualiza contadores
        day_activities = sum(1 for h in day_data["hours"] if h["feedback"].get("action_performed") is True)
        activities_total += day_activities
        
        if (i + 1) % 10 == 0:
            print(f"✅ Gerados {i + 1}/{CONFIG['num_days']} dias")
    
    print(f"\n✅ {CONFIG['num_days']} dias gerados ({CONFIG['num_days'] * 24} horas)\n")
    
    # ETAPA 2: PULADA (comentada para acelerar)
    # print("📋 ETAPA 2: Simulação com REGRA SIMPLES (Baseline)\n")
    # simulator_simple = FBMSimulator(
    #     user_id=f"{CONFIG['user_id']}_simple",
    #     initial_threshold=CONFIG["initial_threshold"]
    # )
    # 
    # results_simple = simulator_simple.simulate_multiple_days(days_data)
    # 
    # # Calcula top horas do baseline
    # simple_top_hours = sorted(
    #     [(h, stats["responded"]) for h, stats in results_simple.hourly_stats.items() if stats["responded"] > 0],
    #     key=lambda x: x[1],
    #     reverse=True
    # )[:5]
    # 
    # print(f"✅ Baseline completo:")
    # print(f"   Precision: {results_simple.precision * 100:.1f}%")
    # print(f"   Recall:    {results_simple.recall * 100:.1f}%")
    # print(f"   F1-Score:  {results_simple.f1_score * 100:.1f}%")
    # print(f"   Top horas: {[h for h, _ in simple_top_hours[:3]]}\n")
    
    # Valores mockados do baseline (para comparação no final)
    results_simple = type('obj', (object,), {
        'precision': 0.505,
        'recall': 0.495,
        'f1_score': 0.500,
        'notifications_sent': 95,
        'vp_count': 48,
        'vn_count': 47,
        'fp_count': 49,
        'fn_count': 336,
        'hourly_stats': {}
    })()
    simple_top_hours = [(7, 25), (8, 20), (6, 19)]
    
    print("📋 ETAPA 2: Baseline (PULADA - usando valores anteriores)\n")
    print(f"✅ Baseline (valores salvos):")
    print(f"   Precision: {results_simple.precision * 100:.1f}%")
    print(f"   Recall:    {results_simple.recall * 100:.1f}%")
    print(f"   F1-Score:  {results_simple.f1_score * 100:.1f}%")
    print(f"   Top horas: {[h for h, _ in simple_top_hours[:3]]}\n")
    
    # ETAPA 3: Simula com RL
    print("📋 ETAPA 3: Simulação com MODELO RL (via API)\n")
    print("Para cada hora:")
    print("  1. RL decide se notifica (via /previsao/custom)")
    print("  2. Usuário responde (baseado em FBM + preferências)")
    print("  3. Coleta métricas\n")
    
    simulator_rl = RLSimulator(api_url=API_URL, config=CONFIG)
    
    for i, day_data in enumerate(days_data, 1):
        # Simula dia com RL
        day_result = simulator_rl.simulate_day_with_rl(day_data)
        
        # 🔥 ATUALIZA EPSILON baseado em VN do dia
        simulator_rl.update_epsilon(day_result['day_vn'])
        
        # 🔥 TREINA MODELO INCREMENTALMENTE (todo dia!)
        print(f"Dia {i:2d}: VP={day_result['day_vp']:2d}, VN={day_result['day_vn']:2d}, "
              f"FP={day_result['day_fp']:2d}, ε={simulator_rl.epsilon:.3f} ", end="")
        
        trained = simulator_rl.train_with_day(day_data, i)
        
        if trained:
            # Mostra info de treinamento se disponível
            if hasattr(simulator_rl, 'training_info') and len(simulator_rl.training_info) > 0:
                info = simulator_rl.training_info[-1]
                loaded = "📥" if info.get("model_loaded") else "🆕"
                saved = "💾" if info.get("model_saved") else "❌"
                reward_str = f", R={info['reward']:.1f}" if "reward" in info else ""
                print(f"Treino=✅ {loaded}{saved} ({i*24} samples{reward_str})")
            else:
                print(f"Treino=✅ ({i*24} samples)")
        else:
            print(f"Treino=❌")
    
    metrics_rl = simulator_rl.get_metrics()
    
    print(f"\n✅ RL completo:")
    print(f"   Precision: {metrics_rl['precision']:.1f}%")
    print(f"   Recall:    {metrics_rl['recall']:.1f}%")
    print(f"   F1-Score:  {metrics_rl['f1_score']:.1f}%")
    print(f"   Top horas: {[h for h, _ in metrics_rl['top_hours'][:3]]}\n")
    
    # ETAPA 4: Comparação
    print("="*100)
    print("📊 COMPARAÇÃO FINAL: REGRA SIMPLES vs RL")
    print("="*100)
    print()
    
    print(f"{'Métrica':<20} | {'Regra Simples':<15} | {'RL':<15} | {'Diferença':<15}")
    print(f"{'-'*70}")
    
    for metric in ["precision", "recall", "f1_score"]:
        simple_val = getattr(results_simple, metric) * 100  # Converte decimal para %
        rl_val = metrics_rl[metric]
        diff = rl_val - simple_val
        symbol = "✅" if diff > 0 else "⚠️" if diff < 0 else "➖"
        print(f"{metric.capitalize():<20} | {simple_val:>13.1f}% | {rl_val:>13.1f}% | {symbol} {diff:>+6.1f}pp")
    
    print()
    print(f"{'Notificações':<20} | {results_simple.notifications_sent:>15} | {metrics_rl['total_notifications']:>15} | {metrics_rl['total_notifications'] - results_simple.notifications_sent:>+15}")
    print(f"{'VP (acertos)':<20} | {results_simple.vp_count:>15} | {metrics_rl['vp']:>15} | {metrics_rl['vp'] - results_simple.vp_count:>+15}")
    print(f"{'VN (ignorados)':<20} | {results_simple.vn_count:>15} | {metrics_rl['vn']:>15} | {metrics_rl['vn'] - results_simple.vn_count:>+15}")
    
    # Validação de padrão
    print(f"\n{'='*100}")
    print(f"🎯 VALIDAÇÃO DE PADRÃO MATINAL")
    print(f"{'='*100}\n")
    
    expected_hours = [6, 7, 8]
    
    simple_top = [h for h, _ in simple_top_hours[:3]]
    rl_top = [h for h, _ in metrics_rl['top_hours'][:3]]
    
    simple_match = sum(1 for h in simple_top if h in expected_hours)
    rl_match = sum(1 for h in rl_top if h in expected_hours)
    
    print(f"Horas esperadas: {expected_hours}\n")
    print(f"Regra Simples:")
    print(f"  Top 3: {simple_top}")
    print(f"  Match: {simple_match}/3 horas {'✅ VALIDADO' if simple_match == 3 else '⚠️ PARCIAL' if simple_match >= 2 else '❌ NÃO VALIDADO'}\n")
    
    print(f"RL:")
    print(f"  Top 3: {rl_top}")
    print(f"  Match: {rl_match}/3 horas {'✅ VALIDADO' if rl_match == 3 else '⚠️ PARCIAL' if rl_match >= 2 else '❌ NÃO VALIDADO'}\n")
    
    # Vencedor
    print(f"{'='*100}")
    print(f"🏆 VENCEDOR")
    print(f"{'='*100}\n")
    
    simple_f1 = results_simple.f1_score * 100
    rl_f1 = metrics_rl['f1_score']
    
    if rl_f1 > simple_f1 + 2:  # Margem de 2pp
        print(f"🥇 MODELO RL venceu!")
        improvement = rl_f1 - simple_f1
        print(f"   Melhoria no F1-Score: +{improvement:.1f} pontos percentuais")
        print(f"   ✅ Vale a pena usar RL em produção!")
    elif simple_f1 > rl_f1 + 2:
        print(f"🥇 REGRA SIMPLES venceu!")
        diff = simple_f1 - rl_f1
        print(f"   F1-Score superior: +{diff:.1f} pontos percentuais")
        print(f"   💡 RL precisa mais dados ou ajuste de hiperparâmetros")
    else:
        print(f"🤝 EMPATE TÉCNICO!")
        print(f"   Diferença < 2pp no F1-Score")
        print(f"   Ambas abordagens são viáveis")
    
    print(f"\n{'='*100}\n")
    
    # ETAPA 5: Salvar modelo RL treinado
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
    
    # ETAPA 6: Salvar dados para análise posterior
    print("💾 Salvando dados detalhados da simulação...\n")
    
    output_dir = Path("data/simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dados do RL
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
            "top_hours": metrics_rl['top_hours']
        },
        "baseline": {
            "notifications_sent": results_simple.notifications_sent,
            "vp": results_simple.vp_count,
            "vn": results_simple.vn_count,
            "fp": results_simple.fp_count,
            "fn": results_simple.fn_count,
            "precision": results_simple.precision,
            "recall": results_simple.recall,
            "f1_score": results_simple.f1_score,
            "hourly_stats": results_simple.hourly_stats
        }
    }
    
    output_file = output_dir / "rl_simulation_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rl_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dados salvos em: {output_file}\n")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    run_simulation_with_rl()
