"""
Pipeline de Treinamento Incremental.

Fluxo:
1. Carrega dados sintéticos de 30 dias
2. Para cada dia:
   a. Envia para API (/treino)
   b. Treina modelo PPO com dados acumulados
   c. Prevê melhores horários para o dia seguinte
   d. Compara previsão com comportamento esperado (6-8h para matinal)
3. Gera relatório final de análise
"""

import sys
import json
import requests
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Adiciona path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rl.trainer import RLTrainer
from app.services.storage import StorageService
from app.services.threshold_manager import ThresholdManager


API_URL = "http://localhost:8000"

# Horas esperadas para perfil MATINAL
EXPECTED_HOURS_MATINAL = [6, 7, 8]


def load_monthly_data(data_dir: str = "data/synthetic") -> List[Dict]:
    """Carrega os 30 dias de dados sintéticos."""
    
    all_days_file = Path(data_dir) / "user_matinal_30dias_all_days.json"
    
    if not all_days_file.exists():
        print(f"❌ Arquivo não encontrado: {all_days_file}")
        print("   Execute primeiro: python tests/generate_monthly_data.py")
        return []
    
    with open(all_days_file, "r", encoding="utf-8") as f:
        days = json.load(f)
    
    print(f"✅ Carregados {len(days)} dias de dados")
    return days


def send_day_to_api(day_data: Dict) -> Dict:
    """Envia dados de um dia para a API."""
    
    response = requests.post(
        f"{API_URL}/treino",
        json=day_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 201:
        return response.json()
    else:
        return {"error": response.text, "status": response.status_code}


def get_training_data_from_api(user_id: str) -> List[Dict]:
    """Recupera dados de treinamento da API."""
    
    response = requests.get(f"{API_URL}/treino/dados-treinamento/{user_id}")
    
    if response.status_code == 200:
        return response.json()["data"]
    else:
        return []


def get_prediction_from_api(user_id: str) -> Dict[str, Any]:
    """Obtém previsão de melhores horários via API."""
    
    response = requests.get(f"{API_URL}/previsao/{user_id}")
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"recommended_hours": [], "error": response.text}


def send_feedback_to_api(user_id: str, hour: int, notified: bool, executed: bool, fbm_score: float) -> Dict[str, Any]:
    """Envia feedback para ajustar o threshold dinâmico."""
    
    response = requests.post(
        f"{API_URL}/threshold/{user_id}/feedback",
        json={
            "hour": hour,
            "notified": notified,
            "executed": executed,
            "fbm_score": fbm_score
        },
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}


def get_threshold_from_api(user_id: str) -> float:
    """Obtém threshold atual do usuário."""
    
    response = requests.get(f"{API_URL}/threshold/{user_id}")
    
    if response.status_code == 200:
        return response.json().get("current_threshold", 15.0)
    else:
        return 15.0


def get_threshold_stats_from_api(user_id: str) -> Dict[str, Any]:
    """Obtém estatísticas do threshold."""
    
    response = requests.get(f"{API_URL}/threshold/{user_id}/stats")
    
    if response.status_code == 200:
        return response.json()
    else:
        return {}


def calculate_prediction_accuracy(
    predicted_hours: List[int],
    expected_hours: List[int]
) -> Dict[str, Any]:
    """
    Calcula acurácia da previsão comparando com horas esperadas.
    """
    if not predicted_hours:
        return {"accuracy": 0, "overlap": 0, "predicted": [], "expected": expected_hours}
    
    predicted_set = set(predicted_hours[:3])  # Top 3
    expected_set = set(expected_hours)
    
    overlap = predicted_set & expected_set
    
    return {
        "accuracy": len(overlap) / len(expected_set) * 100,
        "overlap": len(overlap),
        "overlap_hours": list(overlap),
        "predicted": list(predicted_set),
        "expected": list(expected_set),
        "correct_predictions": len(overlap),
        "total_expected": len(expected_set)
    }


def run_training_pipeline(
    days_data: List[Dict],
    user_id: str = "user_matinal_30dias",
    train_every_n_days: int = 1,
    timesteps_per_train: int = 500
) -> Dict[str, Any]:
    """
    Executa o pipeline completo de treinamento incremental.
    
    Args:
        days_data: Lista de 30 JSONs diários
        user_id: ID do usuário
        train_every_n_days: Treinar a cada N dias
        timesteps_per_train: Timesteps por sessão de treino
    
    Returns:
        Relatório completo com análise
    """
    
    print("\n" + "=" * 100)
    print("🚀 INICIANDO PIPELINE DE TREINAMENTO INCREMENTAL")
    print("=" * 100)
    print(f"\n📊 Configuração:")
    print(f"   - Total de dias: {len(days_data)}")
    print(f"   - Treinar a cada: {train_every_n_days} dia(s)")
    print(f"   - Timesteps por treino: {timesteps_per_train}")
    print(f"   - Horas esperadas (matinal): {EXPECTED_HOURS_MATINAL}")
    print()
    
    # Inicializa trainer
    trainer = RLTrainer(
        model_path=f"models/ppo_{user_id}",
        verbose=0
    )
    
    # Resultados por dia
    daily_results = []
    
    # Métricas acumuladas
    cumulative_accuracy = []
    api_responses = []
    
    # Threshold dinâmico
    threshold_stats = {"vp": 0, "vn": 0, "fp": 0, "fn": 0}
    
    print("-" * 160)
    print(f"{'Dia':^4} | {'Data':^12} | {'M':^4} | {'A':^4} | {'T':^4} | {'FBM':^5} | {'Thr':^5} | {'API':^4} | {'Treino':^6} | {'Previsão Top 3':^18} | {'Esperado':^12} | {'Acurácia':^8}")
    print("-" * 160)
    
    for day_idx, day_data in enumerate(days_data):
        day_num = day_idx + 1
        day_date = day_data["date"]
        
        # 1. Envia para API
        api_result = send_day_to_api(day_data)
        api_ok = "error" not in api_result
        api_responses.append(api_result)
        
        # 2. Recupera dados de treinamento acumulados
        training_data = get_training_data_from_api(user_id)
        
        # 3. Adiciona ao trainer
        if training_data:
            # Pega apenas os dados do dia atual (últimas 24 amostras)
            day_training_data = training_data[-24:]
            trainer.add_day_data(day_training_data)
        
        # 4. Treina se for o momento
        train_result = None
        if day_num % train_every_n_days == 0 or day_num == len(days_data):
            train_result = trainer.train(total_timesteps=timesteps_per_train)
        
        # 5. Prevê próximo dia via API (simula produção)
        prediction = {"recommended_hours": [], "top_3_hours": []}
        accuracy_result = {"accuracy": 0}
        
        if day_idx < len(days_data) - 1 and day_num >= 5:
            # Usa rota /previsao da API (como em produção)
            prediction = get_prediction_from_api(user_id)
            
            # Calcula acurácia
            predicted_hours = prediction.get("recommended_hours", [])[:3]
            accuracy_result = calculate_prediction_accuracy(predicted_hours, EXPECTED_HOURS_MATINAL)
            cumulative_accuracy.append(accuracy_result["accuracy"])
        
        # Registra resultado do dia
        daily_results.append({
            "day": day_num,
            "date": day_date,
            "api_success": api_ok,
            "trained": train_result is not None,
            "train_result": train_result,
            "prediction": prediction,
            "accuracy": accuracy_result
        })
        
        # 6. Obtém threshold atual
        current_threshold = get_threshold_from_api(user_id)
        
        # 7. Registra feedback para cada hora do dia (ajusta threshold)
        recommended_hours = prediction.get("recommended_hours", [])[:3]
        
        # Calcula médias FBM do dia e envia feedback
        day_m_avg = 0
        day_a_avg = 0
        day_t_avg = 0
        day_fbm_avg = 0
        awake_hours = 0
        
        # Coleta dados detalhados por hora (para print dos dias 1-8)
        hourly_details = []
        
        for hour_data in day_data["hours"]:
            mf = hour_data["motivation_factors"]
            af = hour_data["ability_factors"]
            tf = hour_data["trigger_factors"]
            ctx = hour_data["context"]
            fb = hour_data["feedback"]
            
            if not tf["sleeping"]:
                awake_hours += 1
                
                # Motivação
                m = (1 if mf["valence"] == 1 else 0) + 1 + \
                    (1 if mf["last_activity_score"] == 1 else 0) + \
                    (1 if mf["hours_slept_last_night"] >= 7 else 0)
                
                # Habilidade
                a = (1 if af["cognitive_load"] == 0 else 0) + \
                    (1 if af["activities_performed_today"] <= 1 else 0) + \
                    (1 if af["time_since_last_activity_hours"] >= 1 else 0) + \
                    (1 if af["confidence_score"] >= 4 else 0)
                
                # Gatilho
                t = 1 + (1 if tf["arousal"] == 1 else 0) + \
                    (1 if tf["location"] == "home" else 0) + \
                    (1 if tf["motion_activity"] == "stationary" else 0) + \
                    (1 if ctx["day_period"] == 1 else 0) + \
                    (1 if ctx["is_weekend"] else 0)
                
                fbm_score = m * a * t
                
                day_m_avg += m
                day_a_avg += a
                day_t_avg += t
                day_fbm_avg += fbm_score
                
                # Envia feedback para threshold dinâmico
                hour = hour_data["hour"]
                notified = hour in recommended_hours
                executed = fb.get("action_performed", False)
                
                # Coleta detalhes da hora
                hourly_details.append({
                    "hour": hour,
                    "m": m,
                    "a": a,
                    "t": t,
                    "fbm": fbm_score,
                    "notified": notified,
                    "executed": executed,
                    "sleeping": False
                })
                
                # Registra feedback via API
                feedback_result = send_feedback_to_api(
                    user_id=user_id,
                    hour=hour,
                    notified=notified,
                    executed=executed,
                    fbm_score=fbm_score
                )
                
                # Atualiza contadores locais
                if "feedback_type" in feedback_result:
                    ft = feedback_result["feedback_type"]
                    if ft == "VP":
                        threshold_stats["vp"] += 1
                    elif ft == "VN":
                        threshold_stats["vn"] += 1
                    elif ft == "FP":
                        threshold_stats["fp"] += 1
                    elif ft == "FN":
                        threshold_stats["fn"] += 1
            else:
                # Hora dormindo
                hour = hour_data["hour"]
                hourly_details.append({
                    "hour": hour,
                    "m": 0,
                    "a": 0,
                    "t": 0,
                    "fbm": 0,
                    "notified": False,
                    "executed": False,
                    "sleeping": True
                })
        
        if awake_hours > 0:
            day_m_avg /= awake_hours
            day_a_avg /= awake_hours
            day_t_avg /= awake_hours
            day_fbm_avg /= awake_hours
        
        # Print do dia
        predicted_str = ",".join([f"{h}h" for h in prediction.get("recommended_hours", [])[:3]]) or "-"
        expected_str = ",".join([f"{h}h" for h in EXPECTED_HOURS_MATINAL])
        train_str = "✅" if train_result else "⏭️"
        api_str = "✅" if api_ok else "❌"
        acc_str = f"{accuracy_result['accuracy']:.0f}%" if accuracy_result['accuracy'] > 0 else "-"
        
        print(f" {day_num:2d}  | {day_date} | {day_m_avg:.1f} | {day_a_avg:.1f} | {day_t_avg:.1f} | {day_fbm_avg:4.0f} | {current_threshold:5.1f} | {api_str:^4} | {train_str:^6} | {predicted_str:^18} | {expected_str:^12} | {acc_str:^8}")
        
        # Print detalhado das 24h para o dia 8
        if day_num == 8:
            print(f"\n    📋 Detalhes das 24h do Dia {day_num} ({day_date}):")
            print(f"    {'Hora':^6} | {'M':^3} | {'A':^3} | {'T':^3} | {'FBM':^5} | {'Notif':^6} | {'Exec':^6} | {'Status':^10}")
            print(f"    {'-'*6}-+-{'-'*3}-+-{'-'*3}-+-{'-'*3}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}")
            
            for hd in sorted(hourly_details, key=lambda x: x["hour"]):
                if hd["sleeping"]:
                    status = "💤 Dorm."
                    print(f"    {hd['hour']:2d}h   |  -  |  -  |  -  |   -   |   -    |   -    | {status}")
                else:
                    notif_str = "✅" if hd["notified"] else "❌"
                    exec_str = "✅" if hd["executed"] else "❌"
                    
                    # Determina status VP/VN/FP/FN
                    if hd["notified"] and hd["executed"]:
                        status = "VP ✓"
                    elif hd["notified"] and not hd["executed"]:
                        status = "VN ↑"
                    elif not hd["notified"] and hd["executed"]:
                        status = "FP ↓"
                    else:
                        status = "FN ✓"
                    
                    print(f"    {hd['hour']:2d}h   | {hd['m']:^3} | {hd['a']:^3} | {hd['t']:^3} | {hd['fbm']:^5} | {notif_str:^6} | {exec_str:^6} | {status:^10}")
            
            print()
    
    print("-" * 100)
    
    # Salva modelo final
    model_path = trainer.save_model()
    
    # Obtém estatísticas finais do threshold
    final_threshold_stats = get_threshold_stats_from_api(user_id)
    final_threshold = get_threshold_from_api(user_id)
    
    # Análise final
    analysis = analyze_results(daily_results, cumulative_accuracy, threshold_stats, final_threshold)
    
    return {
        "daily_results": daily_results,
        "analysis": analysis,
        "model_path": model_path,
        "training_summary": trainer.get_training_summary(),
        "threshold_stats": final_threshold_stats,
        "final_threshold": final_threshold
    }


def analyze_results(
    daily_results: List[Dict],
    cumulative_accuracy: List[float],
    threshold_stats: Dict[str, int] = None,
    final_threshold: float = 15.0
) -> Dict[str, Any]:
    """Analisa resultados do pipeline."""
    
    if threshold_stats is None:
        threshold_stats = {"vp": 0, "vn": 0, "fp": 0, "fn": 0}
    
    print(f"\n{'='*60}")
    print("📊 ANÁLISE FINAL")
    print(f"{'='*60}")
    
    # Acurácia ao longo do tempo
    if cumulative_accuracy:
        avg_accuracy = sum(cumulative_accuracy) / len(cumulative_accuracy)
        
        # Divide em períodos
        n = len(cumulative_accuracy)
        first_week = cumulative_accuracy[:7] if n >= 7 else cumulative_accuracy
        last_week = cumulative_accuracy[-7:] if n >= 7 else cumulative_accuracy
        
        avg_first = sum(first_week) / len(first_week) if first_week else 0
        avg_last = sum(last_week) / len(last_week) if last_week else 0
        improvement = avg_last - avg_first
        
        print(f"\n📈 Evolução da Acurácia:")
        print(f"   - Primeira semana: {avg_first:.1f}%")
        print(f"   - Última semana: {avg_last:.1f}%")
        print(f"   - Melhoria: {improvement:+.1f}%")
        print(f"   - Média geral: {avg_accuracy:.1f}%")
        
        # Gráfico ASCII simples
        print(f"\n📉 Acurácia por dia:")
        for i, acc in enumerate(cumulative_accuracy):
            bar = "█" * int(acc / 10)
            print(f"   Dia {i+1:2d}: {bar} {acc:.0f}%")
    
    # Contagem de previsões corretas
    correct_predictions = sum(1 for r in daily_results if r["accuracy"].get("accuracy", 0) >= 66.7)
    total_predictions = len([r for r in daily_results if r["accuracy"].get("accuracy", 0) > 0 or r["accuracy"].get("predicted")])
    
    print(f"\n🎯 Previsões Corretas (≥2 de 3 horas):")
    print(f"   - {correct_predictions}/{total_predictions} dias ({correct_predictions/total_predictions*100:.1f}%)" if total_predictions > 0 else "   - N/A")
    
    # APIs bem-sucedidas
    api_success = sum(1 for r in daily_results if r["api_success"])
    print(f"\n🌐 API:")
    print(f"   - Requisições bem-sucedidas: {api_success}/{len(daily_results)}")
    
    # Threshold dinâmico
    total_feedback = threshold_stats["vp"] + threshold_stats["vn"] + threshold_stats["fp"] + threshold_stats["fn"]
    if total_feedback > 0:
        print(f"\n🎚️ Threshold Dinâmico:")
        print(f"   - Threshold final: {final_threshold:.1f} (inicial: 15.0)")
        print(f"   - VP (notificou+executou): {threshold_stats['vp']} ({threshold_stats['vp']/total_feedback*100:.1f}%)")
        print(f"   - VN (notificou+não exec): {threshold_stats['vn']} ({threshold_stats['vn']/total_feedback*100:.1f}%)")
        print(f"   - FP (não notif+executou): {threshold_stats['fp']} ({threshold_stats['fp']/total_feedback*100:.1f}%)")
        print(f"   - FN (não notif+não exec): {threshold_stats['fn']} ({threshold_stats['fn']/total_feedback*100:.1f}%)")
        
        # Métricas
        precision = threshold_stats['vp'] / (threshold_stats['vp'] + threshold_stats['vn']) if (threshold_stats['vp'] + threshold_stats['vn']) > 0 else 0
        recall = threshold_stats['vp'] / (threshold_stats['vp'] + threshold_stats['fp']) if (threshold_stats['vp'] + threshold_stats['fp']) > 0 else 0
        print(f"   - Precisão: {precision:.1%}")
        print(f"   - Recall: {recall:.1%}")
    
    # Conclusão
    print(f"\n{'='*60}")
    print("📝 CONCLUSÃO")
    print(f"{'='*60}")
    
    if cumulative_accuracy and avg_last >= 66.7:
        print("\n✅ O modelo APRENDEU o padrão matinal!")
        print(f"   O modelo está recomendando corretamente as horas 6-8h")
        print(f"   com acurácia de {avg_last:.1f}% na última semana.")
    elif cumulative_accuracy and improvement > 10:
        print("\n📈 O modelo está APRENDENDO o padrão!")
        print(f"   Houve melhoria de {improvement:.1f}% ao longo do mês.")
        print("   Mais dados podem melhorar ainda mais.")
    else:
        print("\n⚠️ O modelo precisa de mais treinamento.")
        print("   Considere aumentar timesteps ou adicionar mais dados.")
    
    return {
        "avg_accuracy": avg_accuracy if cumulative_accuracy else 0,
        "first_week_avg": avg_first if cumulative_accuracy else 0,
        "last_week_avg": avg_last if cumulative_accuracy else 0,
        "improvement": improvement if cumulative_accuracy else 0,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "cumulative_accuracy": cumulative_accuracy,
        "threshold_stats": threshold_stats,
        "final_threshold": final_threshold
    }


def check_api_online() -> bool:
    """Verifica se a API está online."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def clear_user_data(user_id: str) -> bool:
    """Remove dados do usuário para teste limpo."""
    try:
        response = requests.delete(f"{API_URL}/treino/historico/{user_id}")
        return response.status_code in [200, 404]
    except:
        return False


def reset_threshold(user_id: str) -> bool:
    """Reseta o threshold do usuário para valor inicial."""
    try:
        response = requests.post(f"{API_URL}/threshold/{user_id}/reset")
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    # Verifica API
    if not check_api_online():
        print("❌ API não está online!")
        print("   Execute: python -m uvicorn main:app --port 8000 --reload")
        sys.exit(1)
    
    print("✅ API online!")
    
    # Carrega dados
    days = load_monthly_data()
    if not days:
        sys.exit(1)
    
    # Limpa dados anteriores do usuário
    user_id = "user_matinal_30dias"
    print(f"\n🗑️ Limpando dados anteriores de '{user_id}'...")
    clear_user_data(user_id)
    reset_threshold(user_id)
    print("✅ Dados e threshold resetados")
    
    # Executa pipeline
    results = run_training_pipeline(
        days_data=days,
        user_id=user_id,
        train_every_n_days=1,  # Treina todo dia
        timesteps_per_train=500  # Timesteps por treino (aumentado)
    )
    
    # Salva resultados
    output_file = Path("data/results") / f"{user_id}_training_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Simplifica para JSON (remove objetos não serializáveis)
    serializable_results = {
        "analysis": results["analysis"],
        "model_path": results["model_path"],
        "training_summary": results["training_summary"],
        "threshold_stats": results.get("threshold_stats", {}),
        "final_threshold": results.get("final_threshold", 15.0)
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: {output_file}")
