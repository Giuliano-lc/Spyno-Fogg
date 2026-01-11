"""
Analisa os dados da simulação para identificar problemas.
"""

import json
from pathlib import Path
from collections import defaultdict

def calculate_fbm(hour_data):
    """Calcula FBM de uma hora."""
    mf = hour_data["motivation_factors"]
    af = hour_data["ability_factors"]
    tf = hour_data["trigger_factors"]
    ctx = hour_data["context"]
    
    # Motivação
    m = (1 if mf["valence"] == 1 else 0) + \
        1 + \
        (1 if mf["last_activity_score"] == 1 else 0) + \
        (1 if mf["hours_slept_last_night"] >= 7 else 0)
    
    # Habilidade
    a = (1 if af["cognitive_load"] == 0 else 0) + \
        (1 if af["activities_performed_today"] <= 1 else 0) + \
        (1 if af["time_since_last_activity_hours"] >= 1 else 0) + \
        (1 if af["confidence_score"] >= 4 else 0)
    
    # Gatilho
    if tf["sleeping"]:
        t = 0
    else:
        t = 1 + \
            (1 if tf["arousal"] == 1 else 0) + \
            (1 if tf["location"] == "home" else 0) + \
            (1 if tf["motion_activity"] == "stationary" else 0) + \
            (1 if ctx["day_period"] == 1 else 0) + \
            (1 if ctx["is_weekend"] else 0)
    
    return m * a * t

def analyze_data():
    """Analisa dados brutos da simulação."""
    
    data_file = Path("data/simulation/user_matinal_sim_raw_data.json")
    
    with open(data_file, "r", encoding="utf-8") as f:
        days = json.load(f)
    
    print(f"{'='*100}")
    print(f"📊 ANÁLISE DOS DADOS SINTÉTICOS")
    print(f"{'='*100}\n")
    
    # Estatísticas por hora
    hourly_stats = defaultdict(lambda: {"count": 0, "fbm_sum": 0, "sleeping": 0, "fbm_scores": []})
    
    for day in days:
        for hour_data in day["hours"]:
            hour = hour_data["hour"]
            fbm = calculate_fbm(hour_data)
            sleeping = hour_data["trigger_factors"]["sleeping"]
            
            hourly_stats[hour]["count"] += 1
            hourly_stats[hour]["fbm_sum"] += fbm
            hourly_stats[hour]["fbm_scores"].append(fbm)
            if sleeping:
                hourly_stats[hour]["sleeping"] += 1
    
    # Imprime estatísticas
    print(f"{'Hora':^4} | {'FBM Médio':^10} | {'FBM Min':^8} | {'FBM Max':^8} | {'Dormindo':^10} | {'Obs':^20}")
    print("-" * 100)
    
    for hour in range(24):
        stats = hourly_stats[hour]
        avg_fbm = stats["fbm_sum"] / stats["count"] if stats["count"] > 0 else 0
        min_fbm = min(stats["fbm_scores"]) if stats["fbm_scores"] else 0
        max_fbm = max(stats["fbm_scores"]) if stats["fbm_scores"] else 0
        sleep_pct = stats["sleeping"] / stats["count"] * 100 if stats["count"] > 0 else 0
        
        obs = ""
        if hour in [6, 7, 8]:
            obs = "⭐ HORA PREFERIDA"
        if avg_fbm > 50:
            obs += " 🔥 FBM ALTO"
        
        print(f" {hour:02d}h | {avg_fbm:^10.1f} | {min_fbm:^8} | {max_fbm:^8} | {sleep_pct:^9.0f}% | {obs:^20}")
    
    # Top 10 horas por FBM médio
    print(f"\n{'='*100}")
    print("🏆 TOP 10 HORAS POR FBM MÉDIO")
    print(f"{'='*100}\n")
    
    sorted_hours = sorted(hourly_stats.items(), 
                         key=lambda x: x[1]["fbm_sum"] / x[1]["count"] if x[1]["count"] > 0 else 0, 
                         reverse=True)
    
    for i, (hour, stats) in enumerate(sorted_hours[:10], 1):
        avg_fbm = stats["fbm_sum"] / stats["count"]
        print(f"{i:2d}. Hora {hour:02d}h: FBM médio = {avg_fbm:.1f}")
    
    # Análise específica das horas preferidas
    print(f"\n{'='*100}")
    print("⭐ ANÁLISE DAS HORAS PREFERIDAS (6h, 7h, 8h)")
    print(f"{'='*100}\n")
    
    for hour in [6, 7, 8]:
        stats = hourly_stats[hour]
        avg_fbm = stats["fbm_sum"] / stats["count"]
        print(f"Hora {hour:02d}h:")
        print(f"  FBM médio: {avg_fbm:.1f}")
        print(f"  FBM min/max: {min(stats['fbm_scores'])}/{max(stats['fbm_scores'])}")
        print(f"  Dias analisados: {stats['count']}")
        print()
    
    # Análise das horas identificadas como top (16-19h)
    print(f"\n{'='*100}")
    print("❓ ANÁLISE DAS HORAS IDENTIFICADAS COMO TOP (16h, 17h, 19h)")
    print(f"{'='*100}\n")
    
    for hour in [16, 17, 19]:
        stats = hourly_stats[hour]
        avg_fbm = stats["fbm_sum"] / stats["count"]
        print(f"Hora {hour:02d}h:")
        print(f"  FBM médio: {avg_fbm:.1f}")
        print(f"  FBM min/max: {min(stats['fbm_scores'])}/{max(stats['fbm_scores'])}")
        print(f"  Dias analisados: {stats['count']}")
        print()
    
    # Conclusões
    print(f"\n{'='*100}")
    print("🎯 CONCLUSÕES")
    print(f"{'='*100}\n")
    
    preferidas_avg = sum(hourly_stats[h]["fbm_sum"] / hourly_stats[h]["count"] for h in [6,7,8]) / 3
    top_avg = sum(hourly_stats[h]["fbm_sum"] / hourly_stats[h]["count"] for h in [16,17,19]) / 3
    
    print(f"FBM médio horas preferidas (6-8h): {preferidas_avg:.1f}")
    print(f"FBM médio horas top simulação (16-19h): {top_avg:.1f}")
    print(f"Diferença: {top_avg - preferidas_avg:+.1f}")
    print()
    
    if top_avg > preferidas_avg:
        print("⚠️ PROBLEMA IDENTIFICADO:")
        print("   As horas da tarde/noite têm FBM médio MAIOR que as horas da manhã.")
        print("   Isso explica por que o sistema não identificou o padrão matinal.")
        print()
        print("💡 POSSÍVEIS CAUSAS:")
        print("   1. Gerador sintético não está priorizando horas preferidas")
        print("   2. Componentes FBM (M, A, T) não estão calibrados corretamente")
        print("   3. Fatores como cognitive_load, arousal estão favorecendo tarde/noite")

if __name__ == "__main__":
    analyze_data()
