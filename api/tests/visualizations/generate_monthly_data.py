"""
Gerador de bateria de dados sintéticos para 1 mês (30 dias).
Perfil: Usuário MATINAL com variações realistas.

Total: 24 horas × 30 dias = 720 registros horários
"""

import random
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Any

from synthetic_data_generator import UserProfile, SyntheticDataGenerator


class MatinalProfileWithVariations(UserProfile):
    """
    Perfil MATINAL com variações realistas dia a dia.
    
    Comportamento base (90% dos dias):
    - Prefere treinar entre 6h-8h
    - Dorme 22h, acorda 6h
    - Alta motivação pela manhã
    
    Variações (10% dos dias):
    - Dias de "folga" (não treina)
    - Dias de treino à noite (exceção)
    - Dias de sono ruim (afeta motivação)
    """
    pass


def create_matinal_profile_for_day(
    day_number: int,
    is_weekend: bool,
    variation_seed: float
) -> Dict[str, Any]:
    """
    Cria variações no perfil matinal para cada dia.
    
    Variações implementadas:
    - 5% chance de dia de folga (não treina)
    - 5% chance de trocar para treino noturno
    - 10% chance de sono ruim (afeta motivação)
    - 15% chance de acordar mais tarde no fim de semana
    - Variação na confiança baseada no histórico
    """
    
    # Perfil base matinal
    profile_params = {
        "name": "matinal",
        "preferred_hours": [6, 7, 8],
        "sleep_start": 22,
        "sleep_end": 6,
        "work_start": 8,
        "work_end": 17,
        "base_motivation": 0.8,
        "base_confidence": 6,
        "exercise_probability": 0.85,
        "has_family": True,
        "description": "Usuário matinal"
    }
    
    variation_type = "normal"
    
    # Dia de folga (5%)
    if variation_seed < 0.05:
        profile_params["exercise_probability"] = 0.05  # Quase não treina
        variation_type = "folga"
    
    # Treino noturno excepcional (5%)
    elif variation_seed < 0.10:
        profile_params["preferred_hours"] = [19, 20, 21]
        profile_params["base_motivation"] = 0.6
        variation_type = "noturno"
    
    # Sono ruim (10%)
    elif variation_seed < 0.20:
        profile_params["base_motivation"] = 0.5
        profile_params["exercise_probability"] = 0.6
        variation_type = "sono_ruim"
    
    # Fim de semana - acorda mais tarde (15% nos fins de semana)
    elif is_weekend and variation_seed < 0.35:
        profile_params["sleep_end"] = 8
        profile_params["preferred_hours"] = [8, 9, 10]
        variation_type = "fds_tardio"
    
    # Alta motivação (10%) - dias muito bons
    elif variation_seed > 0.90:
        profile_params["base_motivation"] = 0.95
        profile_params["exercise_probability"] = 0.95
        profile_params["base_confidence"] = 8
        variation_type = "alta_motivacao"
    
    return {
        "profile": UserProfile(**profile_params),
        "variation_type": variation_type
    }


def generate_month_data(
    user_id: str = "user_matinal_30dias",
    start_date: date = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Gera 30 dias de dados com variações realistas.
    
    Returns:
        Dict com:
        - days: Lista de 30 JSONs diários
        - stats: Estatísticas do mês
        - variations: Registro das variações aplicadas
    """
    
    random.seed(seed)
    
    if start_date is None:
        start_date = date.today() - timedelta(days=30)
    
    days_data = []
    variations_log = []
    
    # Estatísticas globais
    total_notifications = 0
    total_actions = 0
    total_vp = 0
    total_fn = 0
    total_fp = 0
    total_vn = 0
    activities_by_hour = {h: 0 for h in range(24)}
    activities_by_period = {"Manhã": 0, "MeioDia": 0, "Noite": 0, "Madrugada": 0}
    
    # Histórico para afetar dias seguintes
    previous_activities_total = 0
    last_completed = True
    consecutive_rest_days = 0
    
    print("=" * 100)
    print(f"🗓️  GERANDO 30 DIAS DE DADOS - Usuário: {user_id}")
    print(f"📅 Período: {start_date} a {start_date + timedelta(days=29)}")
    print("=" * 100)
    print()
    print(f"{'Dia':^4} | {'Data':^12} | {'Sem':^3} | {'Variação':^15} | {'Notif':^5} | {'Ações':^5} | {'VP':^3} | {'FN':^3} | {'Horas Treino':^20}")
    print("-" * 100)
    
    for day_num in range(30):
        current_date = start_date + timedelta(days=day_num)
        is_weekend = current_date.weekday() >= 5
        day_of_week = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"][current_date.weekday()]
        
        # Gera variação para este dia
        variation_seed = random.random()
        
        # Se muitos dias sem treinar, aumenta motivação
        if consecutive_rest_days >= 2:
            variation_seed = max(variation_seed, 0.5)  # Força dia normal ou bom
        
        day_profile = create_matinal_profile_for_day(
            day_number=day_num,
            is_weekend=is_weekend,
            variation_seed=variation_seed
        )
        
        # Cria gerador com o perfil do dia
        generator = SyntheticDataGenerator(
            profile=day_profile["profile"],
            seed=seed + day_num  # Seed diferente por dia
        )
        
        # Gera dados do dia
        day_data = generator.generate_day(
            user_id=user_id,
            target_date=current_date,
            previous_activities_total=previous_activities_total,
            last_completed=last_completed,
            notification_strategy="smart"
        )
        
        # Calcula métricas do dia
        day_notifications = 0
        day_actions = 0
        day_vp = 0
        day_fn = 0
        day_fp = 0
        day_vn = 0
        day_training_hours = []
        
        for hour_data in day_data["hours"]:
            fb = hour_data["feedback"]
            hour = hour_data["hour"]
            ctx = hour_data["context"]
            
            if fb["notification_sent"]:
                day_notifications += 1
            if fb["action_performed"]:
                day_actions += 1
                day_training_hours.append(hour)
                activities_by_hour[hour] += 1
                
                # Período
                period_map = {0: "Manhã", 1: "MeioDia", 2: "Noite", 3: "Madrugada"}
                activities_by_period[period_map[ctx["day_period"]]] += 1
            
            # Matriz de confusão
            if fb["notification_sent"] and fb["action_performed"]:
                day_vp += 1
            elif fb["notification_sent"] and not fb["action_performed"]:
                day_fn += 1
            elif not fb["notification_sent"] and fb["action_performed"]:
                day_fp += 1
            else:
                day_vn += 1
        
        # Atualiza histórico
        previous_activities_total += day_actions
        if day_actions > 0:
            last_completed = True
            consecutive_rest_days = 0
        else:
            consecutive_rest_days += 1
        
        # Acumula estatísticas
        total_notifications += day_notifications
        total_actions += day_actions
        total_vp += day_vp
        total_fn += day_fn
        total_fp += day_fp
        total_vn += day_vn
        
        # Log da variação
        variations_log.append({
            "day": day_num + 1,
            "date": current_date.isoformat(),
            "variation": day_profile["variation_type"],
            "is_weekend": is_weekend,
            "notifications": day_notifications,
            "actions": day_actions
        })
        
        # Print do dia
        training_str = ",".join([f"{h}h" for h in day_training_hours]) if day_training_hours else "-"
        var_emoji = {
            "normal": "✅",
            "folga": "🛋️",
            "noturno": "🌙",
            "sono_ruim": "😴",
            "fds_tardio": "🏖️",
            "alta_motivacao": "🔥"
        }
        
        print(f" {day_num+1:2d}  | {current_date} | {day_of_week} | "
              f"{var_emoji.get(day_profile['variation_type'], '')}{day_profile['variation_type']:^13} | "
              f"  {day_notifications}  |   {day_actions}   |  {day_vp} |  {day_fn} | {training_str:^20}")
        
        days_data.append(day_data)
    
    print("-" * 100)
    
    # Estatísticas finais
    stats = {
        "total_days": 30,
        "total_hours": 720,
        "total_notifications": total_notifications,
        "total_actions": total_actions,
        "total_vp": total_vp,
        "total_vn": total_vn,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "activities_by_hour": activities_by_hour,
        "activities_by_period": activities_by_period,
        "avg_actions_per_day": total_actions / 30,
        "success_rate": total_vp / total_notifications * 100 if total_notifications > 0 else 0
    }
    
    # Resumo
    print(f"\n{'='*60}")
    print("📊 RESUMO DO MÊS")
    print(f"{'='*60}")
    
    print(f"\n📱 Notificações:")
    print(f"   - Total enviadas: {total_notifications}")
    print(f"   - Média por dia: {total_notifications/30:.1f}")
    
    print(f"\n✅ Ações (Treinos):")
    print(f"   - Total executadas: {total_actions}")
    print(f"   - Média por dia: {total_actions/30:.1f}")
    
    print(f"\n📈 Matriz de Confusão (Mês):")
    print(f"   - VP (notificou + executou): {total_vp}")
    print(f"   - VN (não notificou + não executou): {total_vn}")
    print(f"   - FP (não notificou + executou): {total_fp}")
    print(f"   - FN (notificou + não executou): {total_fn}")
    
    if total_notifications > 0:
        print(f"\n📊 Métricas:")
        print(f"   - Taxa de sucesso: {total_vp/total_notifications*100:.1f}%")
        precision = total_vp / (total_vp + total_fn) * 100 if (total_vp + total_fn) > 0 else 0
        print(f"   - Precisão: {precision:.1f}%")
    
    # Distribuição por hora
    print(f"\n⏰ Top 5 Horas com mais treinos:")
    sorted_hours = sorted(activities_by_hour.items(), key=lambda x: x[1], reverse=True)[:5]
    for hour, count in sorted_hours:
        bar = "█" * count
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"   {hour:02d}h: {bar} ({count} treinos, {pct:.1f}%)")
    
    print(f"\n🌅 Distribuição por Período:")
    for period, count in activities_by_period.items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        if count > 0:
            print(f"   - {period}: {count} treinos ({pct:.1f}%)")
    
    # Variações aplicadas
    print(f"\n🎲 Variações Aplicadas:")
    variation_counts = {}
    for v in variations_log:
        vtype = v["variation"]
        variation_counts[vtype] = variation_counts.get(vtype, 0) + 1
    for vtype, count in sorted(variation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {vtype}: {count} dias ({count/30*100:.1f}%)")
    
    return {
        "user_id": user_id,
        "start_date": start_date.isoformat(),
        "end_date": (start_date + timedelta(days=29)).isoformat(),
        "days": days_data,
        "stats": stats,
        "variations": variations_log
    }


def save_month_data(data: Dict[str, Any], output_dir: str = "data/synthetic"):
    """Salva os dados do mês em arquivos JSON."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salva cada dia separadamente
    days_dir = output_path / data["user_id"]
    days_dir.mkdir(exist_ok=True)
    
    for i, day_data in enumerate(data["days"]):
        day_file = days_dir / f"day_{i+1:02d}_{day_data['date']}.json"
        with open(day_file, "w", encoding="utf-8") as f:
            json.dump(day_data, f, indent=2, ensure_ascii=False)
    
    # Salva estatísticas e resumo
    summary = {
        "user_id": data["user_id"],
        "start_date": data["start_date"],
        "end_date": data["end_date"],
        "stats": data["stats"],
        "variations": data["variations"]
    }
    
    summary_file = output_path / f"{data['user_id']}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Salva todos os dias em um único arquivo
    all_days_file = output_path / f"{data['user_id']}_all_days.json"
    with open(all_days_file, "w", encoding="utf-8") as f:
        json.dump(data["days"], f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Dados salvos em:")
    print(f"   - Dias individuais: {days_dir}/")
    print(f"   - Resumo: {summary_file}")
    print(f"   - Todos os dias: {all_days_file}")
    
    return {
        "days_dir": str(days_dir),
        "summary_file": str(summary_file),
        "all_days_file": str(all_days_file)
    }


if __name__ == "__main__":
    # Gera dados de 1 mês
    month_data = generate_month_data(
        user_id="user_matinal_30dias",
        start_date=date(2025, 11, 21),  # 1 mês atrás
        seed=42
    )
    
    # Salva os dados
    files = save_month_data(month_data)
    
    print(f"\n✅ Geração completa!")
    print(f"   - Total de dias: {len(month_data['days'])}")
    print(f"   - Total de horas: {len(month_data['days']) * 24}")
