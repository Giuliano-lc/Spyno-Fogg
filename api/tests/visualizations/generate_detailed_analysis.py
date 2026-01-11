"""
Script para gerar análise detalhada dia-a-dia da simulação.
Ideal para inclusão no TCC.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_simulation_data():
    """Carrega dados da simulação."""
    data_path = Path("data/simulation/user_matinal_rl_raw_data.json")
    
    if not data_path.exists():
        print("❌ Arquivo não encontrado. Execute run_simulation_with_rl_v2.py primeiro.")
        return None
    
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_day(day_data: Dict, day_num: int) -> Dict[str, Any]:
    """Analisa um dia específico em detalhes."""
    
    # Contadores
    notif_sent = 0
    actions_performed = 0
    vp = vn = fp = fn = 0
    
    # FBM scores
    fbm_when_notified = []
    fbm_when_responded = []
    fbm_when_ignored = []
    
    # Distribuição horária
    hourly_activity = {}
    
    for hour_data in day_data["hours"]:
        hour = hour_data["hour"]
        sleeping = hour_data["trigger_factors"].get("sleeping", False)
        
        if sleeping:
            continue
        
        # Calcula FBM
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
        
        # Simula decisão (threshold = 40 inicial)
        # Esta é uma simulação simplificada
        notified = fbm_score >= 40
        responded = fbm_score >= 35  # Simplificado
        
        if notified:
            notif_sent += 1
            fbm_when_notified.append(fbm_score)
            if responded:
                vp += 1
                fbm_when_responded.append(fbm_score)
            else:
                vn += 1
                fbm_when_ignored.append(fbm_score)
        else:
            if responded:
                fp += 1
                fbm_when_responded.append(fbm_score)
            else:
                fn += 1
        
        if responded:
            actions_performed += 1
        
        # Distribuição horária
        hourly_activity[hour] = {
            "fbm": fbm_score,
            "notified": notified,
            "responded": responded,
            "m": m, "a": a, "t": t
        }
    
    return {
        "day_num": day_num,
        "date": day_data["date"],
        "notif_sent": notif_sent,
        "actions_performed": actions_performed,
        "vp": vp,
        "vn": vn,
        "fp": fp,
        "fn": fn,
        "avg_fbm_notified": sum(fbm_when_notified) / len(fbm_when_notified) if fbm_when_notified else 0,
        "avg_fbm_responded": sum(fbm_when_responded) / len(fbm_when_responded) if fbm_when_responded else 0,
        "avg_fbm_ignored": sum(fbm_when_ignored) / len(fbm_when_ignored) if fbm_when_ignored else 0,
        "hourly_activity": hourly_activity,
        "precision": (vp / (vp + vn) * 100) if (vp + vn) > 0 else 0,
        "recall": (vp / (vp + fp) * 100) if (vp + fp) > 0 else 0
    }


def generate_markdown_report(analyses: List[Dict]) -> str:
    """Gera relatório detalhado em Markdown."""
    
    report = []
    report.append("# 📊 Análise Detalhada da Simulação - Dia a Dia\n")
    report.append(f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    report.append(f"**Total de dias analisados:** {len(analyses)}\n")
    report.append("---\n")
    
    # Resumo geral
    report.append("\n## 📈 Resumo Geral\n")
    
    total_notif = sum(a["notif_sent"] for a in analyses)
    total_actions = sum(a["actions_performed"] for a in analyses)
    total_vp = sum(a["vp"] for a in analyses)
    total_vn = sum(a["vn"] for a in analyses)
    total_fp = sum(a["fp"] for a in analyses)
    total_fn = sum(a["fn"] for a in analyses)
    
    overall_precision = (total_vp / (total_vp + total_vn) * 100) if (total_vp + total_vn) > 0 else 0
    overall_recall = (total_vp / (total_vp + total_fp) * 100) if (total_vp + total_fp) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0
    
    report.append(f"- **Total de notificações enviadas:** {total_notif}\n")
    report.append(f"- **Total de ações executadas:** {total_actions}\n")
    report.append(f"- **Taxa de resposta:** {(total_actions / total_notif * 100):.1f}%\n")
    report.append(f"- **Precision geral:** {overall_precision:.1f}%\n")
    report.append(f"- **Recall geral:** {overall_recall:.1f}%\n")
    report.append(f"- **F1-Score geral:** {overall_f1:.1f}%\n")
    
    # Métricas por semana
    report.append("\n## 📅 Evolução Semanal\n")
    
    weeks = [analyses[i:i+7] for i in range(0, len(analyses), 7)]
    
    for week_num, week in enumerate(weeks, 1):
        week_notif = sum(a["notif_sent"] for a in week)
        week_vp = sum(a["vp"] for a in week)
        week_vn = sum(a["vn"] for a in week)
        week_precision = (week_vp / (week_vp + week_vn) * 100) if (week_vp + week_vn) > 0 else 0
        
        report.append(f"\n### Semana {week_num} (Dias {week[0]['day_num']}-{week[-1]['day_num']})\n")
        report.append(f"- Notificações: {week_notif}\n")
        report.append(f"- VP: {week_vp}, VN: {week_vn}\n")
        report.append(f"- Precision: {week_precision:.1f}%\n")
    
    # Análise dia a dia
    report.append("\n---\n")
    report.append("\n## 📋 Análise Dia a Dia\n")
    
    for analysis in analyses:
        report.append(f"\n### Dia {analysis['day_num']} - {analysis['date']}\n")
        report.append(f"- **Notificações enviadas:** {analysis['notif_sent']}\n")
        report.append(f"- **Ações executadas:** {analysis['actions_performed']}\n")
        report.append(f"- **Matriz de Confusão:** VP={analysis['vp']}, VN={analysis['vn']}, FP={analysis['fp']}, FN={analysis['fn']}\n")
        report.append(f"- **Precision:** {analysis['precision']:.1f}%\n")
        report.append(f"- **Recall:** {analysis['recall']:.1f}%\n")
        
        if analysis['avg_fbm_notified'] > 0:
            report.append(f"- **FBM médio quando notificado:** {analysis['avg_fbm_notified']:.1f}\n")
        if analysis['avg_fbm_responded'] > 0:
            report.append(f"- **FBM médio quando respondeu:** {analysis['avg_fbm_responded']:.1f}\n")
        
        # Horas mais ativas
        if analysis['hourly_activity']:
            active_hours = sorted(
                [(h, data) for h, data in analysis['hourly_activity'].items() if data['responded']],
                key=lambda x: x[1]['fbm'],
                reverse=True
            )[:3]
            
            if active_hours:
                report.append(f"- **Horas mais ativas:**\n")
                for hour, data in active_hours:
                    report.append(f"  - **{hour:02d}h**: FBM={data['fbm']} (M={data['m']}, A={data['a']}, T={data['t']})\n")
    
    # Insights
    report.append("\n---\n")
    report.append("\n## 💡 Insights e Observações\n")
    
    # Dias com melhor precision
    best_precision_days = sorted(analyses, key=lambda x: x['precision'], reverse=True)[:5]
    report.append("\n### 🏆 Top 5 Dias com Melhor Precision\n")
    for day in best_precision_days:
        if day['precision'] > 0:
            report.append(f"- Dia {day['day_num']} ({day['date']}): {day['precision']:.1f}% (VP={day['vp']}, VN={day['vn']})\n")
    
    # Dias com mais atividade
    most_active_days = sorted(analyses, key=lambda x: x['actions_performed'], reverse=True)[:5]
    report.append("\n### 🎯 Top 5 Dias com Mais Atividade\n")
    for day in most_active_days:
        if day['actions_performed'] > 0:
            report.append(f"- Dia {day['day_num']} ({day['date']}): {day['actions_performed']} ações\n")
    
    # Padrão horário geral
    report.append("\n### ⏰ Padrão Horário Geral\n")
    
    hourly_totals = {}
    for analysis in analyses:
        for hour, data in analysis['hourly_activity'].items():
            if hour not in hourly_totals:
                hourly_totals[hour] = {"notif": 0, "resp": 0, "fbm_sum": 0, "count": 0}
            
            hourly_totals[hour]["count"] += 1
            hourly_totals[hour]["fbm_sum"] += data["fbm"]
            if data["notified"]:
                hourly_totals[hour]["notif"] += 1
            if data["responded"]:
                hourly_totals[hour]["resp"] += 1
    
    top_response_hours = sorted(
        [(h, data["resp"], data["fbm_sum"] / data["count"]) for h, data in hourly_totals.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    report.append("\n**Top 10 horas com mais respostas:**\n")
    for hour, responses, avg_fbm in top_response_hours:
        report.append(f"- **{hour:02d}h**: {responses} respostas, FBM médio: {avg_fbm:.1f}\n")
    
    return "".join(report)


def main():
    """Função principal."""
    print("="*80)
    print("📊 GERADOR DE ANÁLISE DETALHADA - DIA A DIA")
    print("="*80)
    print()
    
    # Carrega dados
    print("📂 Carregando dados da simulação...")
    days_data = load_simulation_data()
    
    if not days_data:
        return
    
    print(f"✅ {len(days_data)} dias carregados\n")
    
    # Analisa cada dia
    print("🔍 Analisando cada dia...")
    analyses = []
    
    for i, day_data in enumerate(days_data, 1):
        analysis = analyze_day(day_data, i)
        analyses.append(analysis)
        
        if i % 10 == 0:
            print(f"   Processados {i}/{len(days_data)} dias...")
    
    print(f"✅ Análise completa!\n")
    
    # Gera relatório
    print("📝 Gerando relatório Markdown...")
    report = generate_markdown_report(analyses)
    
    # Salva
    output_path = Path("data/simulation/ANALISE_DETALHADA_DIA_A_DIA.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✅ Relatório salvo: {output_path}\n")
    
    # Estatísticas
    print("="*80)
    print("📊 RESUMO DA ANÁLISE")
    print("="*80)
    print(f"Total de dias: {len(analyses)}")
    print(f"Total de notificações: {sum(a['notif_sent'] for a in analyses)}")
    print(f"Total de ações: {sum(a['actions_performed'] for a in analyses)}")
    print(f"VP total: {sum(a['vp'] for a in analyses)}")
    print(f"VN total: {sum(a['vn'] for a in analyses)}")
    print()
    print(f"📄 Relatório completo disponível em:")
    print(f"   {output_path.absolute()}")
    print()


if __name__ == "__main__":
    main()
