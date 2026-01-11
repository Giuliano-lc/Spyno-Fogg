"""
Script Principal de Simulação FBM-Based

Este script:
1. Gera dados sintéticos do perfil matinal (sem decisão de notificação)
2. Simula notificações baseadas em FBM e threshold dinâmico
3. Simula respostas do usuário baseadas em níveis de FBM
4. Gera análise completa e validação do sistema
"""

import json
from pathlib import Path
from datetime import date, timedelta

from synthetic_data_generator import SyntheticDataGenerator, PERFIL_MATINAL
from fbm_simulation import FBMSimulator, save_simulation_results


def generate_synthetic_data_for_simulation(
    num_days: int = 30,
    user_id: str = "user_matinal_sim",
    seed: int = 42
) -> list:
    """
    Gera dados sintéticos com FBM calculado mas SEM decisão de notificação.
    O sistema decidirá baseado no threshold.
    """
    
    print(f"{'='*100}")
    print(f"🌅 GERANDO DADOS SINTÉTICOS - Perfil Matinal")
    print(f"{'='*100}\n")
    print(f"📋 Configuração:")
    print(f"   - Perfil: {PERFIL_MATINAL.name}")
    print(f"   - Horas preferidas: {PERFIL_MATINAL.preferred_hours}")
    print(f"   - Dias a gerar: {num_days}")
    print(f"   - Estratégia: FBM-based (sistema decide quando notificar)")
    print(f"\n{'='*100}\n")
    
    generator = SyntheticDataGenerator(PERFIL_MATINAL, seed=seed)
    
    start_date = date.today() - timedelta(days=num_days)
    days_data = []
    
    activities_total = 0
    last_completed = True
    
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        
        day_data = generator.generate_day(
            user_id=user_id,
            target_date=current_date,
            previous_activities_total=activities_total,
            last_completed=last_completed,
            notification_strategy="fbm_based"  # Sistema decide
        )
        
        days_data.append(day_data)
        
        # Conta atividades (None porque sistema ainda não decidiu)
        for hour in day_data["hours"]:
            if hour["feedback"]["action_performed"] is True:
                activities_total += 1
        
        if (i + 1) % 10 == 0:
            print(f"✅ Gerados {i + 1}/{num_days} dias")
    
    print(f"\n✅ Geração completa: {num_days} dias, {num_days * 24} horas\n")
    
    return days_data


def run_full_simulation(
    num_days: int = 30,
    user_id: str = "user_matinal_sim",
    initial_threshold: float = 15.0,
    seed: int = 42
):
    """
    Executa simulação completa:
    1. Gera dados sintéticos
    2. Simula notificações e respostas
    3. Analisa resultados
    """
    
    print(f"\n{'#'*100}")
    print(f"# SIMULAÇÃO COMPLETA - Sistema de Notificação Baseado em FBM")
    print(f"{'#'*100}\n")
    
    # ETAPA 1: Gera dados sintéticos
    print("📋 ETAPA 1: Geração de Dados Sintéticos\n")
    days_data = generate_synthetic_data_for_simulation(
        num_days=num_days,
        user_id=user_id,
        seed=seed
    )
    
    # Salva dados brutos
    output_dir = Path("data/simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data_file = output_dir / f"{user_id}_raw_data.json"
    with open(raw_data_file, "w", encoding="utf-8") as f:
        json.dump(days_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Dados brutos salvos: {raw_data_file}\n")
    
    # ETAPA 2: Simula notificações e respostas
    print(f"\n{'='*100}")
    print("📋 ETAPA 2: Simulação de Notificações e Respostas")
    print(f"{'='*100}\n")
    
    simulator = FBMSimulator(
        user_id=user_id,
        initial_threshold=initial_threshold,
        seed=seed
    )
    
    result = simulator.simulate_multiple_days(days_data)
    
    # ETAPA 3: Salva resultados
    print("\n📋 ETAPA 3: Salvando Resultados\n")
    result_file = save_simulation_results(result, output_dir=str(output_dir))
    
    # ETAPA 4: Análise e Conclusões
    print(f"\n{'='*100}")
    print("📋 ETAPA 4: Análise e Conclusões")
    print(f"{'='*100}\n")
    
    print_final_analysis(result)
    
    return result


def print_final_analysis(result):
    """Imprime análise final e conclusões."""
    
    print("🎯 CONCLUSÕES DA SIMULAÇÃO:\n")
    
    # Efetividade do sistema
    print("1️⃣ Efetividade do Sistema de Threshold Dinâmico:")
    print(f"   - O threshold {'aumentou' if result.threshold_change > 0 else 'diminuiu'} "
          f"{abs(result.threshold_change):.2f} pontos")
    print(f"   - Mudança relativa: {result.threshold_change/result.initial_threshold*100:+.1f}%")
    
    if result.threshold_change > 0:
        print("   ✅ Sistema identificou que usuário é mais exigente/seletivo")
    elif result.threshold_change < 0:
        print("   ✅ Sistema identificou que threshold inicial estava alto demais")
    else:
        print("   ✅ Sistema encontrou threshold ideal rapidamente")
    
    # Qualidade das notificações
    print(f"\n2️⃣ Qualidade das Notificações:")
    print(f"   - Precisão: {result.precision*100:.1f}%")
    
    if result.precision >= 0.7:
        print("   ✅ EXCELENTE: Mais de 70% das notificações resultam em ação")
    elif result.precision >= 0.5:
        print("   ⚠️ BOM: Mais de 50% das notificações são efetivas")
    else:
        print("   ❌ PRECISA MELHORAR: Muitas notificações ignoradas")
    
    # Cobertura
    print(f"\n3️⃣ Cobertura de Oportunidades:")
    print(f"   - Recall: {result.recall*100:.1f}%")
    print(f"   - Falsos Positivos (FP): {result.fp_count}")
    
    if result.recall >= 0.8:
        print("   ✅ EXCELENTE: Sistema captura maioria das oportunidades")
    elif result.recall >= 0.6:
        print("   ⚠️ BOM: Sistema perde algumas oportunidades")
    else:
        print(f"   ❌ PRECISA MELHORAR: Muitas oportunidades perdidas ({result.fp_count} FPs)")
    
    # Balanceamento
    print(f"\n4️⃣ Balanceamento (F1-Score):")
    print(f"   - F1-Score: {result.f1_score*100:.1f}%")
    
    if result.f1_score >= 0.7:
        print("   ✅ EXCELENTE: Bom equilíbrio entre precisão e cobertura")
    elif result.f1_score >= 0.5:
        print("   ⚠️ BOM: Sistema razoavelmente balanceado")
    else:
        print("   ❌ PRECISA MELHORAR: Desbalanceamento entre precisão/recall")
    
    # FBM insights
    print(f"\n5️⃣ Insights de FBM:")
    print(f"   - FBM médio quando notificou: {result.avg_fbm_when_notified:.1f}")
    print(f"   - FBM médio quando respondeu: {result.avg_fbm_when_responded:.1f}")
    print(f"   - FBM médio quando ignorou:   {result.avg_fbm_when_ignored:.1f}")
    
    fbm_diff = result.avg_fbm_when_responded - result.avg_fbm_when_ignored
    print(f"   - Diferença: {fbm_diff:+.1f} pontos")
    
    if fbm_diff > 10:
        print("   ✅ Sistema consegue distinguir bem momentos propícios")
    else:
        print("   ⚠️ Pouca diferença entre FBM de resposta e ignorado")
    
    # Padrões horários
    print(f"\n6️⃣ Validação do Perfil Matinal:")
    top_hours = sorted(result.hourly_stats.items(), 
                      key=lambda x: x[1]["responded"], reverse=True)[:3]
    top_hour_numbers = [h[0] for h in top_hours if h[1]["responded"] > 0]
    
    print(f"   - Horas com mais respostas: {top_hour_numbers}")
    print(f"   - Horas preferidas do perfil: {[6, 7, 8]}")
    
    matches = len(set(top_hour_numbers) & set([6, 7, 8]))
    if matches >= 2:
        print(f"   ✅ VALIDADO: Sistema identificou corretamente padrão matinal ({matches}/3 horas)")
    else:
        print(f"   ⚠️ Sistema não capturou bem o padrão matinal")
    
    # Recomendações
    print(f"\n7️⃣ Recomendações para Produção:")
    
    if result.precision < 0.6:
        print("   - ⚠️ Aumentar threshold inicial para reduzir notificações desperdiçadas")
    
    if result.recall < 0.7:
        print("   - ⚠️ Diminuir threshold inicial para capturar mais oportunidades")
    
    if result.vn_count > result.vp_count:
        print("   - ⚠️ Muitas notificações ignoradas - ajustar algoritmo de resposta")
    
    if result.fp_count > result.vp_count * 0.3:
        print("   - ⚠️ Muitas ações sem notificação - threshold muito conservador")
    
    if result.f1_score >= 0.7 and abs(result.threshold_change) < 5:
        print("   - ✅ Sistema está bem calibrado e estável")
    
    print(f"\n{'='*100}")
    print("🎉 ANÁLISE COMPLETA!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    # Configuração da simulação
    CONFIG = {
        "num_days": 30,
        "user_id": "user_matinal_sim",
        "initial_threshold": 40.0,  # Ajustado de 15.0 para 40.0 baseado em análise
        "seed": 42
    }
    
    print("\n🚀 Iniciando Simulação FBM-Based\n")
    print(f"Configuração:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Executa simulação
    result = run_full_simulation(**CONFIG)
    
    print("\n✅ Simulação finalizada com sucesso!")
    print(f"   Precision: {result.precision*100:.1f}%")
    print(f"   Recall:    {result.recall*100:.1f}%")
    print(f"   F1-Score:  {result.f1_score*100:.1f}%")
    print(f"   Threshold: {result.initial_threshold:.2f} → {result.final_threshold:.2f}\n")
