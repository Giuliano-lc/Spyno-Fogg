"""
Análise dos resultados da Epoch 1 - Simulação de Shift de Comportamento
Baseado nos logs gerados da execução.
"""

import json
from pathlib import Path

# Dados extraídos dos logs da Epoch 1
EPOCH1_DATA = {
    "fase1_matinal": [
        # (dia, VP, VN, FP, epsilon)
        (1, 1, 2, 3, 0.294),
        (2, 0, 1, 2, 0.288),
        (3, 1, 2, 3, 0.282),
        (4, 3, 1, 1, 0.277),
        (5, 0, 1, 1, 0.271),
        (6, 0, 0, 5, 0.266),
        (7, 2, 0, 3, 0.260),
        (8, 0, 0, 5, 0.255),
        (9, 2, 1, 3, 0.250),
        (10, 1, 1, 1, 0.245),
        (11, 3, 0, 1, 0.240),
        (12, 4, 4, 0, 0.235),
        (13, 1, 1, 3, 0.231),
        (14, 2, 4, 1, 0.226),
        (15, 2, 0, 0, 0.222),
        (16, 4, 1, 1, 0.217),
        (17, 5, 1, 0, 0.200),
        (18, 4, 3, 0, 0.196),
        (19, 6, 1, 0, 0.180),
        (20, 3, 1, 1, 0.177),
        (21, 3, 0, 2, 0.173),
        (22, 3, 1, 0, 0.170),
        (23, 5, 1, 1, 0.156),
        (24, 2, 2, 1, 0.153),
        (25, 4, 1, 1, 0.150),
        (26, 4, 3, 1, 0.147),
        (27, 3, 1, 2, 0.144),
        (28, 4, 1, 2, 0.141),
        (29, 4, 1, 1, 0.138),
        (30, 6, 2, 0, 0.127),
    ],
    "fase2_noturno": [
        # Dias 31-90 (após shift)
        (31, 2, 6, 3, 0.500),  # SHIFT DETECTADO!
        (32, 0, 7, 2, 0.490),
        (33, 1, 5, 2, 0.480),
        (34, 1, 4, 3, 0.471),
        (35, 3, 5, 2, 0.461),
        (36, 2, 2, 1, 0.452),
        (37, 1, 4, 5, 0.443),
        (38, 1, 4, 4, 0.434),
        (39, 0, 7, 2, 0.425),
        (40, 2, 3, 1, 0.417),
        (41, 0, 6, 4, 0.409),
        (42, 0, 5, 1, 0.400),
        (43, 1, 6, 2, 0.392),
        (44, 0, 1, 2, 0.385),
        (45, 1, 3, 4, 0.377),
        (46, 0, 2, 1, 0.369),
        (47, 0, 2, 5, 0.362),
        (48, 1, 2, 3, 0.355),
        (49, 2, 2, 1, 0.348),
        (50, 1, 4, 2, 0.341),
        (51, 0, 3, 2, 0.334),
        (52, 1, 0, 0, 0.327),
        (53, 2, 1, 2, 0.321),
        (54, 0, 4, 5, 0.314),
        (55, 2, 1, 1, 0.308),
        (56, 1, 2, 1, 0.302),
        (57, 2, 2, 1, 0.296),
        (58, 2, 1, 3, 0.290),
        (59, 1, 1, 1, 0.284),
        (60, 1, 1, 2, 0.278),
        (61, 0, 3, 2, 0.500),  # Falso shift detectado
        (62, 0, 3, 2, 0.490),
        (63, 2, 4, 4, 0.480),
        (64, 0, 2, 2, 0.471),
        (65, 0, 4, 4, 0.461),
        (66, 0, 3, 1, 0.452),
        (67, 2, 2, 3, 0.443),
        (68, 2, 2, 0, 0.434),
        (69, 1, 3, 5, 0.425),
        (70, 2, 1, 2, 0.417),
        (71, 1, 0, 0, 0.409),
        (72, 1, 1, 2, 0.400),
        (73, 1, 3, 3, 0.392),
        (74, 2, 5, 1, 0.385),
        (75, 1, 2, 3, 0.377),
        (76, 1, 3, 3, 0.369),
        (77, 0, 3, 2, 0.362),
        (78, 2, 1, 4, 0.355),
        (79, 3, 2, 0, 0.348),
        (80, 1, 2, 2, 0.341),
        (81, 0, 2, 4, 0.334),
        (82, 0, 1, 5, 0.327),
        (83, 1, 2, 3, 0.321),
        (84, 0, 2, 3, 0.314),
        (85, 3, 0, 3, 0.308),
        (86, 2, 4, 2, 0.302),
        (87, 0, 2, 5, 0.296),
        (88, 1, 3, 1, 0.290),
        (89, 0, 5, 2, 0.500),  # Falso shift detectado
        (90, 3, 3, 1, 0.490),
    ]
}


def analyze_phase(data, phase_name):
    """Analisa uma fase da simulação."""
    total_vp = sum(d[1] for d in data)
    total_vn = sum(d[2] for d in data)
    total_fp = sum(d[3] for d in data)
    total_fn = len(data) * 18 - total_vp - total_vn - total_fp  # ~18 horas ativas por dia
    
    precision = (total_vp / (total_vp + total_vn) * 100) if (total_vp + total_vn) > 0 else 0
    recall = (total_vp / (total_vp + total_fp) * 100) if (total_vp + total_fp) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    # Evolução temporal
    first_10 = data[:10]
    last_10 = data[-10:]
    
    vp_first = sum(d[1] for d in first_10)
    vp_last = sum(d[1] for d in last_10)
    vn_first = sum(d[2] for d in first_10)
    vn_last = sum(d[2] for d in last_10)
    
    return {
        "phase": phase_name,
        "days": len(data),
        "total_vp": total_vp,
        "total_vn": total_vn,
        "total_fp": total_fp,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "epsilon_start": data[0][4],
        "epsilon_end": data[-1][4],
        "evolution": {
            "first_10_days": {"vp": vp_first, "vn": vn_first, "avg_vp": vp_first/10, "avg_vn": vn_first/10},
            "last_10_days": {"vp": vp_last, "vn": vn_last, "avg_vp": vp_last/10, "avg_vn": vn_last/10}
        }
    }


def print_analysis():
    """Imprime análise completa."""
    
    print("\n" + "=" * 100)
    print("📊 ANÁLISE DA SIMULAÇÃO DE SHIFT DE COMPORTAMENTO - EPOCH 1")
    print("=" * 100)
    
    # Análise Fase 1 (Matinal)
    fase1 = analyze_phase(EPOCH1_DATA["fase1_matinal"], "Fase 1 - MATINAL")
    
    print(f"\n{'─' * 100}")
    print(f"☀️  FASE 1: PERFIL MATINAL (Dias 1-30)")
    print(f"{'─' * 100}")
    print(f"\n📈 Métricas Globais:")
    print(f"   VP (acertos): {fase1['total_vp']}")
    print(f"   VN (erros):   {fase1['total_vn']}")
    print(f"   FP (perdidos): {fase1['total_fp']}")
    print(f"   Precision: {fase1['precision']:.1f}%")
    print(f"   Recall:    {fase1['recall']:.1f}%")
    print(f"   F1-Score:  {fase1['f1_score']:.1f}%")
    
    print(f"\n📉 Evolução do Epsilon:")
    print(f"   Início: {fase1['epsilon_start']:.3f}")
    print(f"   Final:  {fase1['epsilon_end']:.3f}")
    print(f"   Redução: {((fase1['epsilon_start'] - fase1['epsilon_end']) / fase1['epsilon_start'] * 100):.1f}%")
    
    print(f"\n📊 Evolução Temporal:")
    print(f"   Primeiros 10 dias: VP médio = {fase1['evolution']['first_10_days']['avg_vp']:.1f}, VN médio = {fase1['evolution']['first_10_days']['avg_vn']:.1f}")
    print(f"   Últimos 10 dias:   VP médio = {fase1['evolution']['last_10_days']['avg_vp']:.1f}, VN médio = {fase1['evolution']['last_10_days']['avg_vn']:.1f}")
    
    vp_improvement = fase1['evolution']['last_10_days']['avg_vp'] - fase1['evolution']['first_10_days']['avg_vp']
    print(f"   Melhoria VP: {'+' if vp_improvement >= 0 else ''}{vp_improvement:.1f} VP/dia")
    
    if fase1['evolution']['last_10_days']['avg_vp'] >= 3:
        print(f"\n   ✅ SUCESSO! Modelo APRENDEU o padrão matinal")
    else:
        print(f"\n   ⚠️ Modelo ainda aprendendo...")
    
    # Análise Fase 2 (Noturno)
    fase2 = analyze_phase(EPOCH1_DATA["fase2_noturno"], "Fase 2 - NOTURNO")
    
    print(f"\n{'─' * 100}")
    print(f"🌙 FASE 2: PERFIL NOTURNO (Dias 31-90)")
    print(f"{'─' * 100}")
    print(f"\n📈 Métricas Globais:")
    print(f"   VP (acertos): {fase2['total_vp']}")
    print(f"   VN (erros):   {fase2['total_vn']}")
    print(f"   FP (perdidos): {fase2['total_fp']}")
    print(f"   Precision: {fase2['precision']:.1f}%")
    print(f"   Recall:    {fase2['recall']:.1f}%")
    print(f"   F1-Score:  {fase2['f1_score']:.1f}%")
    
    print(f"\n🔄 Detecção de Shift:")
    print(f"   ✅ Shift DETECTADO corretamente no dia 31")
    print(f"   ⚠️ Falsos positivos: dias 61 e 89 (re-detectou shift)")
    
    print(f"\n📊 Evolução Temporal:")
    print(f"   Primeiros 10 dias após shift: VP médio = {fase2['evolution']['first_10_days']['avg_vp']:.1f}, VN médio = {fase2['evolution']['first_10_days']['avg_vn']:.1f}")
    print(f"   Últimos 10 dias:              VP médio = {fase2['evolution']['last_10_days']['avg_vp']:.1f}, VN médio = {fase2['evolution']['last_10_days']['avg_vn']:.1f}")
    
    vn_reduction = fase2['evolution']['first_10_days']['avg_vn'] - fase2['evolution']['last_10_days']['avg_vn']
    print(f"   Redução VN: {vn_reduction:.1f} VN/dia")
    
    # Comparação Fases
    print(f"\n{'─' * 100}")
    print(f"📊 COMPARAÇÃO ENTRE FASES")
    print(f"{'─' * 100}")
    
    print(f"\n{'Métrica':<20} | {'Fase 1 (Matinal)':<20} | {'Fase 2 (Noturno)':<20} | {'Diferença':<15}")
    print(f"{'-' * 80}")
    print(f"{'Precision':<20} | {fase1['precision']:>18.1f}% | {fase2['precision']:>18.1f}% | {fase2['precision'] - fase1['precision']:>+13.1f}%")
    print(f"{'Recall':<20} | {fase1['recall']:>18.1f}% | {fase2['recall']:>18.1f}% | {fase2['recall'] - fase1['recall']:>+13.1f}%")
    print(f"{'F1-Score':<20} | {fase1['f1_score']:>18.1f}% | {fase2['f1_score']:>18.1f}% | {fase2['f1_score'] - fase1['f1_score']:>+13.1f}%")
    print(f"{'VP Total':<20} | {fase1['total_vp']:>19} | {fase2['total_vp']:>19} | {fase2['total_vp'] - fase1['total_vp']:>+14}")
    print(f"{'VN Total':<20} | {fase1['total_vn']:>19} | {fase2['total_vn']:>19} | {fase2['total_vn'] - fase1['total_vn']:>+14}")
    
    # Conclusões
    print(f"\n{'=' * 100}")
    print(f"🎯 CONCLUSÕES")
    print(f"{'=' * 100}")
    
    print(f"""
1. APRENDIZADO INICIAL (Fase 1 - Matinal):
   ✅ O modelo APRENDEU o padrão matinal com sucesso
   - VP aumentou de ~1.0/dia para ~3.9/dia
   - Epsilon decaiu corretamente (0.294 → 0.127)
   - Precision final: {fase1['precision']:.1f}%

2. DETECÇÃO DE SHIFT:
   ✅ O sistema DETECTOU o shift corretamente no dia 31
   - VP caiu de 6 para 2
   - VN subiu de 2 para 6
   - Epsilon foi boosted para 0.500

3. ADAPTAÇÃO AO NOVO PADRÃO (Fase 2 - Noturno):
   ⚠️ O modelo teve DIFICULDADE em se adaptar
   - VP médio na fase 2: {fase2['total_vp']/60:.1f}/dia (vs {fase1['total_vp']/30:.1f}/dia na fase 1)
   - VN permaneceu alto: {fase2['total_vn']/60:.1f}/dia
   - FP alto indica que usuário responde mas modelo não notifica

4. PROBLEMAS IDENTIFICADOS:
   ❌ Modelo continua "preso" ao padrão matinal
   ❌ Falsos positivos de shift (dias 61, 89)
   ❌ Precisão caiu de {fase1['precision']:.1f}% para {fase2['precision']:.1f}%

5. POSSÍVEIS MELHORIAS:
   → Aumentar exploration_boost_days de 15 para 20-25
   → Aumentar epsilon_max de 0.50 para 0.60-0.70
   → Penalizar mais fortemente VN em horários com FBM alto
   → Considerar "esquecimento" do padrão antigo após shift
""")
    
    # Salvar dados
    output_data = {
        "epoch": 1,
        "config": {
            "total_days": 90,
            "shift_day": 30,
            "scenario": "Matinal (dias 1-30) → Noturno (dias 31-90)"
        },
        "fase1_matinal": fase1,
        "fase2_noturno": fase2,
        "raw_data": EPOCH1_DATA,
        "shift_detection": {
            "detected": True,
            "detection_day": 31,
            "false_positives": [61, 89]
        },
        "conclusions": {
            "learning_success": True,
            "shift_detection_success": True,
            "adaptation_success": False,
            "main_issue": "Modelo não conseguiu adaptar ao novo padrão noturno"
        }
    }
    
    output_dir = Path(__file__).parent.parent / "data" / "simulation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "shift_behavior_epoch1_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Dados salvos em: {output_file}")
    print(f"\n{'=' * 100}\n")


if __name__ == "__main__":
    print_analysis()
