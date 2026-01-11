"""
Gera gráficos de análise do SHIFT DE COMPORTAMENTO usando dados reais.
Analisa a capacidade do RL de detectar e adaptar-se a mudanças de padrão.

Dados: shift_behavior_epoch1_analysis.json
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import json
from pathlib import Path

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

# Pastas
data_file = Path(__file__).parent.parent / "data" / "simulation" / "shift_behavior_epoch1_analysis.json"
output_dir = Path(__file__).parent.parent / "data" / "simulation" / "graficos_shift_behavior"
output_dir.mkdir(parents=True, exist_ok=True)


def load_data():
    """Carrega dados da análise."""
    if not data_file.exists():
        print(f"❌ Arquivo não encontrado: {data_file}")
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_vp_vn_evolution(data):
    """
    Gráfico 1: EVOLUÇÃO VP E VN AO LONGO DOS 90 DIAS
    Mostra claramente o momento do shift e a dificuldade de adaptação
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Extrai dados
    fase1 = data['raw_data']['fase1_matinal']
    fase2 = data['raw_data']['fase2_noturno']
    
    all_days = fase1 + fase2
    dias = [d[0] for d in all_days]
    vp = [d[1] for d in all_days]
    vn = [d[2] for d in all_days]
    fp = [d[3] for d in all_days]
    epsilon = [d[4] for d in all_days]
    
    # Gráfico 1: VP e VN empilhados
    colors_vp = ['#2ecc71' if d <= 30 else '#27ae60' for d in dias]
    colors_vn = ['#e74c3c' if d <= 30 else '#c0392b' for d in dias]
    
    ax1.bar(dias, vp, label='VP (Acertos)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(dias, vn, bottom=vp, label='VN (Erros/Spam)', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Linha vertical no shift
    ax1.axvline(x=30.5, color='purple', linestyle='--', linewidth=3, label='SHIFT (Dia 30→31)')
    
    # Zonas de fase
    ax1.axvspan(0, 30.5, alpha=0.1, color='yellow', label='Fase 1: Matinal')
    ax1.axvspan(30.5, 91, alpha=0.1, color='blue', label='Fase 2: Noturno')
    
    # Média móvel de VP
    vp_smooth = np.convolve(vp, np.ones(5)/5, mode='same')
    ax1.plot(dias, vp_smooth, color='darkgreen', linewidth=2.5, linestyle='--', label='VP (média móvel)')
    
    ax1.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax1.set_title('Evolução de VP (Acertos) e VN (Erros) ao Longo de 90 Dias\n'
                  '☀️ Fase 1: Matinal (Dias 1-30) | 🌙 Fase 2: Noturno (Dias 31-90)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 91)
    ax1.grid(axis='y', alpha=0.3)
    
    # Anotações
    ax1.annotate('🚨 SHIFT\nDETECTADO!', xy=(31, 8), fontsize=11, fontweight='bold',
                color='purple', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple', alpha=0.9))
    
    ax1.annotate('VP médio: 2.7/dia', xy=(15, max(vp[:30])+2), fontsize=10,
                ha='center', color='green', fontweight='bold')
    ax1.annotate('VP médio: 1.1/dia', xy=(60, max(vp[30:])+2), fontsize=10,
                ha='center', color='darkgreen', fontweight='bold')
    
    # Gráfico 2: Epsilon e Taxa de Acerto
    ax2_twin = ax2.twinx()
    
    # Taxa de acerto por dia
    taxa_acerto = [(v / (v + n) * 100) if (v + n) > 0 else 0 for v, n in zip(vp, vn)]
    
    ax2.plot(dias, taxa_acerto, color='#3498db', linewidth=2, marker='o', markersize=3,
             label='Taxa de Acerto (%)', alpha=0.8)
    ax2.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Meta (70%)')
    
    # Epsilon
    ax2_twin.plot(dias, epsilon, color='#9b59b6', linewidth=2.5, label='Epsilon (Exploração)')
    ax2_twin.fill_between(dias, epsilon, alpha=0.2, color='#9b59b6')
    
    # Shift
    ax2.axvline(x=30.5, color='purple', linestyle='--', linewidth=3)
    
    # Marca os boosts de epsilon
    boost_days = [31, 61, 89]
    for bd in boost_days:
        ax2_twin.annotate('↑ BOOST', xy=(bd, 0.5), fontsize=8, color='purple',
                         ha='center', fontweight='bold')
    
    ax2.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Taxa de Acerto (%)', fontsize=12, fontweight='bold', color='#3498db')
    ax2_twin.set_ylabel('Epsilon', fontsize=12, fontweight='bold', color='#9b59b6')
    
    ax2.set_title('Taxa de Acerto e Exploração (Epsilon) ao Longo do Tempo\n'
                  'Epsilon aumenta quando shift é detectado para re-explorar',
                  fontsize=13, fontweight='bold', pad=15)
    
    ax2.set_xlim(0, 91)
    ax2.set_ylim(0, 100)
    ax2_twin.set_ylim(0, 0.6)
    
    # Legendas combinadas
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph1_vp_vn_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 1: Evolução VP/VN ao longo dos 90 dias")


def plot_phase_comparison(data):
    """
    Gráfico 2: COMPARAÇÃO ENTRE FASES
    Mostra métricas lado a lado
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fase1 = data['fase1_matinal']
    fase2 = data['fase2_noturno']
    
    # 1. Precision e Recall
    ax1 = axes[0, 0]
    metrics = ['Precision', 'Recall', 'F1-Score']
    fase1_vals = [fase1['precision'], fase1['recall'], fase1['f1_score']]
    fase2_vals = [fase2['precision'], fase2['recall'], fase2['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, fase1_vals, width, label='Fase 1 (Matinal)', color='#f39c12', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, fase2_vals, width, label='Fase 2 (Noturno)', color='#3498db', alpha=0.8, edgecolor='black')
    
    ax1.axhline(y=70, color='green', linestyle='--', linewidth=2, label='Meta (70%)')
    
    # Valores nas barras
    for bar, val in zip(bars1, fase1_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%',
                ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, fase2_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%',
                ha='center', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Porcentagem (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Comparação de Métricas entre Fases\n'
                  '☀️ Matinal: SUCESSO | 🌙 Noturno: FALHA',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. VP, VN, FP totais
    ax2 = axes[0, 1]
    outcomes = ['VP\n(Acertos)', 'VN\n(Erros)', 'FP\n(Perdidos)']
    fase1_outcomes = [fase1['total_vp'], fase1['total_vn'], fase1['total_fp']]
    fase2_outcomes = [fase2['total_vp'], fase2['total_vn'], fase2['total_fp']]
    
    x = np.arange(len(outcomes))
    
    bars1 = ax2.bar(x - width/2, fase1_outcomes, width, label='Fase 1 (Matinal)', 
                    color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, fase2_outcomes, width, label='Fase 2 (Noturno)',
                    color=['#27ae60', '#c0392b', '#d68910'], alpha=0.7, edgecolor='black')
    
    # Valores
    for i, (v1, v2) in enumerate(zip(fase1_outcomes, fase2_outcomes)):
        ax2.text(i - width/2, v1 + 3, str(v1), ha='center', fontsize=10, fontweight='bold')
        ax2.text(i + width/2, v2 + 3, str(v2), ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Quantidade', fontsize=11, fontweight='bold')
    ax2.set_title('Totais de VP, VN e FP por Fase\n'
                  'VN explodiu na Fase 2 (+131)',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcomes)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Evolução temporal dentro de cada fase
    ax3 = axes[1, 0]
    
    labels = ['Primeiros\n10 dias', 'Últimos\n10 dias']
    fase1_vp_evol = [fase1['evolution']['first_10_days']['avg_vp'], fase1['evolution']['last_10_days']['avg_vp']]
    fase2_vp_evol = [fase2['evolution']['first_10_days']['avg_vp'], fase2['evolution']['last_10_days']['avg_vp']]
    
    x = np.arange(len(labels))
    
    ax3.bar(x - width/2, fase1_vp_evol, width, label='Fase 1 (Matinal)', color='#f39c12', alpha=0.8, edgecolor='black')
    ax3.bar(x + width/2, fase2_vp_evol, width, label='Fase 2 (Noturno)', color='#3498db', alpha=0.8, edgecolor='black')
    
    # Setas de melhoria/piora
    ax3.annotate('', xy=(0.5, fase1_vp_evol[1]), xytext=(0, fase1_vp_evol[0]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(0.25, (fase1_vp_evol[0] + fase1_vp_evol[1])/2 + 0.5, '+2.8 ✅',
            fontsize=11, fontweight='bold', color='green')
    
    ax3.annotate('', xy=(1.5, fase2_vp_evol[1]), xytext=(1, fase2_vp_evol[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.text(1.25, (fase2_vp_evol[0] + fase2_vp_evol[1])/2 + 0.3, '-0.3 ❌',
            fontsize=11, fontweight='bold', color='red')
    
    ax3.set_ylabel('VP Médio por Dia', fontsize=11, fontweight='bold')
    ax3.set_title('Evolução do VP Dentro de Cada Fase\n'
                  'Fase 1: Melhorou | Fase 2: Não melhorou',
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Resumo visual
    ax4 = axes[1, 1]
    
    categories = ['Aprendeu\nPadrão', 'Detectou\nShift', 'Adaptou ao\nNovo Padrão']
    scores = [100, 100, 0]  # Baseado nos resultados
    colors = ['#2ecc71', '#2ecc71', '#e74c3c']
    
    bars = ax4.bar(categories, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    symbols = ['✅', '✅', '❌']
    for bar, symbol in zip(bars, symbols):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 5, symbol,
                ha='center', fontsize=24)
    
    ax4.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Resumo: Capacidades do Sistema\n'
                  '2/3 ✅ | 1/3 ❌',
                  fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 120)
    ax4.axhline(y=70, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('COMPARAÇÃO: Fase 1 (Matinal) vs Fase 2 (Noturno)',
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph2_phase_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 2: Comparação entre Fases")


def plot_shift_detection_analysis(data):
    """
    Gráfico 3: ANÁLISE DA DETECÇÃO DE SHIFT
    Mostra o momento do shift e a resposta do sistema
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    fase1 = data['raw_data']['fase1_matinal']
    fase2 = data['raw_data']['fase2_noturno']
    
    # Dados ao redor do shift (dias 25-40)
    pre_shift = fase1[-6:]  # Dias 25-30
    post_shift = fase2[:10]  # Dias 31-40
    
    dias_zoom = list(range(25, 41))
    vp_zoom = [d[1] for d in pre_shift] + [d[1] for d in post_shift]
    vn_zoom = [d[2] for d in pre_shift] + [d[2] for d in post_shift]
    epsilon_zoom = [d[4] for d in pre_shift] + [d[4] for d in post_shift]
    
    # Gráfico 1: VP e VN ao redor do shift
    ax1.bar(dias_zoom, vp_zoom, label='VP (Acertos)', color='#2ecc71', alpha=0.8, edgecolor='black')
    ax1.bar(dias_zoom, vn_zoom, bottom=vp_zoom, label='VN (Erros)', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Linha do shift
    ax1.axvline(x=30.5, color='purple', linestyle='--', linewidth=4, label='SHIFT')
    
    # Zonas
    ax1.axvspan(24.5, 30.5, alpha=0.2, color='yellow')
    ax1.axvspan(30.5, 40.5, alpha=0.2, color='blue')
    
    # Anotações detalhadas
    ax1.annotate('Dia 30:\nVP=6, VN=2\n✅ Funcionando bem', 
                xy=(30, 8), xytext=(27, 12),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))
    
    ax1.annotate('Dia 31:\nVP=2, VN=6\n🚨 SHIFT DETECTADO!', 
                xy=(31, 8), xytext=(34, 12),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
    
    ax1.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax1.set_title('Zoom no Momento do SHIFT (Dias 25-40)\n'
                  'VP despenca de 6 para 2 | VN sobe de 2 para 6',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper right')
    ax1.set_xlim(24.5, 40.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Resposta do Epsilon
    ax2.plot(dias_zoom, epsilon_zoom, color='#9b59b6', linewidth=3, marker='o', markersize=8)
    ax2.fill_between(dias_zoom, epsilon_zoom, alpha=0.3, color='#9b59b6')
    
    # Linha do shift
    ax2.axvline(x=30.5, color='purple', linestyle='--', linewidth=4)
    
    # Anotações
    ax2.annotate('Epsilon = 0.127\nMínimo (modelo confiante)', 
                xy=(30, 0.127), xytext=(26, 0.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue'))
    
    ax2.annotate('Epsilon = 0.500\nBOOST! (re-exploração)', 
                xy=(31, 0.5), xytext=(35, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'),
                fontsize=10, fontweight='bold', color='purple',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple'))
    
    ax2.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Epsilon (Exploração)', fontsize=12, fontweight='bold')
    ax2.set_title('Resposta do Sistema: BOOST de Epsilon\n'
                  'Sistema detectou anomalia e aumentou exploração de 0.127 → 0.500 (+294%)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlim(24.5, 40.5)
    ax2.set_ylim(0, 0.6)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph3_shift_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 3: Análise da Detecção de Shift")


def plot_precision_evolution(data):
    """
    Gráfico 4: EVOLUÇÃO DA PRECISION ACUMULADA
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    fase1 = data['raw_data']['fase1_matinal']
    fase2 = data['raw_data']['fase2_noturno']
    all_days = fase1 + fase2
    
    # Calcula precision acumulada
    dias = []
    precision_acum = []
    vp_acum = 0
    vn_acum = 0
    
    for d in all_days:
        dias.append(d[0])
        vp_acum += d[1]
        vn_acum += d[2]
        if (vp_acum + vn_acum) > 0:
            precision_acum.append(vp_acum / (vp_acum + vn_acum) * 100)
        else:
            precision_acum.append(0)
    
    # Plot principal
    ax.plot(dias, precision_acum, color='#3498db', linewidth=3, marker='o', markersize=4)
    
    # Linha do shift
    ax.axvline(x=30.5, color='purple', linestyle='--', linewidth=3, label='SHIFT (Dia 30→31)')
    
    # Zonas
    ax.axvspan(0, 30.5, alpha=0.1, color='yellow', label='Fase 1: Matinal')
    ax.axvspan(30.5, 91, alpha=0.1, color='blue', label='Fase 2: Noturno')
    
    # Meta
    ax.axhline(y=70, color='green', linestyle='--', linewidth=2, label='Meta (70%)')
    
    # Preenchimento
    ax.fill_between(dias, precision_acum, 70, where=[p >= 70 for p in precision_acum],
                   color='green', alpha=0.2)
    ax.fill_between(dias, precision_acum, 70, where=[p < 70 for p in precision_acum],
                   color='red', alpha=0.2)
    
    # Anotações
    # Pico da fase 1
    precision_dia30 = precision_acum[29]
    ax.annotate(f'Pico: {precision_dia30:.1f}%\n✅ Acima da meta!',
               xy=(30, precision_dia30), xytext=(20, precision_dia30 + 10),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'),
               fontsize=11, fontweight='bold', color='green',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))
    
    # Final
    precision_final = precision_acum[-1]
    ax.annotate(f'Final: {precision_final:.1f}%\n❌ Abaixo da meta',
               xy=(90, precision_final), xytext=(75, precision_final - 15),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               fontsize=11, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
    
    # Queda após shift
    ax.annotate('📉 Queda após shift\nNão conseguiu adaptar',
               xy=(50, precision_acum[49]), xytext=(55, precision_acum[49] + 15),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'),
               fontsize=10, fontweight='bold', color='orange',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='orange'))
    
    ax.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision Acumulada (%)', fontsize=12, fontweight='bold')
    ax.set_title('Evolução da Precision Acumulada ao Longo de 90 Dias\n'
                 f'Fase 1: Subiu até {precision_dia30:.1f}% | '
                 f'Fase 2: Caiu para {precision_final:.1f}%',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim(0, 91)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph4_precision_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 4: Evolução da Precision")


def plot_summary_dashboard(data):
    """
    Gráfico 5: DASHBOARD RESUMO
    Visão geral de todos os resultados
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Layout: 3 linhas
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    fase1 = data['fase1_matinal']
    fase2 = data['fase2_noturno']
    conclusions = data['conclusions']
    shift_info = data['shift_detection']
    
    # 1. Métricas Fase 1 (card)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, 
                                facecolor='#fff9c4', edgecolor='#f9a825', linewidth=3))
    ax1.text(0.5, 0.85, '☀️ FASE 1: MATINAL', fontsize=14, fontweight='bold', 
             ha='center', va='top')
    ax1.text(0.5, 0.70, f'Dias 1-30 ({fase1["days"]} dias)', fontsize=11, ha='center')
    ax1.text(0.5, 0.50, f'Precision: {fase1["precision"]:.1f}%', fontsize=13, 
             fontweight='bold', ha='center', color='green')
    ax1.text(0.5, 0.35, f'VP: {fase1["total_vp"]} | VN: {fase1["total_vn"]}', 
             fontsize=11, ha='center')
    ax1.text(0.5, 0.15, '✅ SUCESSO', fontsize=16, fontweight='bold', 
             ha='center', color='green')
    
    # 2. Métricas Fase 2 (card)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True,
                                facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=3))
    ax2.text(0.5, 0.85, '🌙 FASE 2: NOTURNO', fontsize=14, fontweight='bold',
             ha='center', va='top')
    ax2.text(0.5, 0.70, f'Dias 31-90 ({fase2["days"]} dias)', fontsize=11, ha='center')
    ax2.text(0.5, 0.50, f'Precision: {fase2["precision"]:.1f}%', fontsize=13,
             fontweight='bold', ha='center', color='red')
    ax2.text(0.5, 0.35, f'VP: {fase2["total_vp"]} | VN: {fase2["total_vn"]}',
             fontsize=11, ha='center')
    ax2.text(0.5, 0.15, '❌ FALHA', fontsize=16, fontweight='bold',
             ha='center', color='red')
    
    # 3. Detecção de Shift (card)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True,
                                facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=3))
    ax3.text(0.5, 0.85, '🔄 DETECÇÃO DE SHIFT', fontsize=14, fontweight='bold',
             ha='center', va='top')
    ax3.text(0.5, 0.65, f'Detectado: {"SIM ✅" if shift_info["detected"] else "NÃO ❌"}',
             fontsize=12, ha='center', fontweight='bold',
             color='green' if shift_info["detected"] else 'red')
    ax3.text(0.5, 0.50, f'Dia da detecção: {shift_info["detection_day"]}',
             fontsize=11, ha='center')
    ax3.text(0.5, 0.35, f'Falsos positivos: {len(shift_info["false_positives"])}',
             fontsize=11, ha='center', color='orange')
    ax3.text(0.5, 0.20, f'(Dias {shift_info["false_positives"]})',
             fontsize=9, ha='center', color='gray')
    
    # 4. Gráfico de barras: VP por fase
    ax4 = fig.add_subplot(gs[1, 0])
    
    raw_fase1 = data['raw_data']['fase1_matinal']
    raw_fase2 = data['raw_data']['fase2_noturno']
    
    vp_fase1 = [d[1] for d in raw_fase1]
    vp_fase2 = [d[1] for d in raw_fase2]
    
    ax4.boxplot([vp_fase1, vp_fase2], labels=['Fase 1\n(Matinal)', 'Fase 2\n(Noturno)'],
               patch_artist=True, boxprops=dict(facecolor='#bbdefb', color='black'),
               medianprops=dict(color='red', linewidth=2))
    ax4.set_ylabel('VP por Dia', fontsize=11, fontweight='bold')
    ax4.set_title('Distribuição de VP', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Gráfico de barras: VN por fase
    ax5 = fig.add_subplot(gs[1, 1])
    
    vn_fase1 = [d[2] for d in raw_fase1]
    vn_fase2 = [d[2] for d in raw_fase2]
    
    ax5.boxplot([vn_fase1, vn_fase2], labels=['Fase 1\n(Matinal)', 'Fase 2\n(Noturno)'],
               patch_artist=True, boxprops=dict(facecolor='#ffcdd2', color='black'),
               medianprops=dict(color='darkred', linewidth=2))
    ax5.set_ylabel('VN por Dia', fontsize=11, fontweight='bold')
    ax5.set_title('Distribuição de VN (Erros)', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Conclusões
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    
    ax6.text(0.5, 0.95, '📋 CONCLUSÕES', fontsize=14, fontweight='bold', ha='center')
    
    y = 0.80
    conclusions_text = [
        (conclusions['learning_success'], 'Aprendeu padrão inicial'),
        (conclusions['shift_detection_success'], 'Detectou shift de comportamento'),
        (conclusions['adaptation_success'], 'Adaptou ao novo padrão')
    ]
    
    for success, text in conclusions_text:
        symbol = '✅' if success else '❌'
        color = 'green' if success else 'red'
        ax6.text(0.1, y, f'{symbol} {text}', fontsize=12, color=color, fontweight='bold')
        y -= 0.15
    
    ax6.text(0.1, y - 0.1, f'⚠️ {conclusions["main_issue"]}', fontsize=10,
            color='orange', style='italic', wrap=True)
    
    # 7. Timeline (linha do tempo)
    ax7 = fig.add_subplot(gs[2, :])
    
    ax7.set_xlim(0, 100)
    ax7.set_ylim(0, 10)
    ax7.axis('off')
    
    # Linha base
    ax7.plot([5, 95], [5, 5], color='gray', linewidth=3)
    
    # Marcadores
    markers = [
        (5, 'Início\nDia 1', '#f39c12', '☀️'),
        (35, 'Aprendeu\nDia ~15', '#27ae60', '✅'),
        (50, 'SHIFT\nDia 30→31', '#9b59b6', '🔄'),
        (65, 'Tentando\nadaptar', '#e74c3c', '⚠️'),
        (95, 'Final\nDia 90', '#3498db', '🏁')
    ]
    
    for x, label, color, emoji in markers:
        ax7.plot(x, 5, 'o', markersize=20, color=color)
        ax7.text(x, 7.5, emoji, fontsize=20, ha='center')
        ax7.text(x, 2.5, label, fontsize=10, ha='center', fontweight='bold')
    
    # Setas
    ax7.annotate('', xy=(35, 5), xytext=(5, 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax7.annotate('', xy=(50, 5), xytext=(35, 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax7.annotate('', xy=(95, 5), xytext=(50, 5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax7.set_title('Timeline da Simulação de Shift de Comportamento',
                 fontsize=13, fontweight='bold', y=1.1)
    
    plt.suptitle('DASHBOARD: Simulação de Shift de Comportamento (Epoch 1)\n'
                 'Matinal → Noturno | 90 Dias | Shift no Dia 30',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'graph5_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 5: Dashboard Resumo")


def main():
    """Gera todos os gráficos."""
    print("=" * 80)
    print("📊 GERADOR DE GRÁFICOS - SHIFT DE COMPORTAMENTO")
    print("=" * 80)
    print()
    
    # Carrega dados
    print("📂 Carregando dados...")
    data = load_data()
    
    if not data:
        return
    
    print(f"✅ Dados carregados: {data['config']['total_days']} dias")
    print(f"   Cenário: {data['config']['scenario']}")
    print(f"\n📁 Salvando gráficos em: {output_dir.absolute()}\n")
    
    try:
        plot_vp_vn_evolution(data)
        plot_phase_comparison(data)
        plot_shift_detection_analysis(data)
        plot_precision_evolution(data)
        plot_summary_dashboard(data)
        
        print()
        print("=" * 80)
        print("✅ TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
        print("=" * 80)
        print()
        print(f"📂 Localização: {output_dir.absolute()}")
        print()
        print("Gráficos gerados:")
        print("  1. graph1_vp_vn_evolution.png      - Evolução VP/VN ao longo dos 90 dias")
        print("  2. graph2_phase_comparison.png     - Comparação entre Fases")
        print("  3. graph3_shift_detection.png      - Análise da Detecção de Shift")
        print("  4. graph4_precision_evolution.png  - Evolução da Precision")
        print("  5. graph5_summary_dashboard.png    - Dashboard Resumo")
        print()
        
    except Exception as e:
        print(f"\n❌ Erro ao gerar gráficos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
