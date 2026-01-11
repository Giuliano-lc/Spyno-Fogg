"""
Gera gráficos de análise do SHIFT DE COMPORTAMENTO usando dados reais.
VERSÃO: Apenas até o dia 60 (30 dias matinal + 30 dias noturno)

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
output_dir = Path(__file__).parent.parent / "data" / "simulation" / "graficos_shift_behavior_60dias"
output_dir.mkdir(parents=True, exist_ok=True)

# Limite de dias
MAX_DAY = 60


def load_data():
    """Carrega dados da análise e filtra até dia 60."""
    if not data_file.exists():
        print(f"Arquivo nao encontrado: {data_file}")
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filtra fase2 para apenas 30 dias (31-60)
    fase2_filtered = [d for d in data['raw_data']['fase2_noturno'] if d[0] <= MAX_DAY]
    data['raw_data']['fase2_noturno'] = fase2_filtered
    
    # Recalcula métricas da fase 2
    vp_total = sum(d[1] for d in fase2_filtered)
    vn_total = sum(d[2] for d in fase2_filtered)
    fp_total = sum(d[3] for d in fase2_filtered)
    
    data['fase2_noturno']['days'] = len(fase2_filtered)
    data['fase2_noturno']['total_vp'] = vp_total
    data['fase2_noturno']['total_vn'] = vn_total
    data['fase2_noturno']['total_fp'] = fp_total
    data['fase2_noturno']['precision'] = (vp_total / (vp_total + vn_total) * 100) if (vp_total + vn_total) > 0 else 0
    data['fase2_noturno']['recall'] = (vp_total / (vp_total + fp_total) * 100) if (vp_total + fp_total) > 0 else 0
    
    prec = data['fase2_noturno']['precision']
    rec = data['fase2_noturno']['recall']
    data['fase2_noturno']['f1_score'] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
    
    # Recalcula evolução
    first_10 = fase2_filtered[:10]
    last_10 = fase2_filtered[-10:]
    
    data['fase2_noturno']['evolution'] = {
        'first_10_days': {
            'vp': sum(d[1] for d in first_10),
            'vn': sum(d[2] for d in first_10),
            'avg_vp': sum(d[1] for d in first_10) / 10,
            'avg_vn': sum(d[2] for d in first_10) / 10
        },
        'last_10_days': {
            'vp': sum(d[1] for d in last_10),
            'vn': sum(d[2] for d in last_10),
            'avg_vp': sum(d[1] for d in last_10) / 10,
            'avg_vn': sum(d[2] for d in last_10) / 10
        }
    }
    
    data['fase2_noturno']['epsilon_end'] = fase2_filtered[-1][4]
    
    # Atualiza config
    data['config']['total_days'] = MAX_DAY
    data['config']['scenario'] = f"Matinal (dias 1-30) -> Noturno (dias 31-{MAX_DAY})"
    
    # Atualiza shift detection (remove falsos positivos após dia 60)
    data['shift_detection']['false_positives'] = [d for d in data['shift_detection']['false_positives'] if d <= MAX_DAY]
    
    return data


def plot_vp_vn_evolution(data):
    """
    Gráfico 1: EVOLUÇÃO VP E VN AO LONGO DOS 60 DIAS
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    fase1 = data['raw_data']['fase1_matinal']
    fase2 = data['raw_data']['fase2_noturno']
    
    all_days = fase1 + fase2
    dias = [d[0] for d in all_days]
    vp = [d[1] for d in all_days]
    vn = [d[2] for d in all_days]
    fp = [d[3] for d in all_days]
    epsilon = [d[4] for d in all_days]
    
    # Gráfico 1: VP e VN empilhados
    ax1.bar(dias, vp, label='VP (Acertos)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(dias, vn, bottom=vp, label='VN (Erros/Spam)', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Linha vertical no shift
    ax1.axvline(x=30.5, color='purple', linestyle='--', linewidth=3, label='SHIFT (Dia 30->31)')
    
    # Zonas de fase
    ax1.axvspan(0, 30.5, alpha=0.1, color='yellow', label='Fase 1: Matinal')
    ax1.axvspan(30.5, MAX_DAY + 1, alpha=0.1, color='blue', label='Fase 2: Noturno')
    
    # Média móvel de VP
    vp_smooth = np.convolve(vp, np.ones(5)/5, mode='same')
    ax1.plot(dias, vp_smooth, color='darkgreen', linewidth=2.5, linestyle='--', label='VP (media movel)')
    
    ax1.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax1.set_title(f'Evolucao de VP (Acertos) e VN (Erros) ao Longo de {MAX_DAY} Dias\n'
                  'Fase 1: Matinal (Dias 1-30) | Fase 2: Noturno (Dias 31-60)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, MAX_DAY + 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Anotações
    ax1.annotate('SHIFT\nDETECTADO!', xy=(31, 8), fontsize=11, fontweight='bold',
                color='purple', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple', alpha=0.9))
    
    vp_medio_f1 = sum(d[1] for d in fase1) / len(fase1)
    vp_medio_f2 = sum(d[1] for d in fase2) / len(fase2)
    ax1.annotate(f'VP medio: {vp_medio_f1:.1f}/dia', xy=(15, max(vp[:30])+2), fontsize=10,
                ha='center', color='green', fontweight='bold')
    ax1.annotate(f'VP medio: {vp_medio_f2:.1f}/dia', xy=(45, max(vp[30:])+2), fontsize=10,
                ha='center', color='darkgreen', fontweight='bold')
    
    # Gráfico 2: Epsilon e Taxa de Acerto
    ax2_twin = ax2.twinx()
    
    # Taxa de acerto por dia
    taxa_acerto = [(v / (v + n) * 100) if (v + n) > 0 else 0 for v, n in zip(vp, vn)]
    
    ax2.plot(dias, taxa_acerto, color='#3498db', linewidth=2, marker='o', markersize=3,
             label='Taxa de Acerto (%)', alpha=0.8)
    ax2.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Meta (70%)')
    
    # Epsilon
    ax2_twin.plot(dias, epsilon, color='#9b59b6', linewidth=2.5, label='Epsilon (Exploracao)')
    ax2_twin.fill_between(dias, epsilon, alpha=0.2, color='#9b59b6')
    
    # Shift
    ax2.axvline(x=30.5, color='purple', linestyle='--', linewidth=3)
    
    # Marca o boost de epsilon no dia 31
    ax2_twin.annotate('BOOST', xy=(31, 0.5), fontsize=9, color='purple',
                     ha='center', fontweight='bold')
    
    ax2.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Taxa de Acerto (%)', fontsize=12, fontweight='bold', color='#3498db')
    ax2_twin.set_ylabel('Epsilon', fontsize=12, fontweight='bold', color='#9b59b6')
    
    ax2.set_title('Taxa de Acerto e Exploracao (Epsilon) ao Longo do Tempo\n'
                  'Epsilon aumenta quando shift e detectado para re-explorar',
                  fontsize=13, fontweight='bold', pad=15)
    
    ax2.set_xlim(0, MAX_DAY + 1)
    ax2.set_ylim(0, 100)
    ax2_twin.set_ylim(0, 0.6)
    
    # Legendas combinadas
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph1_vp_vn_evolution_60d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Grafico 1: Evolucao VP/VN ao longo dos 60 dias")


def plot_phase_comparison(data):
    """
    Gráfico 2: COMPARAÇÃO ENTRE FASES
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
    ax1.set_title('Comparacao de Metricas entre Fases\n'
                  'Matinal: SUCESSO | Noturno: FALHA',
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
        ax2.text(i - width/2, v1 + 2, str(v1), ha='center', fontsize=10, fontweight='bold')
        ax2.text(i + width/2, v2 + 2, str(v2), ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Quantidade', fontsize=11, fontweight='bold')
    ax2.set_title(f'Totais de VP, VN e FP por Fase\n'
                  f'VN aumentou de {fase1["total_vn"]} para {fase2["total_vn"]}',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcomes)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Evolução temporal dentro de cada fase
    ax3 = axes[1, 0]
    
    labels = ['Primeiros\n10 dias', 'Ultimos\n10 dias']
    fase1_vp_evol = [fase1['evolution']['first_10_days']['avg_vp'], fase1['evolution']['last_10_days']['avg_vp']]
    fase2_vp_evol = [fase2['evolution']['first_10_days']['avg_vp'], fase2['evolution']['last_10_days']['avg_vp']]
    
    x = np.arange(len(labels))
    
    ax3.bar(x - width/2, fase1_vp_evol, width, label='Fase 1 (Matinal)', color='#f39c12', alpha=0.8, edgecolor='black')
    ax3.bar(x + width/2, fase2_vp_evol, width, label='Fase 2 (Noturno)', color='#3498db', alpha=0.8, edgecolor='black')
    
    # Setas de melhoria/piora
    diff1 = fase1_vp_evol[1] - fase1_vp_evol[0]
    diff2 = fase2_vp_evol[1] - fase2_vp_evol[0]
    
    ax3.annotate('', xy=(0.5, fase1_vp_evol[1]), xytext=(0, fase1_vp_evol[0]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(0.25, (fase1_vp_evol[0] + fase1_vp_evol[1])/2 + 0.5, f'+{diff1:.1f} OK',
            fontsize=11, fontweight='bold', color='green')
    
    color2 = 'green' if diff2 > 0 else 'red'
    symbol2 = 'OK' if diff2 > 0 else 'X'
    ax3.annotate('', xy=(1.5, fase2_vp_evol[1]), xytext=(1, fase2_vp_evol[0]),
                arrowprops=dict(arrowstyle='->', color=color2, lw=2))
    ax3.text(1.25, (fase2_vp_evol[0] + fase2_vp_evol[1])/2 + 0.3, f'{diff2:+.1f} {symbol2}',
            fontsize=11, fontweight='bold', color=color2)
    
    ax3.set_ylabel('VP Medio por Dia', fontsize=11, fontweight='bold')
    ax3.set_title('Evolucao do VP Dentro de Cada Fase\n'
                  'Fase 1: Melhorou | Fase 2: ?',
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Resumo visual
    ax4 = axes[1, 1]
    
    categories = ['Aprendeu\nPadrao', 'Detectou\nShift', 'Adaptou ao\nNovo Padrao']
    
    # Verifica adaptação baseado na evolução da fase 2
    adapted = fase2_vp_evol[1] > fase2_vp_evol[0] and fase2['precision'] > 50
    scores = [100, 100, 50 if adapted else 0]
    colors = ['#2ecc71', '#2ecc71', '#f39c12' if adapted else '#e74c3c']
    
    bars = ax4.bar(categories, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    symbols = ['OK', 'OK', '?' if adapted else 'X']
    for bar, symbol in zip(bars, symbols):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 5, symbol,
                ha='center', fontsize=18, fontweight='bold')
    
    ax4.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Resumo: Capacidades do Sistema\n'
                  '(Ate o dia 60)',
                  fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 120)
    ax4.axhline(y=70, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'COMPARACAO: Fase 1 (Matinal) vs Fase 2 (Noturno) - Ate dia {MAX_DAY}',
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph2_phase_comparison_60d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Grafico 2: Comparacao entre Fases")


def plot_shift_detection_analysis(data):
    """
    Gráfico 3: ANÁLISE DA DETECÇÃO DE SHIFT
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
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
    vp_dia30 = pre_shift[-1][1]
    vn_dia30 = pre_shift[-1][2]
    vp_dia31 = post_shift[0][1]
    vn_dia31 = post_shift[0][2]
    
    ax1.annotate(f'Dia 30:\nVP={vp_dia30}, VN={vn_dia30}\nFuncionando bem', 
                xy=(30, 8), xytext=(27, 12),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))
    
    ax1.annotate(f'Dia 31:\nVP={vp_dia31}, VN={vn_dia31}\nSHIFT DETECTADO!', 
                xy=(31, 8), xytext=(34, 12),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
    
    ax1.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax1.set_title('Zoom no Momento do SHIFT (Dias 25-40)\n'
                  f'VP despenca de {vp_dia30} para {vp_dia31} | VN sobe de {vn_dia30} para {vn_dia31}',
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
    eps_dia30 = pre_shift[-1][4]
    eps_dia31 = post_shift[0][4]
    
    ax2.annotate(f'Epsilon = {eps_dia30:.3f}\nMinimo (modelo confiante)', 
                xy=(30, eps_dia30), xytext=(26, 0.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue'))
    
    ax2.annotate(f'Epsilon = {eps_dia31:.3f}\nBOOST! (re-exploracao)', 
                xy=(31, eps_dia31), xytext=(35, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'),
                fontsize=10, fontweight='bold', color='purple',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple'))
    
    boost_pct = ((eps_dia31 - eps_dia30) / eps_dia30) * 100
    
    ax2.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Epsilon (Exploracao)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Resposta do Sistema: BOOST de Epsilon\n'
                  f'Sistema detectou anomalia e aumentou exploracao de {eps_dia30:.3f} -> {eps_dia31:.3f} (+{boost_pct:.0f}%)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlim(24.5, 40.5)
    ax2.set_ylim(0, 0.6)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph3_shift_detection_60d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Grafico 3: Analise da Deteccao de Shift")


def plot_precision_evolution(data):
    """
    Gráfico 4: EVOLUÇÃO DA PRECISION ACUMULADA
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
    ax.axvline(x=30.5, color='purple', linestyle='--', linewidth=3, label='SHIFT (Dia 30->31)')
    
    # Zonas
    ax.axvspan(0, 30.5, alpha=0.1, color='yellow', label='Fase 1: Matinal')
    ax.axvspan(30.5, MAX_DAY + 1, alpha=0.1, color='blue', label='Fase 2: Noturno')
    
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
    ax.annotate(f'Pico: {precision_dia30:.1f}%\nAcima da meta!',
               xy=(30, precision_dia30), xytext=(20, precision_dia30 + 10),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'),
               fontsize=11, fontweight='bold', color='green',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))
    
    # Final
    precision_final = precision_acum[-1]
    status = 'Acima' if precision_final >= 70 else 'Abaixo'
    color_final = 'green' if precision_final >= 70 else 'red'
    ax.annotate(f'Dia {MAX_DAY}: {precision_final:.1f}%\n{status} da meta',
               xy=(MAX_DAY, precision_final), xytext=(MAX_DAY - 15, precision_final - 15),
               arrowprops=dict(arrowstyle='->', lw=2, color=color_final),
               fontsize=11, fontweight='bold', color=color_final,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=color_final))
    
    # Queda após shift
    ax.annotate('Queda apos shift',
               xy=(45, precision_acum[44]), xytext=(50, precision_acum[44] + 12),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'),
               fontsize=10, fontweight='bold', color='orange',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='orange'))
    
    ax.set_xlabel('Dia', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision Acumulada (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Evolucao da Precision Acumulada ao Longo de {MAX_DAY} Dias\n'
                 f'Fase 1: Subiu ate {precision_dia30:.1f}% | '
                 f'Dia {MAX_DAY}: {precision_final:.1f}%',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim(0, MAX_DAY + 1)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph4_precision_evolution_60d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Grafico 4: Evolucao da Precision")


def plot_summary_dashboard(data):
    """
    Gráfico 5: DASHBOARD RESUMO
    """
    fig = plt.figure(figsize=(16, 12))
    
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
    ax1.text(0.5, 0.85, 'FASE 1: MATINAL', fontsize=14, fontweight='bold', 
             ha='center', va='top')
    ax1.text(0.5, 0.70, f'Dias 1-30 ({fase1["days"]} dias)', fontsize=11, ha='center')
    ax1.text(0.5, 0.50, f'Precision: {fase1["precision"]:.1f}%', fontsize=13, 
             fontweight='bold', ha='center', color='green')
    ax1.text(0.5, 0.35, f'VP: {fase1["total_vp"]} | VN: {fase1["total_vn"]}', 
             fontsize=11, ha='center')
    ax1.text(0.5, 0.15, 'SUCESSO', fontsize=16, fontweight='bold', 
             ha='center', color='green')
    
    # 2. Métricas Fase 2 (card)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    color_f2 = 'green' if fase2['precision'] >= 50 else 'red'
    status_f2 = 'EM PROGRESSO' if fase2['precision'] >= 40 else 'DIFICULDADE'
    
    ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True,
                                facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=3))
    ax2.text(0.5, 0.85, 'FASE 2: NOTURNO', fontsize=14, fontweight='bold',
             ha='center', va='top')
    ax2.text(0.5, 0.70, f'Dias 31-{MAX_DAY} ({fase2["days"]} dias)', fontsize=11, ha='center')
    ax2.text(0.5, 0.50, f'Precision: {fase2["precision"]:.1f}%', fontsize=13,
             fontweight='bold', ha='center', color=color_f2)
    ax2.text(0.5, 0.35, f'VP: {fase2["total_vp"]} | VN: {fase2["total_vn"]}',
             fontsize=11, ha='center')
    ax2.text(0.5, 0.15, status_f2, fontsize=16, fontweight='bold',
             ha='center', color=color_f2)
    
    # 3. Detecção de Shift (card)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True,
                                facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=3))
    ax3.text(0.5, 0.85, 'DETECCAO DE SHIFT', fontsize=14, fontweight='bold',
             ha='center', va='top')
    det_text = 'SIM' if shift_info["detected"] else 'NAO'
    det_color = 'green' if shift_info["detected"] else 'red'
    ax3.text(0.5, 0.65, f'Detectado: {det_text}',
             fontsize=12, ha='center', fontweight='bold', color=det_color)
    ax3.text(0.5, 0.50, f'Dia da deteccao: {shift_info["detection_day"]}',
             fontsize=11, ha='center')
    fp_count = len(shift_info["false_positives"])
    ax3.text(0.5, 0.35, f'Falsos positivos: {fp_count}',
             fontsize=11, ha='center', color='orange' if fp_count > 0 else 'green')
    if shift_info["false_positives"]:
        ax3.text(0.5, 0.20, f'(Dias {shift_info["false_positives"]})',
                 fontsize=9, ha='center', color='gray')
    
    # 4. Gráfico de barras: VP por fase
    ax4 = fig.add_subplot(gs[1, 0])
    
    raw_fase1 = data['raw_data']['fase1_matinal']
    raw_fase2 = data['raw_data']['fase2_noturno']
    
    vp_fase1 = [d[1] for d in raw_fase1]
    vp_fase2 = [d[1] for d in raw_fase2]
    
    ax4.boxplot([vp_fase1, vp_fase2], tick_labels=['Fase 1\n(Matinal)', 'Fase 2\n(Noturno)'],
               patch_artist=True, boxprops=dict(facecolor='#bbdefb', color='black'),
               medianprops=dict(color='red', linewidth=2))
    ax4.set_ylabel('VP por Dia', fontsize=11, fontweight='bold')
    ax4.set_title('Distribuicao de VP', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Gráfico de barras: VN por fase
    ax5 = fig.add_subplot(gs[1, 1])
    
    vn_fase1 = [d[2] for d in raw_fase1]
    vn_fase2 = [d[2] for d in raw_fase2]
    
    ax5.boxplot([vn_fase1, vn_fase2], tick_labels=['Fase 1\n(Matinal)', 'Fase 2\n(Noturno)'],
               patch_artist=True, boxprops=dict(facecolor='#ffcdd2', color='black'),
               medianprops=dict(color='darkred', linewidth=2))
    ax5.set_ylabel('VN por Dia', fontsize=11, fontweight='bold')
    ax5.set_title('Distribuicao de VN (Erros)', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Conclusões
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    
    ax6.text(0.5, 0.95, 'CONCLUSOES (ate dia 60)', fontsize=14, fontweight='bold', ha='center')
    
    y = 0.75
    conclusions_text = [
        (conclusions['learning_success'], 'Aprendeu padrao inicial'),
        (conclusions['shift_detection_success'], 'Detectou shift de comportamento'),
        (fase2['precision'] > 40, 'Tentando adaptar ao novo padrao')
    ]
    
    for success, text in conclusions_text:
        symbol = 'OK' if success else 'X'
        color = 'green' if success else 'red'
        ax6.text(0.1, y, f'[{symbol}] {text}', fontsize=11, color=color, fontweight='bold')
        y -= 0.18
    
    ax6.text(0.1, y, 'Modelo ainda em fase de adaptacao', fontsize=10,
            color='orange', style='italic')
    
    # 7. Timeline
    ax7 = fig.add_subplot(gs[2, :])
    
    ax7.set_xlim(0, 100)
    ax7.set_ylim(0, 10)
    ax7.axis('off')
    
    # Linha base
    ax7.plot([5, 95], [5, 5], color='gray', linewidth=3)
    
    # Marcadores (ajustados para 60 dias)
    markers = [
        (5, 'Inicio\nDia 1', '#f39c12', 'START'),
        (30, 'Aprendeu\nDia ~15', '#27ae60', 'OK'),
        (50, 'SHIFT\nDia 30->31', '#9b59b6', 'SHIFT'),
        (75, 'Adaptando\nDia 45', '#e74c3c', '...'),
        (95, f'Atual\nDia {MAX_DAY}', '#3498db', 'NOW')
    ]
    
    for x, label, color, emoji in markers:
        ax7.plot(x, 5, 'o', markersize=20, color=color)
        ax7.text(x, 7.5, emoji, fontsize=12, ha='center', fontweight='bold')
        ax7.text(x, 2.5, label, fontsize=10, ha='center', fontweight='bold')
    
    # Setas
    ax7.annotate('', xy=(30, 5), xytext=(5, 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax7.annotate('', xy=(50, 5), xytext=(30, 5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax7.annotate('', xy=(95, 5), xytext=(50, 5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    ax7.set_title('Timeline da Simulacao de Shift de Comportamento',
                 fontsize=13, fontweight='bold', y=1.1)
    
    plt.suptitle(f'DASHBOARD: Simulacao de Shift de Comportamento (Ate Dia {MAX_DAY})\n'
                 'Matinal -> Noturno | 60 Dias | Shift no Dia 30',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'graph5_summary_dashboard_60d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Grafico 5: Dashboard Resumo")


def main():
    """Gera todos os gráficos."""
    print("=" * 80)
    print(f"GERADOR DE GRAFICOS - SHIFT DE COMPORTAMENTO (ATE DIA {MAX_DAY})")
    print("=" * 80)
    print()
    
    # Carrega dados
    print("Carregando dados...")
    data = load_data()
    
    if not data:
        return
    
    print(f"Dados carregados e filtrados: {data['config']['total_days']} dias")
    print(f"   Cenario: {data['config']['scenario']}")
    print(f"\nSalvando graficos em: {output_dir.absolute()}\n")
    
    try:
        plot_vp_vn_evolution(data)
        plot_phase_comparison(data)
        plot_shift_detection_analysis(data)
        plot_precision_evolution(data)
        plot_summary_dashboard(data)
        
        print()
        print("=" * 80)
        print("TODOS OS GRAFICOS GERADOS COM SUCESSO!")
        print("=" * 80)
        print()
        print(f"Localizacao: {output_dir.absolute()}")
        print()
        print("Graficos gerados (ate dia 60):")
        print("  1. graph1_vp_vn_evolution_60d.png      - Evolucao VP/VN")
        print("  2. graph2_phase_comparison_60d.png     - Comparacao entre Fases")
        print("  3. graph3_shift_detection_60d.png      - Analise da Deteccao de Shift")
        print("  4. graph4_precision_evolution_60d.png  - Evolucao da Precision")
        print("  5. graph5_summary_dashboard_60d.png    - Dashboard Resumo")
        print()
        
    except Exception as e:
        print(f"\nErro ao gerar graficos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
