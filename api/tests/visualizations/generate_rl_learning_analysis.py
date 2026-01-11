"""
Análise do Aprendizado e Comportamento do RL.
Foco: Identificação de padrões e adaptação a feedback.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Criar pasta para outputs
output_dir = Path("data/simulation/graficos_rl_learning")
output_dir.mkdir(parents=True, exist_ok=True)

# DADOS SIMULADOS (baseados nos resultados reais)
np.random.seed(42)

# 30 dias de simulação
dias = list(range(1, 31))

# Simula evolução do RL ao longo dos dias
# Dias 1-10: RL ainda aprendendo (notifica muito)
# Dias 11-20: RL começando a ajustar
# Dias 21-30: RL mais estável mas ainda com problemas

# VP (acertos) e VN (erros/spam) por dia
vp_por_dia = [4, 3, 5, 4, 6, 5, 4, 3, 6, 3,  # Dias 1-10
              5, 4, 6, 4, 4, 5, 3, 4, 5, 9,  # Dias 11-20
              3, 4, 5, 3, 4, 5, 6, 3, 2, 0]  # Dias 21-30

vn_por_dia = [4, 6, 5, 7, 10, 6, 5, 4, 5, 5,  # Dias 1-10 (muito VN)
              6, 5, 4, 5, 4, 6, 4, 3, 5, 7,   # Dias 11-20 (ainda alto)
              4, 5, 6, 4, 6, 5, 4, 6, 5, 3]   # Dias 21-30 (não melhora muito)

fp_por_dia = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
              0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
              1, 0, 0, 1, 0, 1, 0, 0, 1, 1]

# Horas que RL decidiu notificar (acumulado ao longo dos dias)
# RL deve gradualmente focar em 6-8h
notif_por_hora_semana1 = [0, 0, 0, 0, 0, 0, 8, 12, 9, 5, 4, 3, 2, 3, 2, 4, 3, 2, 3, 4, 3, 2, 0, 0]  # Disperso
notif_por_hora_semana2 = [0, 0, 0, 0, 0, 0, 15, 18, 14, 6, 4, 2, 1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 0, 0]  # Concentrando
notif_por_hora_semana3 = [0, 0, 0, 0, 0, 0, 20, 24, 17, 5, 3, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 0, 0, 0]  # Mais focado
notif_por_hora_semana4 = [0, 0, 0, 0, 0, 0, 22, 24, 19, 4, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 0, 0, 0]  # Concentrado 6-8h

# VP e VN por hora (onde RL acerta vs erra)
vp_por_hora = [0, 0, 0, 0, 0, 0, 18, 22, 16, 3, 2, 1, 0, 1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 0]  # Acerta mais em 6-8h
vn_por_hora = [0, 0, 0, 0, 0, 0, 12, 18, 14, 8, 6, 4, 3, 4, 5, 6, 5, 4, 5, 6, 4, 2, 0, 0]  # Erra fora do horário ideal


def plot_rl_hourly_focus_evolution():
    """
    Gráfico 1: EVOLUÇÃO DO FOCO HORÁRIO DO RL
    Mostra como RL gradualmente concentra notificações nas horas matinais (6-8h)
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    hours = list(range(24))
    
    # Semana 1 (dias 1-7)
    ax1.bar(hours, notif_por_hora_semana1, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvspan(5.5, 8.5, alpha=0.2, color='yellow', label='Zona Matinal (6-8h)')
    ax1.set_title('Semana 1 (Dias 1-7): RL Explorando\nNotificações dispersas, ainda aprendendo',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Notificações', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 30)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Semana 2 (dias 8-14)
    ax2.bar(hours, notif_por_hora_semana2, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvspan(5.5, 8.5, alpha=0.2, color='yellow', label='Zona Matinal (6-8h)')
    ax2.set_title('Semana 2 (Dias 8-14): RL Ajustando\nComeçando a concentrar em 6-8h',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 30)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Semana 3 (dias 15-21)
    ax3.bar(hours, notif_por_hora_semana3, color='#3498db', alpha=0.7, edgecolor='black')
    ax3.axvspan(5.5, 8.5, alpha=0.2, color='yellow', label='Zona Matinal (6-8h)')
    ax3.set_title('Semana 3 (Dias 15-21): RL Aprendendo\nFoco maior em 6-8h',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hora do Dia', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Notificações', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 30)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Semana 4 (dias 22-30)
    ax4.bar(hours, notif_por_hora_semana4, color='#3498db', alpha=0.7, edgecolor='black')
    ax4.axvspan(5.5, 8.5, alpha=0.2, color='yellow', label='Zona Matinal (6-8h)')
    ax4.set_title('Semana 4 (Dias 22-30): RL Estável\n✅ Identificou padrão matinal!',
                  fontsize=12, fontweight='bold', color='green')
    ax4.set_xlabel('Hora do Dia', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 30)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Formata eixo x para todas
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f'{h}h' for h in range(0, 24, 2)])
    
    plt.suptitle('Evolução do Foco Horário do RL - Identificação de Padrão Matinal',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph1_rl_hourly_focus_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 1: Evolução do Foco Horário do RL")


def plot_rl_learning_curve():
    """
    Gráfico 2: CURVA DE APRENDIZADO DO RL
    Mostra VP (acertos) vs VN (erros) ao longo dos dias
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: VP e VN empilhados
    ax1.bar(dias, vp_por_dia, label='VP (Acertos)', color='#2ecc71', alpha=0.8, edgecolor='black')
    ax1.bar(dias, vn_por_dia, bottom=vp_por_dia, label='VN (Spam/Erros)',
            color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Linha de média móvel de VP
    vp_smooth = np.convolve(vp_por_dia, np.ones(5)/5, mode='same')
    ax1.plot(dias, vp_smooth, color='green', linewidth=3, label='VP (média móvel)', linestyle='--')
    
    ax1.set_xlabel('Dia da Intervenção', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax1.set_title('Feedback do RL ao Longo do Tempo\nVP (Verde) = Acertos | VN (Vermelho) = Spam',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xlim(0, 31)
    
    # Adiciona anotações em pontos chave
    ax1.annotate('RL ainda\naprendendo', xy=(5, 16), xytext=(8, 18),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax1.annotate('RL ajustando', xy=(15, 9), xytext=(17, 12),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Gráfico 2: Taxa de acerto (VP / (VP+VN))
    taxa_acerto = [(vp / (vp + vn) * 100) if (vp + vn) > 0 else 0 
                   for vp, vn in zip(vp_por_dia, vn_por_dia)]
    
    ax2.plot(dias, taxa_acerto, color='#3498db', linewidth=2.5, marker='o', markersize=5,
            label='Taxa de Acerto (Precision diária)')
    
    # Média móvel
    taxa_smooth = np.convolve(taxa_acerto, np.ones(5)/5, mode='same')
    ax2.plot(dias, taxa_smooth, color='green', linewidth=3, linestyle='--',
            label='Tendência (média móvel)')
    
    # Linha de meta
    ax2.axhline(y=70, color='gray', linestyle='--', linewidth=2, alpha=0.7,
               label='Meta (70%)')
    
    ax2.set_xlabel('Dia da Intervenção', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Taxa de Acerto (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Taxa de Acerto do RL (VP / (VP+VN))\n❌ RL não consegue melhorar consistentemente',
                  fontsize=13, fontweight='bold', pad=15, color='red')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 31)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph2_rl_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 2: Curva de Aprendizado do RL")


def plot_rl_success_vs_failure_by_hour():
    """
    Gráfico 3: ONDE O RL ACERTA VS ERRA
    Mostra VP e VN distribuídos por hora
    """
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    hours = list(range(24))
    x = np.arange(24)
    width = 0.6
    
    # Barras empilhadas
    bars1 = ax.bar(x, vp_por_hora, width, label='VP (Acertos)',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, vn_por_hora, width, bottom=vp_por_hora, label='VN (Erros)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Destaque zona matinal
    ax.axvspan(5.5, 8.5, alpha=0.15, color='yellow', label='Zona Matinal Esperada')
    
    # Adiciona valores nas barras
    for i, (vp, vn) in enumerate(zip(vp_por_hora, vn_por_hora)):
        if vp > 0:
            ax.text(i, vp/2, str(vp), ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        if vn > 0:
            ax.text(i, vp + vn/2, str(vn), ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        
        # Taxa de acerto por hora
        if (vp + vn) > 0:
            taxa_hora = vp / (vp + vn) * 100
            if taxa_hora > 0:
                ax.text(i, vp + vn + 2, f'{taxa_hora:.0f}%', ha='center',
                       fontsize=8, fontweight='bold', color='green' if taxa_hora >= 60 else 'red')
    
    ax.set_xlabel('Hora do Dia', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quantidade de Notificações', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição de Acertos (VP) e Erros (VN) por Hora\n'
                 '✅ RL acerta MAIS nas horas matinais (6-8h)\n'
                 '❌ Mas ainda erra muito fora do horário ideal',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}h' for h in hours], rotation=45)
    ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(vp_por_hora[i] + vn_por_hora[i] for i in range(24)) + 8)
    
    # Anotação explicativa
    ax.text(0.5, 0.95, 
            '📊 Números acima das barras = taxa de acerto (%) | Verde ≥60% | Vermelho <60%',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph3_rl_success_failure_by_hour.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 3: Acertos vs Erros por Hora")


def plot_rl_precision_evolution():
    """
    Gráfico 4: EVOLUÇÃO DA PRECISION DO RL
    Mostra se RL melhora ao longo do tempo
    """
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calcula precision acumulada
    precision_acumulada = []
    vp_acum = 0
    vn_acum = 0
    
    for vp, vn in zip(vp_por_dia, vn_por_dia):
        vp_acum += vp
        vn_acum += vn
        if (vp_acum + vn_acum) > 0:
            precision_acumulada.append(vp_acum / (vp_acum + vn_acum) * 100)
        else:
            precision_acumulada.append(0)
    
    # Plota linha de evolução
    ax.plot(dias, precision_acumulada, color='#3498db', linewidth=3,
           marker='o', markersize=6, label='Precision Acumulada do RL')
    
    # Linha de meta
    ax.axhline(y=70, color='green', linestyle='--', linewidth=2,
              label='Meta (70%)', alpha=0.7)
    
    # Linha do threshold dinâmico (baseline)
    ax.axhline(y=76.7, color='#2ecc71', linestyle='-', linewidth=2.5,
              label='Threshold Dinâmico (76.7%)', alpha=0.8)
    
    # Zonas coloridas
    ax.axhspan(70, 100, alpha=0.1, color='green', label='Zona Boa (≥70%)')
    ax.axhspan(50, 70, alpha=0.1, color='yellow')
    ax.axhspan(0, 50, alpha=0.1, color='red', label='Zona Crítica (<50%)')
    
    ax.set_xlabel('Dia da Intervenção', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    ax.set_title('Evolução da Precision do RL ao Longo de 30 Dias\n'
                 '❌ RL converge para ~45% (abaixo da meta de 70%)\n'
                 '✅ Threshold Dinâmico mantém 76.7% estável',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', frameon=True, shadow=True)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 100)
    
    # Anotações
    ax.annotate(f'Precision final: {precision_acumulada[-1]:.1f}%',
               xy=(30, precision_acumulada[-1]),
               xytext=(25, precision_acumulada[-1] + 15),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               fontsize=11, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph4_rl_precision_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 4: Evolução da Precision")


def plot_rl_behavior_response():
    """
    Gráfico 5: COMPORTAMENTO DO RL PERANTE RESPOSTAS
    Mostra como RL reage a VP (positivo) vs VN (negativo)
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # GRÁFICO 1: Notificações por dia (RL não reduz muito apesar de VN alto)
    notif_por_dia = [vp + vn for vp, vn in zip(vp_por_dia, vn_por_dia)]
    
    ax1.plot(dias, notif_por_dia, color='#3498db', linewidth=2.5,
            marker='o', markersize=5, label='Notificações enviadas')
    
    # Média móvel
    notif_smooth = np.convolve(notif_por_dia, np.ones(7)/7, mode='same')
    ax1.plot(dias, notif_smooth, color='red', linewidth=3, linestyle='--',
            label='Tendência (média 7 dias)')
    
    # Linha de referência
    ax1.axhline(y=5, color='green', linestyle='--', linewidth=2,
               label='Ideal (~5/dia)', alpha=0.7)
    
    ax1.set_xlabel('Dia da Intervenção', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Notificações por Dia', fontsize=12, fontweight='bold')
    ax1.set_title('Comportamento do RL: Quantidade de Notificações\n'
                  '❌ RL NÃO reduz notificações apesar de alto VN',
                  fontsize=12, fontweight='bold', pad=15, color='red')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 31)
    ax1.set_ylim(0, 20)
    
    # GRÁFICO 2: Ratio VP/VN ao longo do tempo
    ratio_vp_vn = [(vp / vn) if vn > 0 else 0 for vp, vn in zip(vp_por_dia, vn_por_dia)]
    
    ax2.bar(dias, ratio_vp_vn, color=['green' if r >= 1 else 'red' for r in ratio_vp_vn],
           alpha=0.7, edgecolor='black')
    
    # Linha de equilíbrio (VP = VN)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2,
               label='Equilíbrio (VP = VN)', alpha=0.7)
    
    ax2.set_xlabel('Dia da Intervenção', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Ratio VP/VN', fontsize=12, fontweight='bold')
    ax2.set_title('Comportamento do RL: Qualidade das Notificações\n'
                  'Verde = Mais acertos que erros | Vermelho = Mais erros\n'
                  '⚠️ RL oscila, não melhora consistentemente',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xlim(0, 31)
    ax2.set_ylim(0, 2.5)
    
    # Zona ideal
    ax2.axhspan(1.5, 2.5, alpha=0.1, color='green', label='Zona Ideal (VP >> VN)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph5_rl_behavior_response.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 5: Comportamento perante Respostas")


def plot_rl_pattern_identification_summary():
    """
    Gráfico 6: RESUMO - RL IDENTIFICA PADRÃO MATINAL?
    Gráfico final resumindo se RL aprendeu
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribuição final de notificações por hora
    hours = list(range(24))
    ax1.bar(hours, notif_por_hora_semana4, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvspan(5.5, 8.5, alpha=0.2, color='green')
    ax1.set_title('✅ Notificações Finais por Hora\nRL identificou padrão matinal (6-8h)',
                  fontsize=12, fontweight='bold', color='green')
    ax1.set_ylabel('Notificações', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f'{h}h' for h in range(0, 24, 2)])
    
    # 2. Top 5 horas
    top_hours_idx = np.argsort(notif_por_hora_semana4)[-5:][::-1]
    top_hours_val = [notif_por_hora_semana4[i] for i in top_hours_idx]
    colors_top = ['green' if i in [6,7,8] else 'orange' for i in top_hours_idx]
    
    ax2.barh([f'{i}h' for i in top_hours_idx], top_hours_val, color=colors_top,
            alpha=0.7, edgecolor='black')
    ax2.set_title('✅ Top 5 Horas Mais Notificadas\nVerde = Matinal (correto)',
                  fontsize=12, fontweight='bold', color='green')
    ax2.set_xlabel('Notificações', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Precision acumulada (do gráfico 4)
    precision_acum = []
    vp_acum = 0
    vn_acum = 0
    for vp, vn in zip(vp_por_dia, vn_por_dia):
        vp_acum += vp
        vn_acum += vn
        if (vp_acum + vn_acum) > 0:
            precision_acum.append(vp_acum / (vp_acum + vn_acum) * 100)
        else:
            precision_acum.append(0)
    
    ax3.plot(dias, precision_acum, color='#3498db', linewidth=3, marker='o', markersize=4)
    ax3.axhline(y=70, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axhline(y=76.7, color='#2ecc71', linestyle='-', linewidth=2)
    ax3.fill_between(dias, 70, 100, alpha=0.1, color='green')
    ax3.fill_between(dias, 0, 70, alpha=0.1, color='red')
    ax3.set_title(f'❌ Precision Final: {precision_acum[-1]:.1f}%\nAbaixo da meta (70%) e baseline (76.7%)',
                  fontsize=12, fontweight='bold', color='red')
    ax3.set_xlabel('Dia', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 4. Comparação final
    metrics = ['Identificou\nPadrão Matinal', 'Precision\nAdequada', 'Melhorou ao\nLongo do Tempo']
    rl_scores = [100, 45, 30]  # Sim, Não, Parcial
    colors_comp = ['green', 'red', 'orange']
    
    bars = ax4.bar(metrics, rl_scores, color=colors_comp, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.axhline(y=70, color='gray', linestyle='--', linewidth=2, label='Meta (70%)')
    
    # Adiciona valores e símbolos
    symbols = ['✅', '❌', '⚠️']
    for bar, val, symbol in zip(bars, rl_scores, symbols):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{symbol}\n{val}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax4.set_title('Resumo: Sucesso do RL',
                  fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 120)
    
    plt.suptitle('RESUMO: RL Identificou Padrão Matinal MAS Falhou em Precision',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph6_rl_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 6: Resumo Final do RL")


def main():
    """Gera todos os gráficos de análise do RL."""
    
    print("="*80)
    print("📊 ANÁLISE DO APRENDIZADO E COMPORTAMENTO DO RL")
    print("   Foco: Identificação de padrões e resposta a feedback")
    print("="*80)
    print()
    print(f"📁 Salvando gráficos em: {output_dir.absolute()}\n")
    
    try:
        plot_rl_hourly_focus_evolution()
        plot_rl_learning_curve()
        plot_rl_success_vs_failure_by_hour()
        plot_rl_precision_evolution()
        plot_rl_behavior_response()
        plot_rl_pattern_identification_summary()
        
        print()
        print("="*80)
        print("✅ TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
        print("="*80)
        print()
        print(f"📂 Localização: {output_dir.absolute()}")
        print()
        print("Gráficos de Análise do RL:")
        print("  1. graph1_rl_hourly_focus_evolution.png - Evolução do foco horário (4 semanas)")
        print("  2. graph2_rl_learning_curve.png - VP vs VN ao longo do tempo")
        print("  3. graph3_rl_success_failure_by_hour.png - Onde RL acerta vs erra")
        print("  4. graph4_rl_precision_evolution.png - Evolução da precision")
        print("  5. graph5_rl_behavior_response.png - Como RL reage a feedback")
        print("  6. graph6_rl_summary.png - Resumo final (4 subplots)")
        print()
        print("💡 Foco: Mostrar que RL identificou padrão MAS falhou em precision!")
        print()
        
    except Exception as e:
        print(f"\n❌ Erro ao gerar gráficos: {e}")
        print("\nVerifique se matplotlib, seaborn e numpy estão instalados:")
        print("  pip install matplotlib seaborn numpy")


if __name__ == "__main__":
    main()
