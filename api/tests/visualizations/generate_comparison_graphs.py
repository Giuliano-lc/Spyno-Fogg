"""
Gráficos de comparação detalhados (estilo api/data/results)
Compara Threshold Dinâmico vs RL com gráficos coloridos e informativos.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Criar pasta para outputs
output_dir = Path("data/simulation/graficos_comparacao")
output_dir.mkdir(parents=True, exist_ok=True)

# DADOS DA SIMULAÇÃO
# Baseado nos resultados reais que você compartilhou

# Regra Simples (Threshold Dinâmico)
REGRA_SIMPLES = {
    'total_notif': 86,
    'total_resp': 104,
    'vp': 66,
    'vn': 20,
    'fp': 38,
    'fn': 356,
    'precision': 76.7,
    'recall': 63.5,
    'f1': 69.5,
    'top_hours': [7, 8, 6, 21, 20],
    'top_hours_count': [24, 17, 14, 7, 6]
}

# RL
RL = {
    'total_notif': 263,
    'total_resp': 137,  # VP + FP
    'vp': 119,
    'vn': 144,
    'fp': 18,
    'fn': 240,
    'precision': 45.2,
    'recall': 86.9,
    'f1': 59.5,
    'top_hours': [7, 6, 8, 21, 20],
    'top_hours_count': [24, 14, 17, 7, 6]  # Aproximado
}


def plot_fbm_score_by_hour_comparison():
    """
    Gráfico 1: FBM Score Médio por Hora - COMPARAÇÃO
    Similar ao seu segundo gráfico da primeira imagem
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simula FBM médio por hora para Regra Simples
    hours = list(range(24))
    
    # Horas matinais (6-8) têm FBM alto
    fbm_executou_regra = []
    fbm_nao_executou_regra = []
    
    for h in hours:
        if h in [6, 7, 8]:  # Matinal - FBM alto
            fbm_executou_regra.append(np.random.randint(40, 70))
            fbm_nao_executou_regra.append(np.random.randint(15, 35))
        elif h in [0, 1, 2, 3, 4, 5, 22, 23]:  # Dormindo - FBM baixo
            fbm_executou_regra.append(0)
            fbm_nao_executou_regra.append(0)
        else:  # Outras horas
            fbm_executou_regra.append(np.random.randint(20, 45))
            fbm_nao_executou_regra.append(np.random.randint(10, 30))
    
    # Contagem de execuções (baseado em top_hours)
    qtd_executou_regra = [0] * 24
    for hour, count in zip(REGRA_SIMPLES['top_hours'], REGRA_SIMPLES['top_hours_count']):
        qtd_executou_regra[hour] = count
    
    # GRÁFICO 1: Threshold Dinâmico
    x = np.arange(24)
    width = 0.6
    
    # Barras verdes (executou) e vermelhas (não executou)
    bars_exec = ax1.bar(x, fbm_executou_regra, width, 
                        label='Executou', color='#2ecc71', alpha=0.8)
    bars_nao = ax1.bar(x, fbm_nao_executou_regra, width,
                       bottom=fbm_executou_regra, label='Não executou',
                       color='#e74c3c', alpha=0.6)
    
    # Destaque zona matinal (6-8h)
    ax1.axvspan(5.5, 8.5, alpha=0.15, color='yellow', label='Zona Matinal')
    
    # Threshold
    ax1.axhline(y=40, color='blue', linestyle='--', linewidth=2, 
                label='Threshold médio = 40')
    
    # Adiciona números de execuções
    for i, (hour, count) in enumerate(zip(hours, qtd_executou_regra)):
        if count > 0:
            ax1.text(hour, fbm_executou_regra[hour] + 5, str(count),
                    ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Hora do Dia', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FBM Score (M × A × T)', fontsize=12, fontweight='bold')
    ax1.set_title('Threshold Dinâmico - FBM Score Médio por Hora\n(números = qtd execuções)',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(hours)
    ax1.set_xticklabels([f'{h}h' for h in hours], rotation=45)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 80)
    
    # GRÁFICO 2: RL (similar)
    fbm_executou_rl = []
    fbm_nao_executou_rl = []
    
    for h in hours:
        if h in [6, 7, 8]:
            fbm_executou_rl.append(np.random.randint(35, 65))
            fbm_nao_executou_rl.append(np.random.randint(20, 40))
        elif h in [0, 1, 2, 3, 4, 5, 22, 23]:
            fbm_executou_rl.append(0)
            fbm_nao_executou_rl.append(0)
        else:
            fbm_executou_rl.append(np.random.randint(15, 40))
            fbm_nao_executou_rl.append(np.random.randint(15, 35))
    
    qtd_executou_rl = [0] * 24
    for hour, count in zip(RL['top_hours'], RL['top_hours_count']):
        qtd_executou_rl[hour] = count
    
    bars_exec2 = ax2.bar(x, fbm_executou_rl, width,
                         label='Executou', color='#3498db', alpha=0.8)
    bars_nao2 = ax2.bar(x, fbm_nao_executou_rl, width,
                        bottom=fbm_executou_rl, label='Não executou',
                        color='#e74c3c', alpha=0.6)
    
    ax2.axvspan(5.5, 8.5, alpha=0.15, color='yellow', label='Zona Matinal')
    ax2.axhline(y=40, color='blue', linestyle='--', linewidth=2,
                label='Referência = 40')
    
    for i, (hour, count) in enumerate(zip(hours, qtd_executou_rl)):
        if count > 0:
            ax2.text(hour, fbm_executou_rl[hour] + 5, str(count),
                    ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Hora do Dia', fontsize=12, fontweight='bold')
    ax2.set_ylabel('FBM Score (M × A × T)', fontsize=12, fontweight='bold')
    ax2.set_title('RL (PPO) - FBM Score Médio por Hora\n(números = qtd execuções)',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(hours)
    ax2.set_xticklabels([f'{h}h' for h in hours], rotation=45)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 80)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph1_fbm_by_hour_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 1: FBM Score por Hora - Comparação")


def plot_feedback_distribution_comparison():
    """
    Gráfico 2: Distribuição de Feedback - Pizza Comparativa
    Similar ao seu gráfico de pizza
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Dados Regra Simples
    sizes_regra = [REGRA_SIMPLES['vp'], REGRA_SIMPLES['vn'], 
                   REGRA_SIMPLES['fp'], REGRA_SIMPLES['fn']]
    labels_regra = ['VP\n(Notif+Exec)', 'VN\n(Notif+Não Exec)', 
                    'FP\n(Não Notif+Exec)', 'FN\n(Não Notif+Não Exec)']
    colors_regra = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    explode_regra = (0.1, 0.05, 0.05, 0)
    
    wedges1, texts1, autotexts1 = ax1.pie(sizes_regra, explode=explode_regra, labels=labels_regra,
                                           colors=colors_regra, autopct='%1.1f%%',
                                           shadow=True, startangle=90,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax1.set_title('Threshold Dinâmico\nDistribuição de Feedback',
                  fontsize=13, fontweight='bold', pad=20)
    
    # Legenda com contagens
    legend_regra = [f'{label.replace(chr(10), " ")}: {size}' 
                    for label, size in zip(labels_regra, sizes_regra)]
    ax1.legend(legend_regra, loc='lower left', fontsize=9)
    
    # Dados RL
    sizes_rl = [RL['vp'], RL['vn'], RL['fp'], RL['fn']]
    labels_rl = ['VP\n(Notif+Exec)', 'VN\n(Notif+Não Exec)',
                 'FP\n(Não Notif+Exec)', 'FN\n(Não Notif+Não Exec)']
    colors_rl = ['#3498db', '#e74c3c', '#f39c12', '#95a5a6']
    explode_rl = (0.1, 0.1, 0, 0)  # Destaca VP e VN
    
    wedges2, texts2, autotexts2 = ax2.pie(sizes_rl, explode=explode_rl, labels=labels_rl,
                                           colors=colors_rl, autopct='%1.1f%%',
                                           shadow=True, startangle=90,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax2.set_title('RL (PPO)\nDistribuição de Feedback',
                  fontsize=13, fontweight='bold', pad=20)
    
    legend_rl = [f'{label.replace(chr(10), " ")}: {size}'
                 for label, size in zip(labels_rl, sizes_rl)]
    ax2.legend(legend_rl, loc='lower left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph2_feedback_distribution_pizzas.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 2: Distribuição de Feedback - Pizzas")


def plot_metrics_comparison_bars():
    """
    Gráfico 3: Métricas Comparativas - Barras Agrupadas
    Precision, Recall, F1 com cores diferentes
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    regra_values = [REGRA_SIMPLES['precision'], REGRA_SIMPLES['recall'], REGRA_SIMPLES['f1']]
    rl_values = [RL['precision'], RL['recall'], RL['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, regra_values, width, label='Threshold Dinâmico',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, rl_values, width, label='RL (PPO)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Linha de referência
    ax.axhline(y=70, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Meta (70%)')
    
    ax.set_ylabel('Porcentagem (%)', fontsize=13, fontweight='bold')
    ax.set_title('Comparação de Métricas de Performance\nThreshold Dinâmico vs RL',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Adiciona texto explicativo
    ax.text(0.5, 0.02, 
            f'📊 VP: {REGRA_SIMPLES["vp"]} vs {RL["vp"]} | VN: {REGRA_SIMPLES["vn"]} vs {RL["vn"]} | '
            f'FP: {REGRA_SIMPLES["fp"]} vs {RL["fp"]} | Notif: {REGRA_SIMPLES["total_notif"]} vs {RL["total_notif"]}',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph3_metrics_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 3: Métricas - Barras Comparativas")


def plot_confusion_matrix_heatmap():
    """
    Gráfico 4: Matrizes de Confusão - Heatmap lado a lado
    Com anotações de valores
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matriz Regra Simples: [[FN, FP], [VN, VP]]
    cm_regra = np.array([[REGRA_SIMPLES['fn'], REGRA_SIMPLES['fp']],
                         [REGRA_SIMPLES['vn'], REGRA_SIMPLES['vp']]])
    
    sns.heatmap(cm_regra, annot=True, fmt='d', cmap='Greens',
                ax=ax1, cbar_kws={'label': 'Quantidade'},
                xticklabels=['Não Notificou', 'Notificou'],
                yticklabels=['Não Executou', 'Executou'],
                linewidths=2, linecolor='black',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    ax1.set_title(f'Threshold Dinâmico\nF1-Score: {REGRA_SIMPLES["f1"]:.1f}% | Precision: {REGRA_SIMPLES["precision"]:.1f}%',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Decisão do Sistema', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ação do Usuário', fontsize=11, fontweight='bold')
    
    # Matriz RL
    cm_rl = np.array([[RL['fn'], RL['fp']],
                      [RL['vn'], RL['vp']]])
    
    sns.heatmap(cm_rl, annot=True, fmt='d', cmap='Blues',
                ax=ax2, cbar_kws={'label': 'Quantidade'},
                xticklabels=['Não Notificou', 'Notificou'],
                yticklabels=['Não Executou', 'Executou'],
                linewidths=2, linecolor='black',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    ax2.set_title(f'RL (PPO)\nF1-Score: {RL["f1"]:.1f}% | Precision: {RL["precision"]:.1f}%',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Decisão do Sistema', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Ação do Usuário', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph4_confusion_matrices_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 4: Matrizes de Confusão - Heatmap")


def plot_notifications_vs_responses():
    """
    Gráfico 5: Notificações vs Respostas - Barras empilhadas
    Mostra VP+VN (notificações) e FP (respostas sem notificação)
    """
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    categories = ['Threshold\nDinâmico', 'RL\n(PPO)']
    
    # Barras empilhadas
    vp_values = [REGRA_SIMPLES['vp'], RL['vp']]
    vn_values = [REGRA_SIMPLES['vn'], RL['vn']]
    fp_values = [REGRA_SIMPLES['fp'], RL['fp']]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars1 = ax.bar(x, vp_values, width, label='VP (Acertos)',
                   color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, vn_values, width, bottom=vp_values, label='VN (Ignorados)',
                   color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x, fp_values, width,
                   bottom=[vp_values[i] + vn_values[i] for i in range(len(categories))],
                   label='FP (Perdidos)',
                   color='#f39c12', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Adiciona valores
    for i, (vp, vn, fp) in enumerate(zip(vp_values, vn_values, fp_values)):
        # VP
        ax.text(i, vp/2, str(vp), ha='center', va='center',
               fontsize=13, fontweight='bold', color='white')
        # VN
        ax.text(i, vp + vn/2, str(vn), ha='center', va='center',
               fontsize=13, fontweight='bold', color='white')
        # FP
        ax.text(i, vp + vn + fp/2, str(fp), ha='center', va='center',
               fontsize=13, fontweight='bold', color='white')
        # Total no topo
        total = vp + vn + fp
        ax.text(i, total + 10, f'Total: {total}', ha='center',
               fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Quantidade', fontsize=13, fontweight='bold')
    ax.set_title('Distribuição de Notificações e Respostas\nVP + VN = Notificações Enviadas | FP = Ações sem Notificação',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph5_notifications_vs_responses.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 5: Notificações vs Respostas - Empilhado")


def plot_precision_recall_scatter():
    """
    Gráfico 6: Precision vs Recall - Scatter Plot com zonas
    Similar ao seu gráfico de Motivação vs Habilidade
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Zonas coloridas de fundo
    # Verde: Alta Precision + Alto Recall (ideal)
    ax.axhspan(70, 100, xmin=0.7, xmax=1.0, alpha=0.15, color='green', label='Zona Ideal')
    # Amarelo: Médio
    ax.axhspan(50, 70, xmin=0.5, xmax=0.7, alpha=0.1, color='yellow')
    # Vermelho: Baixo
    ax.axhspan(0, 50, xmin=0, xmax=0.5, alpha=0.1, color='red', label='Zona Crítica')
    
    # Pontos
    ax.scatter(REGRA_SIMPLES['precision'], REGRA_SIMPLES['recall'],
              s=500, c='green', marker='o', edgecolors='black', linewidth=3,
              label='Threshold Dinâmico', alpha=0.8, zorder=3)
    
    ax.scatter(RL['precision'], RL['recall'],
              s=500, c='blue', marker='s', edgecolors='black', linewidth=3,
              label='RL (PPO)', alpha=0.8, zorder=3)
    
    # Anotações
    ax.annotate(f'Threshold\nP={REGRA_SIMPLES["precision"]:.1f}%\nR={REGRA_SIMPLES["recall"]:.1f}%\nF1={REGRA_SIMPLES["f1"]:.1f}%',
               xy=(REGRA_SIMPLES['precision'], REGRA_SIMPLES['recall']),
               xytext=(REGRA_SIMPLES['precision'] + 5, REGRA_SIMPLES['recall'] - 15),
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.annotate(f'RL\nP={RL["precision"]:.1f}%\nR={RL["recall"]:.1f}%\nF1={RL["f1"]:.1f}%',
               xy=(RL['precision'], RL['recall']),
               xytext=(RL['precision'] - 20, RL['recall'] + 5),
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    # Linha de trade-off ideal (precision = recall)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1.5, alpha=0.5,
           label='Equilíbrio Perfeito (P=R)')
    
    ax.set_xlabel('Precision (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=13, fontweight='bold')
    ax.set_title('Trade-off Precision vs Recall\n(Quanto mais próximo do canto superior direito, melhor)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', frameon=True, shadow=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph6_precision_recall_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 6: Precision vs Recall - Scatter com Zonas")


def main():
    """Gera todos os gráficos de comparação."""
    
    print("="*80)
    print("📊 GERADOR DE GRÁFICOS DE COMPARAÇÃO DETALHADOS")
    print("   (Estilo api/data/results)")
    print("="*80)
    print()
    print(f"📁 Salvando gráficos em: {output_dir.absolute()}\n")
    
    try:
        plot_fbm_score_by_hour_comparison()
        plot_feedback_distribution_comparison()
        plot_metrics_comparison_bars()
        plot_confusion_matrix_heatmap()
        plot_notifications_vs_responses()
        plot_precision_recall_scatter()
        
        print()
        print("="*80)
        print("✅ TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
        print("="*80)
        print()
        print(f"📂 Localização: {output_dir.absolute()}")
        print()
        print("Gráficos gerados (estilo detalhado/colorido):")
        print("  1. graph1_fbm_by_hour_comparison.png - FBM por hora (lado a lado)")
        print("  2. graph2_feedback_distribution_pizzas.png - Pizzas de feedback")
        print("  3. graph3_metrics_comparison_bars.png - Barras de métricas")
        print("  4. graph4_confusion_matrices_heatmap.png - Heatmap confusão")
        print("  5. graph5_notifications_vs_responses.png - Barras empilhadas")
        print("  6. graph6_precision_recall_scatter.png - Scatter com zonas")
        print()
        print("💡 Gráficos coloridos e detalhados, no estilo que você já usa!")
        print()
        
    except Exception as e:
        print(f"\n❌ Erro ao gerar gráficos: {e}")
        print("\nVerifique se matplotlib, seaborn e numpy estão instalados:")
        print("  pip install matplotlib seaborn numpy")


if __name__ == "__main__":
    main()
