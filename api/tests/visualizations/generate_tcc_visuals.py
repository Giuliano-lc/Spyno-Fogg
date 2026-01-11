"""
Script para gerar visualizações (gráficos) para inclusão no TCC.
Requer matplotlib e seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Criar pasta para outputs
output_dir = Path("data/simulation/graficos_tcc")
output_dir.mkdir(parents=True, exist_ok=True)


def plot_metrics_comparison():
    """Gráfico comparativo das métricas principais."""
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    regra_simples = [76.7, 63.5, 69.5]
    rl = [45.2, 86.9, 59.5]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, regra_simples, width, label='Regra Simples', 
                    color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, rl, width, label='RL (PPO)', 
                    color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Porcentagem (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Métricas: Regra Simples vs RL', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_comparacao_metricas.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 1: Comparação de Métricas")


def plot_confusion_matrix_comparison():
    """Gráficos de matriz de confusão lado a lado."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Regra Simples
    cm_regra = np.array([[356, 20], [38, 66]])
    sns.heatmap(cm_regra, annot=True, fmt='d', cmap='Greens', 
                ax=ax1, cbar_kws={'label': 'Quantidade'},
                xticklabels=['Não Notificou', 'Notificou'],
                yticklabels=['Não Agiu', 'Agiu'])
    ax1.set_title('Matriz de Confusão - Regra Simples\n(F1: 69.5%)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('Realidade', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predição', fontsize=11, fontweight='bold')
    
    # RL
    cm_rl = np.array([[240, 144], [18, 119]])
    sns.heatmap(cm_rl, annot=True, fmt='d', cmap='Blues', 
                ax=ax2, cbar_kws={'label': 'Quantidade'},
                xticklabels=['Não Notificou', 'Notificou'],
                yticklabels=['Não Agiu', 'Agiu'])
    ax2.set_title('Matriz de Confusão - RL (PPO)\n(F1: 59.5%)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel('Realidade', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Predição', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_matrizes_confusao.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 2: Matrizes de Confusão")


def plot_notifications_comparison():
    """Gráfico de barras comparando notificações enviadas."""
    
    categories = ['Notificações\nEnviadas', 'VP\n(Acertos)', 'VN\n(Ignorados)', 'FP\n(Perdidos)']
    regra_simples = [86, 66, 20, 38]
    rl = [263, 119, 144, 18]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, regra_simples, width, label='Regra Simples',
                    color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, rl, width, label='RL (PPO)',
                    color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Notificações e Feedback', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_comparacao_notificacoes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 3: Comparação de Notificações")


def plot_hourly_pattern():
    """Gráfico de padrão horário (horas mais ativas)."""
    
    # Top horas com respostas
    hours_regra = [7, 8, 6, 21, 20]
    responses_regra = [24, 17, 14, 7, 6]
    
    hours_rl = [7, 6, 8, 21, 20]
    responses_rl = [24, 14, 17, 7, 6]  # Aproximado
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Regra Simples
    colors1 = ['#27ae60' if h in [6,7,8] else '#95a5a6' for h in hours_regra]
    ax1.barh(range(len(hours_regra)), responses_regra, color=colors1, alpha=0.8)
    ax1.set_yticks(range(len(hours_regra)))
    ax1.set_yticklabels([f'{h:02d}h' for h in hours_regra], fontsize=11)
    ax1.set_xlabel('Número de Respostas', fontsize=11, fontweight='bold')
    ax1.set_title('Top 5 Horas - Regra Simples\n✅ Verde = Matinal (6h-8h)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # Adiciona valores
    for i, v in enumerate(responses_regra):
        ax1.text(v + 0.5, i, str(v), va='center', fontsize=10, fontweight='bold')
    
    # RL
    colors2 = ['#3498db' if h in [6,7,8] else '#95a5a6' for h in hours_rl]
    ax2.barh(range(len(hours_rl)), responses_rl, color=colors2, alpha=0.8)
    ax2.set_yticks(range(len(hours_rl)))
    ax2.set_yticklabels([f'{h:02d}h' for h in hours_rl], fontsize=11)
    ax2.set_xlabel('Número de Respostas', fontsize=11, fontweight='bold')
    ax2.set_title('Top 5 Horas - RL (PPO)\n✅ Azul = Matinal (6h-8h)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    # Adiciona valores
    for i, v in enumerate(responses_rl):
        ax2.text(v + 0.5, i, str(v), va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_padrao_horario.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 4: Padrão Horário")


def plot_threshold_evolution():
    """Gráfico de evolução do threshold ao longo dos dias."""
    
    days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17, 18, 19, 20, 22, 25, 26, 27, 28, 29, 30]
    threshold = [40, 49.16, 49.16, 52.90, 49.61, 63.48, 60.95, 56.76, 61.67, 58.87, 56.43, 58.09, 
                 69.25, 57.25, 49.92, 47.44, 60.65, 74.35, 64.35, 65.48, 66.91, 69.53, 65.10, 51.56, 41.69]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(days, threshold, marker='o', linewidth=2, markersize=6, 
            color='#e74c3c', label='Threshold Dinâmico')
    ax.axhline(y=40, color='gray', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='Threshold Inicial (40.0)')
    
    ax.set_xlabel('Dia da Simulação', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor do Threshold', fontsize=12, fontweight='bold')
    ax.set_title('Evolução do Threshold Dinâmico ao Longo de 30 Dias', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)
    
    # Destaca pico e vale
    max_idx = threshold.index(max(threshold))
    ax.annotate(f'Pico: {max(threshold):.1f}', 
                xy=(days[max_idx], threshold[max_idx]),
                xytext=(days[max_idx]+2, threshold[max_idx]+3),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_evolucao_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 5: Evolução do Threshold")


def plot_precision_recall_tradeoff():
    """Gráfico de trade-off Precision vs Recall."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Pontos
    ax.scatter(76.7, 63.5, s=300, c='#2ecc71', alpha=0.7, edgecolors='black', linewidth=2,
               label='Regra Simples', zorder=3)
    ax.scatter(45.2, 86.9, s=300, c='#3498db', alpha=0.7, edgecolors='black', linewidth=2,
               label='RL (PPO)', zorder=3)
    
    # Anotações
    ax.annotate('Regra Simples\nF1: 69.5%\nAlta Precision\nSpam Baixo', 
                xy=(76.7, 63.5), xytext=(80, 55),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.3))
    
    ax.annotate('RL (PPO)\nF1: 59.5%\nAlto Recall\nSpam Alto', 
                xy=(45.2, 86.9), xytext=(30, 78),
                arrowprops=dict(arrowstyle='->', lw=1.5),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db', alpha=0.3))
    
    # Zona ideal (alta precision E alto recall)
    from matplotlib.patches import Rectangle
    ideal_zone = Rectangle((65, 65), 35, 35, linewidth=2, 
                           edgecolor='green', facecolor='green', alpha=0.1, linestyle='--')
    ax.add_patch(ideal_zone)
    ax.text(82.5, 95, 'Zona Ideal', fontsize=11, ha='center', 
            color='green', fontweight='bold')
    
    ax.set_xlabel('Precision (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off Precision vs Recall\n(Maior F1-Score = Melhor Balanço)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_xlim(20, 100)
    ax.set_ylim(40, 100)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '6_precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 6: Trade-off Precision vs Recall")


def main():
    """Gera todos os gráficos."""
    
    print("="*80)
    print("📊 GERADOR DE VISUALIZAÇÕES PARA TCC")
    print("="*80)
    print()
    print(f"📁 Salvando gráficos em: {output_dir.absolute()}\n")
    
    try:
        plot_metrics_comparison()
        plot_confusion_matrix_comparison()
        plot_notifications_comparison()
        plot_hourly_pattern()
        plot_threshold_evolution()
        plot_precision_recall_tradeoff()
        
        print()
        print("="*80)
        print("✅ TODOS OS GRÁFICOS GERADOS COM SUCESSO!")
        print("="*80)
        print()
        print(f"📂 Localização: {output_dir.absolute()}")
        print()
        print("Gráficos gerados:")
        print("  1. 1_comparacao_metricas.png")
        print("  2. 2_matrizes_confusao.png")
        print("  3. 3_comparacao_notificacoes.png")
        print("  4. 4_padrao_horario.png")
        print("  5. 5_evolucao_threshold.png")
        print("  6. 6_precision_recall_tradeoff.png")
        print()
        print("💡 Use estes gráficos no seu TCC!")
        print()
        
    except Exception as e:
        print(f"\n❌ Erro ao gerar gráficos: {e}")
        print("\nVerifique se matplotlib e seaborn estão instalados:")
        print("  pip install matplotlib seaborn")


if __name__ == "__main__":
    main()
