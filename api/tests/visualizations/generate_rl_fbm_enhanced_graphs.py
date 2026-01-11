"""
Gera gráficos da simulação RL com FBM Melhorado
Mostra métricas por faixa de FBM e validação do aprendizado
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

def load_simulation_data():
    """Carrega dados da simulação RL FBM enhanced."""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'simulation' / 'rl_fbm_enhanced_simulation_data.json'
    
    print(f"📂 Carregando dados de: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Dados carregados: {len(data['daily_results'])} dias")
    return data

def create_graphs(data, output_dir):
    """Cria todos os gráficos da simulação."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gráfico 1: Evolução do Foco Horário do RL
    print("📊 Gráfico 1: Evolução do Foco Horário...")
    create_hourly_focus_evolution(data, output_dir)
    
    # Gráfico 2: Métricas por Faixa de FBM
    print("📊 Gráfico 2: Métricas por Faixa de FBM...")
    create_fbm_metrics_comparison(data, output_dir)
    
    # Gráfico 3: Distribuição de Notificações por Hora e FBM
    print("📊 Gráfico 3: Distribuição de Notificações...")
    create_notification_distribution(data, output_dir)
    
    # Gráfico 4: Evolução de Epsilon e Performance
    print("📊 Gráfico 4: Epsilon e Performance ao Longo dos Dias...")
    create_epsilon_performance_evolution(data, output_dir)
    
    # Gráfico 5: Heatmap de Decisões do RL
    print("📊 Gráfico 5: Heatmap de Decisões...")
    create_decision_heatmap(data, output_dir)

def create_hourly_focus_evolution(data, output_dir):
    """Mostra como o foco do RL evoluiu ao longo dos dias."""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Divide em 4 semanas
    weeks = [
        (1, 7, "Semana 1 (Dias 1-7): RL Explorando"),
        (8, 14, "Semana 2 (Dias 8-14): RL Ajustando"),
        (15, 21, "Semana 3 (Dias 15-21): RL Aprendendo"),
        (22, 30, "Semana 4 (Dias 22-30): RL Estável")
    ]
    
    for idx, (start, end, title) in enumerate(weeks):
        ax = axes[idx // 2, idx % 2]
        
        # Contabiliza notificações por hora nesse período
        hourly_notifs = {h: 0 for h in range(24)}
        
        for day_result in data['daily_results']:
            day_num = data['daily_results'].index(day_result) + 1
            if start <= day_num <= end:
                for hour_result in day_result['results']:
                    if hour_result.get('rl_notified'):
                        hourly_notifs[hour_result['hour']] += 1
        
        hours = list(range(24))
        notifs = [hourly_notifs[h] for h in hours]
        
        bars = ax.bar(hours, notifs, color='#3498DB', alpha=0.7)
        
        # Destaca zonas de FBM
        ax.axvspan(6, 11, alpha=0.2, color='#27AE60', label='Manhã (FBM Alto)')
        ax.axvspan(12, 17, alpha=0.2, color='#E74C3C', label='Tarde (FBM Baixo)')
        ax.axvspan(18, 23, alpha=0.2, color='#3498DB', label='Noite (FBM Alto)')
        
        ax.set_xlabel('Hora do Dia', fontsize=12, fontweight='bold')
        ax.set_ylabel('Notificações', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(hours)
        ax.set_xticklabels([f'{h}h' for h in hours], rotation=45)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Evolução do Foco Horário do RL - Identificação de Padrão Matinal', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_hourly_focus_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 1 salvo")

def create_fbm_metrics_comparison(data, output_dir):
    """Compara métricas entre faixas de FBM."""
    
    fbm_metrics = data['summary']['fbm_metrics']
    
    categories = ['alto', 'medio', 'baixo']
    labels = ['FBM Alto\n(≥60)', 'FBM Médio\n(40-59)', 'FBM Baixo\n(<40)']
    
    precision = [fbm_metrics[cat]['precision'] for cat in categories]
    recall = [fbm_metrics[cat]['recall'] for cat in categories]
    vp = [fbm_metrics[cat]['vp'] for cat in categories]
    vn = [fbm_metrics[cat]['vn'] for cat in categories]
    notifs = [fbm_metrics[cat]['notified'] for cat in categories]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Precision e Recall
    ax1 = axes[0, 0]
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, precision, width, label='Precision', color='#27AE60', alpha=0.8)
    bars2 = ax1.bar(x + width/2, recall, width, label='Recall', color='#3498DB', alpha=0.8)
    
    ax1.set_ylabel('Porcentagem (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Precision e Recall por Faixa de FBM', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # VP e VN
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width/2, vp, width, label='VP (Acertos)', color='#27AE60', alpha=0.8)
    bars2 = ax2.bar(x + width/2, vn, width, label='VN (Erros)', color='#E74C3C', alpha=0.8)
    
    ax2.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax2.set_title('VP (Acertos) vs VN (Erros) por Faixa de FBM', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Notificações
    ax3 = axes[1, 0]
    colors = ['#27AE60', '#F39C12', '#E74C3C']
    bars = ax3.bar(labels, notifs, color=colors, alpha=0.7)
    
    ax3.set_ylabel('Notificações Enviadas', fontsize=12, fontweight='bold')
    ax3.set_title('Total de Notificações por Faixa de FBM', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Taxa de Sucesso (VP / Notificações)
    ax4 = axes[1, 1]
    success_rate = [(fbm_metrics[cat]['vp'] / fbm_metrics[cat]['notified'] * 100) 
                    if fbm_metrics[cat]['notified'] > 0 else 0 
                    for cat in categories]
    
    bars = ax4.bar(labels, success_rate, color=colors, alpha=0.7)
    
    ax4.set_ylabel('Taxa de Sucesso (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Taxa de Sucesso (VP / Notificações) por Faixa de FBM', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Comparação de Métricas por Faixa de FBM', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_fbm_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 2 salvo")

def create_notification_distribution(data, output_dir):
    """Mostra distribuição de notificações por hora e categoria FBM."""
    
    # Contabiliza notificações por hora e categoria FBM
    hourly_fbm_data = {
        'alto': {h: 0 for h in range(24)},
        'baixo': {h: 0 for h in range(24)}
    }
    
    for day_result in data['daily_results']:
        for hour_result in day_result['results']:
            if hour_result.get('rl_notified'):
                hour = hour_result['hour']
                fbm_cat = hour_result.get('fbm_category', 'medio')
                if fbm_cat in ['alto', 'baixo']:
                    hourly_fbm_data[fbm_cat][hour] += 1
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    hours = list(range(24))
    alto_data = [hourly_fbm_data['alto'][h] for h in hours]
    baixo_data = [hourly_fbm_data['baixo'][h] for h in hours]
    
    width = 0.4
    x = np.arange(len(hours))
    
    bars1 = ax.bar(x - width/2, alto_data, width, label='FBM Alto (≥60)', color='#27AE60', alpha=0.8)
    bars2 = ax.bar(x + width/2, baixo_data, width, label='FBM Baixo (<40)', color='#E74C3C', alpha=0.8)
    
    # Destaca períodos
    ax.axvspan(5.5, 11.5, alpha=0.15, color='#27AE60', zorder=0)
    ax.axvspan(17.5, 23.5, alpha=0.15, color='#3498DB', zorder=0)
    
    ax.set_xlabel('Hora do Dia', fontsize=14, fontweight='bold')
    ax.set_ylabel('Notificações Enviadas', fontsize=14, fontweight='bold')
    ax.set_title('Distribuição de Notificações por Hora e Faixa de FBM (30 dias)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}h' for h in hours])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_notification_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 3 salvo")

def create_epsilon_performance_evolution(data, output_dir):
    """Mostra evolução de epsilon e performance ao longo dos dias."""
    
    days = list(range(1, 31))
    
    # Extrai métricas diárias
    daily_vp = [d['day_vp'] for d in data['daily_results']]
    daily_vn = [d['day_vn'] for d in data['daily_results']]
    daily_fbm_avg = [d.get('fbm_avg', 59.3) for d in data['daily_results']]
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # VP e VN ao longo dos dias
    ax1 = axes[0]
    ax1.plot(days, daily_vp, marker='o', linewidth=2, color='#27AE60', 
             label='VP (Acertos)', markersize=6)
    ax1.plot(days, daily_vn, marker='s', linewidth=2, color='#E74C3C', 
             label='VN (Erros)', markersize=6)
    ax1.fill_between(days, daily_vp, alpha=0.3, color='#27AE60')
    ax1.fill_between(days, daily_vn, alpha=0.3, color='#E74C3C')
    
    ax1.set_xlabel('Dia da Simulação', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
    ax1.set_title('Evolução de VP (Acertos) e VN (Erros) ao Longo de 30 Dias', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 31)
    
    # Taxa de acerto (VP / (VP+VN))
    ax2 = axes[1]
    accuracy = [(daily_vp[i] / (daily_vp[i] + daily_vn[i]) * 100) 
                if (daily_vp[i] + daily_vn[i]) > 0 else 0 
                for i in range(len(days))]
    
    ax2.plot(days, accuracy, marker='D', linewidth=2.5, color='#3498DB', 
             label='Taxa de Acerto', markersize=7)
    ax2.fill_between(days, accuracy, alpha=0.3, color='#3498DB')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (Baseline)')
    
    ax2.set_xlabel('Dia da Simulação', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Taxa de Acerto (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Taxa de Acerto (VP / Total Notificações) ao Longo de 30 Dias', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 31)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_epsilon_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 4 salvo")

def create_decision_heatmap(data, output_dir):
    """Heatmap mostrando quando o RL decidiu notificar."""
    
    # Matriz 30 dias x 24 horas
    decision_matrix = np.zeros((30, 24))
    outcome_matrix = np.full((30, 24), '', dtype=object)
    
    for day_idx, day_result in enumerate(data['daily_results']):
        for hour_result in day_result['results']:
            hour = hour_result['hour']
            if hour_result.get('rl_notified'):
                decision_matrix[day_idx, hour] = 1
                outcome = hour_result.get('outcome', 'N/A')
                outcome_matrix[day_idx, hour] = outcome
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    im = ax.imshow(decision_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h}h' for h in range(24)])
    ax.set_yticks(range(0, 30, 2))
    ax.set_yticklabels([f'Dia {i+1}' for i in range(0, 30, 2)])
    
    # Linhas para delimitar períodos
    ax.axvline(x=5.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=11.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=17.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Hora do Dia', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dias de Simulação', fontsize=14, fontweight='bold')
    ax.set_title('Heatmap de Decisões do RL: Quando Notificou (30 dias x 24 horas)\n' +
                 'Verde = Notificou | Branco = Não Notificou', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Labels dos períodos
    ax.text(3, -1.5, 'Madrugada\n(0-5h)', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#2C3E50', alpha=0.7, edgecolor='white'))
    ax.text(8.5, -1.5, 'Manhã - FBM ALTO\n(6-11h)', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#27AE60', alpha=0.7, edgecolor='white'))
    ax.text(14.5, -1.5, 'Tarde - FBM BAIXO\n(12-17h)', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E74C3C', alpha=0.7, edgecolor='white'))
    ax.text(20.5, -1.5, 'Noite - FBM ALTO\n(18-23h)', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#3498DB', alpha=0.7, edgecolor='white'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_decision_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 5 salvo")

def main():
    print("\n" + "="*80)
    print("📊 GERADOR DE GRÁFICOS - Simulação RL com FBM Melhorado")
    print("="*80 + "\n")
    
    # Carrega dados
    data = load_simulation_data()
    
    # Define diretório de saída com timestamp
    base_dir = Path(__file__).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / 'data' / 'simulation' / 'NovoPerfil' / f'graficos_rl_enhanced_{timestamp}'
    
    print(f"\n📁 Salvando gráficos em: {output_dir}\n")
    
    # Cria gráficos
    create_graphs(data, output_dir)
    
    print("\n" + "="*80)
    print("✅ GRÁFICOS GERADOS COM SUCESSO!")
    print("="*80)
    print(f"\n📂 Localização: {output_dir}")
    print("\nGráficos gerados:")
    print("  1. graph_hourly_focus_evolution.png - Evolução do foco horário por semana")
    print("  2. graph_fbm_metrics_comparison.png - Comparação de métricas por faixa de FBM")
    print("  3. graph_notification_distribution.png - Distribuição de notificações por hora")
    print("  4. graph_epsilon_performance.png - Evolução de performance ao longo dos dias")
    print("  5. graph_decision_heatmap.png - Heatmap de quando o RL decidiu notificar\n")

if __name__ == "__main__":
    main()
