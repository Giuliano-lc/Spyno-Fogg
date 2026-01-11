"""
Gera gráficos mostrando a evolução dos componentes FBM (M x A x T) durante a simulação.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

def calculate_fbm_scores(hour_data):
    """Calcula os scores M, A, T baseado nos fatores."""
    
    # MOTIVATION (0-4)
    motivation_factors = hour_data.get('motivation_factors', {})
    valence = motivation_factors.get('valence', 0)
    last_activity = motivation_factors.get('last_activity_score', 0)
    
    if valence >= 3:
        motivation = 4
    elif valence >= 2:
        motivation = 3
    elif valence >= 1:
        motivation = 2
    elif last_activity >= 3:
        motivation = 1
    else:
        motivation = 0
    
    # ABILITY (0-4)
    ability_factors = hour_data.get('ability_factors', {})
    cognitive_load = ability_factors.get('cognitive_load', 0)
    confidence = ability_factors.get('confidence_score', 0)
    
    if cognitive_load <= 2 and confidence >= 7:
        ability = 4
    elif cognitive_load <= 3 and confidence >= 5:
        ability = 3
    elif cognitive_load <= 5:
        ability = 2
    elif cognitive_load <= 7:
        ability = 1
    else:
        ability = 0
    
    # TRIGGER (0-6)
    trigger_factors = hour_data.get('trigger_factors', {})
    sleeping = trigger_factors.get('sleeping', False)
    arousal = trigger_factors.get('arousal', 0)
    location = trigger_factors.get('location', 'unknown')
    
    if sleeping:
        trigger = 0
    else:
        trigger = arousal
        if location == 'home':
            trigger += 1
    
    trigger = min(trigger, 6)
    
    # FBM Score (produto M x A x T)
    fbm_score = motivation * ability * trigger
    
    return {
        'motivation': motivation,
        'ability': ability,
        'trigger': trigger,
        'fbm_score': fbm_score
    }

def load_user_profile(filepath):
    """Carrega o perfil do usuário com dados sintéticos."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def aggregate_fbm_by_day_and_hour(profile_data):
    """Agrega os scores FBM por dia e hora."""
    daily_fbm = []
    hourly_avg = defaultdict(lambda: {'M': [], 'A': [], 'T': [], 'FBM': []})
    
    for day_idx, day_data in enumerate(profile_data['days'], 1):
        day_scores = []
        
        for hour_data in day_data['hours']:
            scores = calculate_fbm_scores(hour_data)
            hour = hour_data['hour']
            
            day_scores.append({
                'day': day_idx,
                'hour': hour,
                'date': day_data['date'],
                **scores
            })
            
            # Agrega por hora para médias
            hourly_avg[hour]['M'].append(scores['motivation'])
            hourly_avg[hour]['A'].append(scores['ability'])
            hourly_avg[hour]['T'].append(scores['trigger'])
            hourly_avg[hour]['FBM'].append(scores['fbm_score'])
        
        daily_fbm.extend(day_scores)
    
    return daily_fbm, hourly_avg

def create_fbm_evolution_graphs(profile_data, output_dir):
    """Cria gráficos de evolução dos componentes FBM."""
    
    daily_fbm, hourly_avg = aggregate_fbm_by_day_and_hour(profile_data)
    
    # Gráfico 1: Heatmap de FBM Score ao longo dos dias e horas
    print("📊 Gerando Gráfico 1: Heatmap FBM Score (Dia x Hora)...")
    
    # Organiza dados em matriz
    days = sorted(set(item['day'] for item in daily_fbm))
    hours = sorted(set(item['hour'] for item in daily_fbm))
    
    fbm_matrix = np.zeros((len(days), len(hours)))
    for item in daily_fbm:
        day_idx = days.index(item['day'])
        hour_idx = hours.index(item['hour'])
        fbm_matrix[day_idx, hour_idx] = item['fbm_score']
    
    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(fbm_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=96)
    
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f'{h}h' for h in hours])
    ax.set_yticks(range(0, len(days), 5))
    ax.set_yticklabels([f'Dia {days[i]}' for i in range(0, len(days), 5)])
    
    ax.set_xlabel('Hora do Dia', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dias de Simulação', fontsize=14, fontweight='bold')
    ax.set_title('Evolução do FBM Score (M × A × T) ao Longo de 30 Dias\nPerfil Matinal', 
                 fontsize=16, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('FBM Score', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_fbm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 1 salvo")
    
    # Gráfico 2: Evolução dos componentes M, A, T por hora (média de 30 dias)
    print("📊 Gerando Gráfico 2: Componentes M, A, T por Hora...")
    
    hours_sorted = sorted(hourly_avg.keys())
    m_avg = [np.mean(hourly_avg[h]['M']) for h in hours_sorted]
    a_avg = [np.mean(hourly_avg[h]['A']) for h in hours_sorted]
    t_avg = [np.mean(hourly_avg[h]['T']) for h in hours_sorted]
    fbm_avg = [np.mean(hourly_avg[h]['FBM']) for h in hours_sorted]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # M - Motivation
    axes[0, 0].plot(hours_sorted, m_avg, marker='o', linewidth=2, color='#FF6B6B', markersize=6)
    axes[0, 0].fill_between(hours_sorted, m_avg, alpha=0.3, color='#FF6B6B')
    axes[0, 0].set_title('Motivation (M) - Média por Hora', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Hora do Dia', fontsize=12)
    axes[0, 0].set_ylabel('Score (0-4)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.2, 4.5)
    
    # A - Ability
    axes[0, 1].plot(hours_sorted, a_avg, marker='s', linewidth=2, color='#4ECDC4', markersize=6)
    axes[0, 1].fill_between(hours_sorted, a_avg, alpha=0.3, color='#4ECDC4')
    axes[0, 1].set_title('Ability (A) - Média por Hora', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Hora do Dia', fontsize=12)
    axes[0, 1].set_ylabel('Score (0-4)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-0.2, 4.5)
    
    # T - Trigger
    axes[1, 0].plot(hours_sorted, t_avg, marker='^', linewidth=2, color='#95E1D3', markersize=6)
    axes[1, 0].fill_between(hours_sorted, t_avg, alpha=0.3, color='#95E1D3')
    axes[1, 0].set_title('Trigger (T) - Média por Hora', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Hora do Dia', fontsize=12)
    axes[1, 0].set_ylabel('Score (0-6)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.2, 6.5)
    
    # FBM Score (M × A × T)
    axes[1, 1].plot(hours_sorted, fbm_avg, marker='D', linewidth=2.5, color='#F38181', markersize=7)
    axes[1, 1].fill_between(hours_sorted, fbm_avg, alpha=0.3, color='#F38181')
    axes[1, 1].set_title('FBM Score (M × A × T) - Média por Hora', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Hora do Dia', fontsize=12)
    axes[1, 1].set_ylabel('Score (0-96)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Destaca horas matinais (6-8h)
    axes[1, 1].axvspan(6, 8, alpha=0.2, color='yellow', label='Pico Matinal')
    axes[1, 1].legend()
    
    plt.suptitle('Componentes FBM - Média de 30 Dias (Perfil Matinal)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_fbm_components_hourly.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 2 salvo")
    
    # Gráfico 3: Box plot dos componentes por período do dia
    print("📊 Gerando Gráfico 3: Distribuição FBM por Período...")
    
    # Categoriza por período
    periods = {
        'Madrugada\n(0-5h)': [],
        'Manhã\n(6-11h)': [],
        'Tarde\n(12-17h)': [],
        'Noite\n(18-23h)': []
    }
    
    for item in daily_fbm:
        hour = item['hour']
        fbm = item['fbm_score']
        
        if 0 <= hour <= 5:
            periods['Madrugada\n(0-5h)'].append(fbm)
        elif 6 <= hour <= 11:
            periods['Manhã\n(6-11h)'].append(fbm)
        elif 12 <= hour <= 17:
            periods['Tarde\n(12-17h)'].append(fbm)
        else:
            periods['Noite\n(18-23h)'].append(fbm)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    positions = [1, 2, 3, 4]
    data_to_plot = [periods[p] for p in periods.keys()]
    labels = list(periods.keys())
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)
    
    colors = ['#2C3E50', '#E74C3C', '#F39C12', '#3498DB']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(labels)
    ax.set_ylabel('FBM Score', fontsize=14, fontweight='bold')
    ax.set_title('Distribuição do FBM Score por Período do Dia\n30 Dias de Simulação', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adiciona médias
    for i, (period, data) in enumerate(periods.items(), 1):
        mean_val = np.mean(data)
        ax.text(i, mean_val + 5, f'μ={mean_val:.1f}', ha='center', fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'graph_fbm_distribution_by_period.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico 3 salvo")

def main():
    print("\n" + "="*80)
    print("📊 GERADOR DE GRÁFICOS - EVOLUÇÃO FBM (M × A × T)")
    print("="*80 + "\n")
    
    # Caminhos
    base_dir = Path(__file__).parent.parent
    profile_path = base_dir / 'data' / 'users' / 'user_matinal_rl_v2.json'
    output_dir = base_dir / 'data' / 'simulation' / 'graficos_rl_real'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Carregando perfil: {profile_path}")
    profile_data = load_user_profile(profile_path)
    num_days = len(profile_data['days'])
    print(f"✅ Perfil carregado: {num_days} dias\n")
    
    print(f"📁 Salvando gráficos em: {output_dir}\n")
    
    create_fbm_evolution_graphs(profile_data, output_dir)
    
    print("\n" + "="*80)
    print("✅ GRÁFICOS FBM GERADOS COM SUCESSO!")
    print("="*80)
    print(f"\n📂 Localização: {output_dir}")
    print("\nGráficos gerados:")
    print("  1. graph_fbm_heatmap.png - Heatmap FBM ao longo de 30 dias")
    print("  2. graph_fbm_components_hourly.png - Componentes M, A, T por hora")
    print("  3. graph_fbm_distribution_by_period.png - Distribuição por período\n")

if __name__ == "__main__":
    main()
