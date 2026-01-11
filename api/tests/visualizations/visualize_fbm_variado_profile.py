"""
Gera gráficos para visualizar os níveis de FBM do perfil user_fbm_variado.json
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 10)
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

def load_and_analyze_profile(filepath):
    """Carrega perfil e calcula FBM scores."""
    with open(filepath, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    
    # Organiza dados
    days = []
    hours = list(range(24))
    
    fbm_matrix = np.zeros((len(profile['days']), 24))
    
    for day_idx, day_data in enumerate(profile['days']):
        for hour_data in day_data['hours']:
            hour = hour_data['hour']
            scores = calculate_fbm_scores(hour_data)
            fbm_matrix[day_idx, hour] = scores['fbm_score']
    
    return fbm_matrix, profile

def create_visualization(fbm_matrix, profile, output_dir):
    """Cria visualizações do perfil FBM."""
    
    num_days = fbm_matrix.shape[0]
    hours = list(range(24))
    
    # Gráfico 1: Heatmap FBM Score
    print("📊 Gerando Heatmap FBM Score...")
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    im = ax.imshow(fbm_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=96)
    
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h}h' for h in hours])
    ax.set_yticks(range(0, num_days, 5))
    ax.set_yticklabels([f'Dia {i+1}' for i in range(0, num_days, 5)])
    
    ax.set_xlabel('Hora do Dia', fontsize=16, fontweight='bold')
    ax.set_ylabel('Dias de Simulação', fontsize=16, fontweight='bold')
    ax.set_title('FBM Score (M × A × T) - Perfil Manipulado\n' +
                 '🌅 Manhã (6-11h): ALTO | ☀️ Tarde (12-17h): BAIXO | 🌙 Noite (18-23h): ALTO', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Adiciona linhas para demarcar períodos
    ax.axvline(x=5.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=11.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=17.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    
    # Labels dos períodos
    ax.text(3, -2, 'Madrugada\n(0-5h)', ha='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='#2C3E50', alpha=0.7, edgecolor='white'))
    ax.text(8.5, -2, 'Manhã - ALTO\n(6-11h)', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#27AE60', alpha=0.7, edgecolor='white'))
    ax.text(14.5, -2, 'Tarde - BAIXO\n(12-17h)', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E74C3C', alpha=0.7, edgecolor='white'))
    ax.text(20.5, -2, 'Noite - ALTO\n(18-23h)', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#3498DB', alpha=0.7, edgecolor='white'))
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('FBM Score', fontsize=14, fontweight='bold')
    
    # Adiciona linhas de referência no colorbar
    cbar.ax.axhline(y=60, color='yellow', linestyle='--', linewidth=2)
    cbar.ax.text(1.5, 60, 'Alto (≥60)', fontsize=10, va='center', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fbm_variado_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Heatmap salvo")
    
    # Gráfico 2: Média por Hora com destaque de períodos
    print("📊 Gerando gráfico de médias por hora...")
    
    fbm_avg_by_hour = fbm_matrix.mean(axis=0)
    fbm_std_by_hour = fbm_matrix.std(axis=0)
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Plot principal
    ax.plot(hours, fbm_avg_by_hour, marker='o', linewidth=3, color='#2C3E50', 
            markersize=8, label='Média FBM', zorder=5)
    ax.fill_between(hours, fbm_avg_by_hour - fbm_std_by_hour, 
                     fbm_avg_by_hour + fbm_std_by_hour, 
                     alpha=0.3, color='#2C3E50', label='±1 Desvio Padrão')
    
    # Destaca períodos
    ax.axvspan(0, 5, alpha=0.2, color='#2C3E50', label='Madrugada (Baixo)')
    ax.axvspan(6, 11, alpha=0.3, color='#27AE60', label='Manhã (ALTO)')
    ax.axvspan(12, 17, alpha=0.3, color='#E74C3C', label='Tarde (BAIXO)')
    ax.axvspan(18, 23, alpha=0.3, color='#3498DB', label='Noite (ALTO)')
    
    # Linha de referência
    ax.axhline(y=60, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label='Threshold Alto (60)')
    
    ax.set_xlabel('Hora do Dia', fontsize=14, fontweight='bold')
    ax.set_ylabel('FBM Score', fontsize=14, fontweight='bold')
    ax.set_title('FBM Score Médio por Hora - Perfil Manipulado (30 dias)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h}h' for h in hours])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(-5, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fbm_variado_hourly_average.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico de médias salvo")
    
    # Gráfico 3: Componentes M, A, T separados (estilo referência)
    print("📊 Gerando gráfico de componentes M, A, T...")
    
    # Calcula médias dos componentes
    m_values = []
    a_values = []
    t_values = []
    
    for day_data in profile['days']:
        for hour_data in day_data['hours']:
            scores = calculate_fbm_scores(hour_data)
            m_values.append((hour_data['hour'], scores['motivation']))
            a_values.append((hour_data['hour'], scores['ability']))
            t_values.append((hour_data['hour'], scores['trigger']))
    
    # Agrupa por hora
    m_by_hour = {h: [] for h in range(24)}
    a_by_hour = {h: [] for h in range(24)}
    t_by_hour = {h: [] for h in range(24)}
    
    for hour, val in m_values:
        m_by_hour[hour].append(val)
    for hour, val in a_values:
        a_by_hour[hour].append(val)
    for hour, val in t_values:
        t_by_hour[hour].append(val)
    
    m_avg = [np.mean(m_by_hour[h]) for h in range(24)]
    a_avg = [np.mean(a_by_hour[h]) for h in range(24)]
    t_avg = [np.mean(t_by_hour[h]) for h in range(24)]
    fbm_avg = fbm_matrix.mean(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # M - Motivation
    axes[0, 0].plot(hours, m_avg, marker='o', linewidth=2, color='#FF6B6B', markersize=6)
    axes[0, 0].fill_between(hours, m_avg, alpha=0.3, color='#FF6B6B')
    axes[0, 0].set_title('Motivation (M) - Média por Hora', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Hora do Dia', fontsize=12)
    axes[0, 0].set_ylabel('Score (0-4)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.2, 4.5)
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # A - Ability
    axes[0, 1].plot(hours, a_avg, marker='s', linewidth=2, color='#4ECDC4', markersize=6)
    axes[0, 1].fill_between(hours, a_avg, alpha=0.3, color='#4ECDC4')
    axes[0, 1].set_title('Ability (A) - Média por Hora', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Hora do Dia', fontsize=12)
    axes[0, 1].set_ylabel('Score (0-4)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-0.2, 4.5)
    axes[0, 1].set_xticks(range(0, 24, 2))
    
    # T - Trigger
    axes[1, 0].plot(hours, t_avg, marker='^', linewidth=2, color='#95E1D3', markersize=6)
    axes[1, 0].fill_between(hours, t_avg, alpha=0.3, color='#95E1D3')
    axes[1, 0].set_title('Trigger (T) - Média por Hora', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Hora do Dia', fontsize=12)
    axes[1, 0].set_ylabel('Score (0-6)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-0.2, 6.5)
    axes[1, 0].set_xticks(range(0, 24, 2))
    
    # FBM Score (M × A × T)
    axes[1, 1].plot(hours, fbm_avg, marker='D', linewidth=2.5, color='#F38181', markersize=7)
    axes[1, 1].fill_between(hours, fbm_avg, alpha=0.3, color='#F38181')
    axes[1, 1].set_title('FBM Score (M × A × T) - Média por Hora', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Hora do Dia', fontsize=12)
    axes[1, 1].set_ylabel('Score (0-96)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(range(0, 24, 2))
    
    # Destaca períodos
    for ax in axes.flat:
        ax.axvspan(6, 11, alpha=0.15, color='#27AE60', label='Manhã (ALTO)')
        ax.axvspan(12, 17, alpha=0.15, color='#E74C3C', label='Tarde (BAIXO)')
        ax.axvspan(18, 23, alpha=0.15, color='#3498DB', label='Noite (ALTO)')
    
    axes[1, 1].legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Componentes FBM - Média de 30 Dias (Perfil Manipulado)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fbm_variado_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfico de componentes salvo")
    
    # Estatísticas por período
    print("\n📊 Estatísticas por Período:")
    
    periods = {
        'Madrugada (0-5h)': (0, 5),
        'Manhã (6-11h)': (6, 11),
        'Tarde (12-17h)': (12, 17),
        'Noite (18-23h)': (18, 23)
    }
    
    for period_name, (start, end) in periods.items():
        period_values = fbm_matrix[:, start:end+1].flatten()
        print(f"\n{period_name}:")
        print(f"  Média: {period_values.mean():.1f}")
        print(f"  Min: {period_values.min():.0f}")
        print(f"  Max: {period_values.max():.0f}")
        print(f"  % ≥60 (Alto): {(period_values >= 60).sum() / len(period_values) * 100:.1f}%")

def main():
    print("\n" + "="*80)
    print("📊 VISUALIZADOR FBM - Perfil Manipulado")
    print("="*80 + "\n")
    
    base_dir = Path(__file__).parent.parent
    profile_path = base_dir / 'data' / 'simulation' / 'NovoPerfil' / 'user_fbm_variado.json'
    output_dir = base_dir / 'data' / 'simulation' / 'NovoPerfil' / 'graficos'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Carregando perfil: {profile_path}")
    fbm_matrix, profile = load_and_analyze_profile(profile_path)
    print(f"✅ Perfil carregado: {fbm_matrix.shape[0]} dias\n")
    
    print(f"📁 Salvando gráficos em: {output_dir}\n")
    
    create_visualization(fbm_matrix, profile, output_dir)
    
    print("\n" + "="*80)
    print("✅ GRÁFICOS GERADOS COM SUCESSO!")
    print("="*80)
    print(f"\n📂 Localização: {output_dir}")
    print("\nGráficos gerados:")
    print("  1. fbm_variado_heatmap.png - Heatmap 30 dias x 24 horas")
    print("  2. fbm_variado_hourly_average.png - Média por hora com períodos")
    print("  3. fbm_variado_components.png - Componentes M, A, T separados\n")

if __name__ == "__main__":
    main()
