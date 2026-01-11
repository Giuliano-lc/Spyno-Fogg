#!/usr/bin/env python3
"""
Visualização limpa e profissional da evolução do threshold dinâmico.
Focada em insights para o estudo acadêmico.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import seaborn as sns

# Configuração visual
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

API_URL = "http://localhost:8000"
USER_ID = "user_matinal_30dias"

def fetch_threshold_data():
    """Obtém dados do threshold via arquivo local ou API."""
    # Tenta ler arquivo local primeiro
    history_file = Path("data/thresholds/user_matinal_30dias_threshold_history.json")
    stats_file = Path("data/thresholds/user_matinal_30dias_threshold.json")
    
    history = []
    stats = {}
    
    # Lê histórico do arquivo local
    if history_file.exists():
        try:
            import json
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            pass
    
    # Lê stats do arquivo local
    if stats_file.exists():
        try:
            import json
            with open(stats_file, 'r', encoding='utf-8') as f:
                threshold_data = json.load(f)
                stats = {
                    'vp': threshold_data.get('vp_count', 0),
                    'vn': threshold_data.get('vn_count', 0),
                    'fp': threshold_data.get('fp_count', 0),
                    'fn': threshold_data.get('fn_count', 0)
                }
        except:
            pass
    
    # Se não encontrou dados locais, tenta API
    if not history:
        try:
            history_response = requests.get(f"{API_URL}/threshold/{USER_ID}/history")
            history = history_response.json() if history_response.status_code == 200 else []
            
            stats_response = requests.get(f"{API_URL}/threshold/{USER_ID}/stats")
            stats = stats_response.json() if stats_response.status_code == 200 else {}
        except:
            pass
    
    return history, stats

def create_academic_visualization():
    """Cria visualização acadêmica limpa."""
    history, stats = fetch_threshold_data()
    
    if not history:
        print("❌ Sem dados de threshold. Execute o pipeline primeiro.")
        return
    
    # Converte para DataFrame
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calcula dia baseado na HORA do feedback (0-23)
    # Agrupa por ciclos de horas - quando hora volta para 6, é novo dia
    df['hour'] = df['hour'].astype(int)
    
    # Detecta transições de dia (hora diminui ou passa por 0)
    df['day'] = 1
    current_day = 1
    prev_hour = df['hour'].iloc[0]
    
    days = [1]
    for i in range(1, len(df)):
        curr_hour = df['hour'].iloc[i]
        # Se hora atual < hora anterior, mudou de dia
        if curr_hour < prev_hour:
            current_day += 1
        days.append(current_day)
        prev_hour = curr_hour
    
    df['day'] = days
    
    # Configuração da figura
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Análise do Threshold Dinâmico - Fogg Behavior Model', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Evolução do Threshold (Principal)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Agrupa por dia e pega o último threshold de cada dia
    daily_threshold = df.groupby('day')['new_threshold'].last().reset_index()
    
    # Linha do threshold
    ax1.plot(daily_threshold['day'], daily_threshold['new_threshold'], 'b-', linewidth=2.5, 
             label='Threshold Dinâmico', marker='o', markersize=6)
    
    # Linha de referência (threshold inicial)
    ax1.axhline(y=15.0, color='gray', linestyle='--', alpha=0.7, 
                label='Threshold Inicial (15.0)')
    
    # Zonas de comportamento
    ax1.fill_between(daily_threshold['day'], 5, 15, alpha=0.1, color='green', 
                     label='Zona Permissiva (< 15)')
    ax1.fill_between(daily_threshold['day'], 15, 25, alpha=0.1, color='orange', 
                     label='Zona Moderada (15-25)')
    
    ax1.set_xlabel('Dia do Experimento', fontsize=12)
    ax1.set_ylabel('Threshold FBM', fontsize=12)
    ax1.set_title('A) Evolução do Threshold ao Longo do Tempo', fontsize=14, pad=20)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(daily_threshold['new_threshold']) + 5)
    ax1.set_xlim(0, max(daily_threshold['day']) + 1)
    
    # 2. Distribuição de Feedback (Pie Chart Limpo)
    ax2 = fig.add_subplot(gs[0, 2])
    
    feedback_data = [
        stats.get('vp', 0),
        stats.get('vn', 0), 
        stats.get('fp', 0),
        stats.get('fn', 0)
    ]
    labels = ['VP\n(Acerto)', 'VN\n(Falso Alarme)', 'FP\n(Oportunidade\nPerdida)', 'FN\n(Acerto)']
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    
    # Remove fatias muito pequenas para clareza
    threshold_min = 0.02 * sum(feedback_data)
    filtered_data = []
    filtered_labels = []
    filtered_colors = []
    
    for i, (data, label, color) in enumerate(zip(feedback_data, labels, colors)):
        if data > threshold_min:
            filtered_data.append(data)
            filtered_labels.append(f'{label}\n{data} ({data/sum(feedback_data)*100:.1f}%)')
            filtered_colors.append(color)
    
    wedges, texts, autotexts = ax2.pie(filtered_data, labels=filtered_labels, 
                                       colors=filtered_colors, autopct='',
                                       startangle=90, textprops={'fontsize': 9})
    
    ax2.set_title('B) Distribuição de Feedback', fontsize=14, pad=20)
    
    # 3. Padrão de Ajustes por Dia da Semana
    ax3 = fig.add_subplot(gs[1, 0])
    
    df['weekday'] = df['timestamp'].dt.day_name()
    weekday_adjustments = df.groupby('weekday')['adjustment'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ], fill_value=0)
    
    bars = ax3.bar(range(len(weekday_adjustments)), weekday_adjustments.values, 
                   color=['#3498db' if x >= 0 else '#e74c3c' for x in weekday_adjustments.values])
    
    ax3.set_xlabel('Dia da Semana', fontsize=12)
    ax3.set_ylabel('Ajuste Total', fontsize=12)
    ax3.set_title('C) Ajustes por Dia da Semana', fontsize=14, pad=20)
    ax3.set_xticks(range(len(weekday_adjustments)))
    ax3.set_xticklabels(['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linewidth=0.8)
    
    # 4. Histograma de Ajustes
    ax4 = fig.add_subplot(gs[1, 1])
    
    adjustments = df['adjustment'].values
    ax4.hist(adjustments, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Sem Ajuste')
    ax4.axvline(x=np.mean(adjustments), color='green', linestyle='-', 
                label=f'Média: {np.mean(adjustments):.1f}')
    
    ax4.set_xlabel('Magnitude do Ajuste', fontsize=12)
    ax4.set_ylabel('Frequência', fontsize=12)
    ax4.set_title('D) Distribuição dos Ajustes', fontsize=14, pad=20)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Métricas de Performance
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Calcula métricas
    total_feedback = sum(feedback_data)
    vp, vn, fp, fn = feedback_data
    
    precision = vp / (vp + vn) if (vp + vn) > 0 else 0
    recall = vp / (vp + fp) if (vp + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    threshold_change = df['new_threshold'].iloc[-1] - df['new_threshold'].iloc[0]
    avg_adjustment = np.mean(np.abs(df['adjustment']))
    
    metrics_text = f"""
E) Metricas do Sistema

Performance:
• Precisao: {precision:.1%}
• Recall: {recall:.1%}
• F1-Score: {f1_score:.1%}

Threshold:
• Inicial: {df['new_threshold'].iloc[0]:.1f}
• Final: {df['new_threshold'].iloc[-1]:.1f}
• Variacao: {threshold_change:+.1f}

Adaptacao:
• Ajustes: {len(df)} eventos
• Media |ajuste|: {avg_adjustment:.1f}
• Total feedback: {total_feedback}
    """
    
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Salva figura
    output_dir = Path("data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "threshold_analysis_clean.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "threshold_analysis_clean.pdf", 
                bbox_inches='tight', facecolor='white')
    
    print(f"✅ Visualização salva em: {output_dir}")
    plt.show()

def create_summary_report():
    """Cria relatório resumido para o estudo."""
    history, stats = fetch_threshold_data()
    
    if not history:
        return
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Análise estatística
    threshold_initial = df['new_threshold'].iloc[0]
    threshold_final = df['new_threshold'].iloc[-1]
    threshold_range = df['new_threshold'].max() - df['new_threshold'].min()
    
    vp, vn, fp, fn = stats.get('vp', 0), stats.get('vn', 0), stats.get('fp', 0), stats.get('fn', 0)
    total = vp + vn + fp + fn
    
    report = f"""
📋 RELATÓRIO DE ANÁLISE DO THRESHOLD DINÂMICO

🎯 Objetivo: Avaliar a adaptação automática do threshold FBM baseado em feedback do usuário

📊 Resultados Principais:
• Threshold adaptou de {threshold_initial:.1f} → {threshold_final:.1f} ({threshold_final-threshold_initial:+.1f})
• Amplitude de variação: {threshold_range:.1f} pontos
• Total de ajustes: {len(df)} eventos

🔍 Distribuição de Feedback:
• VP (Verdadeiro Positivo): {vp} ({vp/total*100:.1f}%) - Sistema acertou ao notificar
• VN (Verdadeiro Negativo): {vn} ({vn/total*100:.1f}%) - Sistema errou ao notificar  
• FP (Falso Positivo): {fp} ({fp/total*100:.1f}%) - Sistema perdeu oportunidade
• FN (Falso Negativo): {fn} ({fn/total*100:.1f}%) - Sistema acertou ao não notificar

📈 Interpretação:
{'• Sistema ficou MAIS PERMISSIVO (threshold diminuiu)' if threshold_final < threshold_initial else '• Sistema ficou MAIS RESTRITIVO (threshold aumentou)'}
{'• Muitas oportunidades perdidas (FP > VN) levaram à redução do threshold' if fp > vn else '• Muitos falsos alarmes (VN > FP) levaram ao aumento do threshold'}

💡 Conclusão:
O threshold dinâmico demonstrou capacidade de adaptação ao comportamento do usuário,
{'reduzindo a exigência para capturar mais oportunidades de exercício.' if threshold_final < threshold_initial else 'aumentando a exigência para reduzir notificações desnecessárias.'}
    """
    
    print(report)
    
    # Salva relatório
    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "threshold_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n📄 Relatório salvo em: {output_dir}/threshold_analysis_report.txt")

if __name__ == "__main__":
    print("🎨 Gerando visualização acadêmica do threshold dinâmico...")
    create_academic_visualization()
    
    print("\n📊 Gerando relatório de análise...")
    create_summary_report()
