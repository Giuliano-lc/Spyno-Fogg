"""
Visualização da Evolução do Threshold Dinâmico.

Plota:
1. Evolução do threshold ao longo do tempo
2. Distribuição de VP/VN/FP/FN
3. Comparação threshold vs FBM scores
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Adiciona path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_threshold_history(user_id: str) -> list:
    """Carrega histórico de ajustes do threshold."""
    history_path = Path(f"data/thresholds/{user_id}_threshold_history.json")
    
    if not history_path.exists():
        print(f"❌ Histórico não encontrado: {history_path}")
        return []
    
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_threshold_state(user_id: str) -> dict:
    """Carrega estado atual do threshold."""
    state_path = Path(f"data/thresholds/{user_id}_threshold.json")
    
    if not state_path.exists():
        return {}
    
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_threshold_evolution(history: list, state: dict, user_id: str):
    """Plota evolução do threshold."""
    
    if not history:
        print("❌ Sem dados de histórico para plotar")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Evolução do Threshold Dinâmico - {user_id}", fontsize=16, fontweight='bold')
    
    # ===== GRÁFICO 1: Evolução do Threshold =====
    ax1 = axes[0, 0]
    
    thresholds = [h["new_threshold"] for h in history]
    events = range(len(thresholds))
    
    # Cores por tipo de feedback
    colors = []
    for h in history:
        if h["feedback_type"] == "VP":
            colors.append("green")
        elif h["feedback_type"] == "VN":
            colors.append("red")
        elif h["feedback_type"] == "FP":
            colors.append("orange")
        else:
            colors.append("gray")
    
    ax1.plot(events, thresholds, 'b-', linewidth=1, alpha=0.5)
    ax1.scatter(events, thresholds, c=colors, s=30, zorder=5)
    
    # Linha inicial
    initial = state.get("initial_threshold", 15)
    ax1.axhline(y=initial, color='purple', linestyle='--', linewidth=2, label=f'Inicial = {initial}')
    
    # Threshold final
    final = thresholds[-1] if thresholds else initial
    ax1.axhline(y=final, color='blue', linestyle='-', linewidth=2, alpha=0.5, label=f'Atual = {final:.1f}')
    
    ax1.set_xlabel('Evento', fontsize=12)
    ax1.set_ylabel('Threshold', fontsize=12)
    ax1.set_title('Evolução do Threshold ao Longo do Tempo', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Legenda de cores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='VP (mantém)'),
        Patch(facecolor='red', label='VN (aumenta)'),
        Patch(facecolor='orange', label='FP (diminui)'),
        Patch(facecolor='gray', label='FN (mantém)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # ===== GRÁFICO 2: Distribuição VP/VN/FP/FN =====
    ax2 = axes[0, 1]
    
    feedback_counts = defaultdict(int)
    for h in history:
        feedback_counts[h["feedback_type"]] += 1
    
    labels = ['VP\n(Notificou+Executou)', 'VN\n(Notificou+Não Exec)', 
              'FP\n(Não Notif+Executou)', 'FN\n(Não Notif+Não Exec)']
    values = [feedback_counts["VP"], feedback_counts["VN"], 
              feedback_counts["FP"], feedback_counts["FN"]]
    colors_pie = ['green', 'red', 'orange', 'gray']
    
    # Remove zeros para o pie chart
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors_pie) if v > 0]
    if non_zero:
        labels_nz, values_nz, colors_nz = zip(*non_zero)
        wedges, texts, autotexts = ax2.pie(
            values_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%',
            startangle=90, explode=[0.02]*len(values_nz)
        )
        ax2.set_title('Distribuição de Feedback', fontsize=14)
    else:
        ax2.text(0.5, 0.5, 'Sem dados', ha='center', va='center', fontsize=14)
        ax2.set_title('Distribuição de Feedback', fontsize=14)
    
    # ===== GRÁFICO 3: FBM Score vs Threshold =====
    ax3 = axes[1, 0]
    
    fbm_scores = [h["fbm_score"] for h in history]
    threshold_at_event = [h["old_threshold"] for h in history]
    
    ax3.scatter(range(len(fbm_scores)), fbm_scores, c='blue', alpha=0.6, label='FBM Score', s=20)
    ax3.plot(range(len(threshold_at_event)), threshold_at_event, 'r-', linewidth=2, label='Threshold')
    
    # Marca execuções
    exec_idx = [i for i, h in enumerate(history) if h["executed"]]
    exec_fbm = [history[i]["fbm_score"] for i in exec_idx]
    ax3.scatter(exec_idx, exec_fbm, c='green', s=80, marker='*', label='Executou', zorder=10)
    
    ax3.set_xlabel('Evento', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('FBM Score vs Threshold ao Longo do Tempo', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ===== GRÁFICO 4: Ajustes por Hora =====
    ax4 = axes[1, 1]
    
    hour_adjustments = defaultdict(lambda: {"up": 0, "down": 0, "same": 0})
    for h in history:
        hour = h["hour"]
        adj = h["adjustment"]
        if adj > 0:
            hour_adjustments[hour]["up"] += 1
        elif adj < 0:
            hour_adjustments[hour]["down"] += 1
        else:
            hour_adjustments[hour]["same"] += 1
    
    hours = range(24)
    ups = [hour_adjustments[h]["up"] for h in hours]
    downs = [hour_adjustments[h]["down"] for h in hours]
    sames = [hour_adjustments[h]["same"] for h in hours]
    
    x = np.arange(24)
    width = 0.25
    
    ax4.bar(x - width, ups, width, label='Aumentou (VN)', color='red', alpha=0.7)
    ax4.bar(x, downs, width, label='Diminuiu (FP)', color='orange', alpha=0.7)
    ax4.bar(x + width, sames, width, label='Manteve (VP/FN)', color='green', alpha=0.7)
    
    ax4.set_xlabel('Hora do Dia', fontsize=12)
    ax4.set_ylabel('Número de Ajustes', fontsize=12)
    ax4.set_title('Ajustes de Threshold por Hora', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{h}h' for h in hours], rotation=45, fontsize=8)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Destaca horas matinais
    for h in [6, 7, 8]:
        ax4.axvspan(h - 0.5, h + 0.5, alpha=0.1, color='yellow')
    
    plt.tight_layout()
    
    # Salva
    output_path = Path("data/results")
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f"threshold_evolution_{user_id}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Gráfico salvo em: {fig_path}")
    
    plt.show()


def print_statistics(history: list, state: dict):
    """Imprime estatísticas do threshold."""
    
    print("\n" + "=" * 60)
    print("📊 ESTATÍSTICAS DO THRESHOLD DINÂMICO")
    print("=" * 60)
    
    if not history:
        print("❌ Sem histórico disponível")
        return
    
    # Contagens
    feedback_counts = defaultdict(int)
    for h in history:
        feedback_counts[h["feedback_type"]] += 1
    
    total = len(history)
    vp = feedback_counts["VP"]
    vn = feedback_counts["VN"]
    fp = feedback_counts["FP"]
    fn = feedback_counts["FN"]
    
    print(f"\n📈 Total de eventos: {total}")
    print(f"\n   Tipo     | Contagem | Percentual | Significado")
    print(f"   {'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*30}")
    print(f"   VP       | {vp:^8} | {vp/total*100:^9.1f}% | Notificou + Executou (OK)")
    print(f"   VN       | {vn:^8} | {vn/total*100:^9.1f}% | Notificou + Não Exec (↑ th)")
    print(f"   FP       | {fp:^8} | {fp/total*100:^9.1f}% | Não Notif + Executou (↓ th)")
    print(f"   FN       | {fn:^8} | {fn/total*100:^9.1f}% | Não Notif + Não Exec (OK)")
    
    # Threshold
    initial = state.get("initial_threshold", 15)
    current = state.get("current_threshold", 15)
    
    print(f"\n🎯 Threshold:")
    print(f"   - Inicial: {initial}")
    print(f"   - Atual: {current:.1f}")
    print(f"   - Variação: {current - initial:+.1f}")
    
    # Métricas
    precision = vp / (vp + vn) if (vp + vn) > 0 else 0
    recall = vp / (vp + fp) if (vp + fp) > 0 else 0
    accuracy = (vp + fn) / total if total > 0 else 0
    
    print(f"\n📊 Métricas:")
    print(f"   - Precisão: {precision:.1%} (VP / (VP + VN))")
    print(f"   - Recall: {recall:.1%} (VP / (VP + FP))")
    print(f"   - Acurácia: {accuracy:.1%} ((VP + FN) / Total)")
    
    # FBM médio
    fbm_exec = [h["fbm_score"] for h in history if h["executed"]]
    fbm_no_exec = [h["fbm_score"] for h in history if not h["executed"]]
    
    print(f"\n📉 FBM Score:")
    if fbm_exec:
        print(f"   - Média quando executou: {np.mean(fbm_exec):.1f}")
    if fbm_no_exec:
        print(f"   - Média quando não executou: {np.mean(fbm_no_exec):.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("📊 VISUALIZAÇÃO DO THRESHOLD DINÂMICO")
    print("=" * 60)
    
    # Usuário padrão
    user_id = "user_matinal_30dias"
    
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    
    print(f"\n🔍 Carregando dados do usuário: {user_id}")
    
    # Carrega dados
    history = load_threshold_history(user_id)
    state = load_threshold_state(user_id)
    
    if not history:
        print("\n⚠️  Nenhum histórico de threshold encontrado.")
        print("   Execute o pipeline de treinamento com feedback para gerar dados.")
        print("\n   Comandos:")
        print("   1. python tests/training_pipeline.py")
        print("   2. python tests/plot_threshold_evolution.py")
        sys.exit(0)
    
    print(f"✅ {len(history)} eventos carregados")
    
    # Estatísticas
    print_statistics(history, state)
    
    # Plota
    print("\n📈 Gerando gráficos...")
    plot_threshold_evolution(history, state, user_id)
    
    print("\n✅ Análise completa!")
