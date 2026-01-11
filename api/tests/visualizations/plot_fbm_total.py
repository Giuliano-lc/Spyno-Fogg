"""
Visualização do FBM Score Total (M×A×T) com threshold correto.

Baseado no paper: action_threshold = 20-25
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Adiciona path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Threshold do paper (linha 191)
ACTION_THRESHOLD = 20


def load_data_from_file(file_path: str) -> list:
    """Carrega dados de um arquivo JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        days = json.load(f)
    
    all_data = []
    for day in days:
        for hour_data in day["hours"]:
            mf = hour_data["motivation_factors"]
            af = hour_data["ability_factors"]
            tf = hour_data["trigger_factors"]
            ctx = hour_data["context"]
            fb = hour_data["feedback"]
            
            # Motivação (max 4) - conforme paper linha 101
            m = (1 if mf["valence"] == 1 else 0) + \
                1 + \
                (1 if mf["last_activity_score"] == 1 else 0) + \
                (1 if mf["hours_slept_last_night"] >= 7 else 0)
            
            # Habilidade (max 3) - conforme paper linha 118
            a = (1 if af["cognitive_load"] == 0 else 0) + \
                (1 if af["confidence_score"] >= 4 else 0) + \
                (1 if af["activities_performed_today"] <= 2 else 0)
            
            # Gatilho (max 5) - conforme paper linha 128
            if tf["sleeping"]:
                t = 0
            else:
                t = (1 if tf["arousal"] == 1 else 0) + \
                    (1 if ctx["is_weekend"] else 0) + \
                    (1 if 10 <= hour_data["hour"] <= 14 else 0) + \
                    (1 if tf["location"] == "home" else 0) + \
                    (1 if tf["motion_activity"] == "stationary" else 0)
            
            fbm_total = m * a * t
            
            all_data.append({
                "hour": hour_data["hour"],
                "date": day["date"],
                "motivation": m,
                "ability": a,
                "trigger": t,
                "m_x_a": m * a,
                "fbm_total": fbm_total,
                "sleeping": tf["sleeping"],
                "action_performed": fb["action_performed"],
                "notification_sent": fb["notification_sent"]
            })
    
    return all_data


def plot_fbm_total_threshold(data: list):
    """
    Plota gráfico do FBM Score Total com threshold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== GRÁFICO 1: FBM Total - Executou vs Não Executou =====
    ax1 = axes[0, 0]
    
    # Separa dados
    awake = [d for d in data if not d["sleeping"]]
    exec_fbm = [d["fbm_total"] for d in awake if d["action_performed"]]
    no_exec_fbm = [d["fbm_total"] for d in awake if not d["action_performed"]]
    
    # Histograma
    bins = range(0, 65, 5)
    ax1.hist(no_exec_fbm, bins=bins, alpha=0.5, label=f'Não executou (n={len(no_exec_fbm)})', color='red', edgecolor='darkred')
    ax1.hist(exec_fbm, bins=bins, alpha=0.7, label=f'Executou (n={len(exec_fbm)})', color='green', edgecolor='darkgreen')
    
    # Linha de threshold
    ax1.axvline(x=ACTION_THRESHOLD, color='blue', linestyle='--', linewidth=3, label=f'Threshold = {ACTION_THRESHOLD}')
    
    ax1.set_xlabel('FBM Score Total (M × A × T)', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.set_title('Distribuição do FBM Score Total\n(threshold do paper = 20)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Anotações
    exec_above = sum(1 for d in awake if d["action_performed"] and d["fbm_total"] >= ACTION_THRESHOLD)
    exec_below = sum(1 for d in awake if d["action_performed"] and d["fbm_total"] < ACTION_THRESHOLD)
    no_exec_above = sum(1 for d in awake if not d["action_performed"] and d["fbm_total"] >= ACTION_THRESHOLD)
    no_exec_below = sum(1 for d in awake if not d["action_performed"] and d["fbm_total"] < ACTION_THRESHOLD)
    
    ax1.annotate(f'Execuções\nacima: {exec_above}\nabaixo: {exec_below}', 
                 xy=(45, ax1.get_ylim()[1]*0.7), fontsize=10, color='darkgreen',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax1.annotate(f'Não-exec\nacima: {no_exec_above}\nabaixo: {no_exec_below}', 
                 xy=(5, ax1.get_ylim()[1]*0.7), fontsize=10, color='darkred',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # ===== GRÁFICO 2: Boxplot por componente =====
    ax2 = axes[0, 1]
    
    exec_data = [d for d in awake if d["action_performed"]]
    no_exec_data = [d for d in awake if not d["action_performed"]]
    
    # Dados para boxplot
    labels = ['M (exec)', 'M (não)', 'A (exec)', 'A (não)', 'T (exec)', 'T (não)']
    data_box = [
        [d["motivation"] for d in exec_data],
        [d["motivation"] for d in no_exec_data],
        [d["ability"] for d in exec_data],
        [d["ability"] for d in no_exec_data],
        [d["trigger"] for d in exec_data],
        [d["trigger"] for d in no_exec_data]
    ]
    
    colors = ['green', 'red', 'green', 'red', 'green', 'red']
    bp = ax2.boxplot(data_box, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Distribuição dos Componentes FBM\n(Executou vs Não Executou)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Linhas de max
    ax2.axhline(y=4, color='blue', linestyle=':', alpha=0.5, label='Max M=4')
    ax2.axhline(y=3, color='orange', linestyle=':', alpha=0.5, label='Max A=3')
    ax2.axhline(y=5, color='purple', linestyle=':', alpha=0.5, label='Max T=5')
    ax2.legend(loc='upper right', fontsize=9)
    
    # ===== GRÁFICO 3: FBM Score por Hora =====
    ax3 = axes[1, 0]
    
    hours_data = defaultdict(lambda: {"exec": [], "no_exec": []})
    for d in awake:
        if d["action_performed"]:
            hours_data[d["hour"]]["exec"].append(d["fbm_total"])
        else:
            hours_data[d["hour"]]["no_exec"].append(d["fbm_total"])
    
    hours = range(24)
    exec_means = [np.mean(hours_data[h]["exec"]) if hours_data[h]["exec"] else 0 for h in hours]
    no_exec_means = [np.mean(hours_data[h]["no_exec"]) if hours_data[h]["no_exec"] else 0 for h in hours]
    exec_counts = [len(hours_data[h]["exec"]) for h in hours]
    
    x = np.arange(24)
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, exec_means, width, label='Executou', color='green', alpha=0.7)
    bars2 = ax3.bar(x + width/2, no_exec_means, width, label='Não executou', color='red', alpha=0.4)
    
    # Contagem de execuções
    for i, count in enumerate(exec_counts):
        if count > 0:
            ax3.annotate(f'{count}', xy=(i - width/2, exec_means[i] + 1), 
                        ha='center', va='bottom', fontsize=8, color='darkgreen')
    
    ax3.axhline(y=ACTION_THRESHOLD, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {ACTION_THRESHOLD}')
    
    ax3.set_xlabel('Hora do Dia', fontsize=12)
    ax3.set_ylabel('FBM Score Médio (M × A × T)', fontsize=12)
    ax3.set_title('FBM Score por Hora\n(números = qtd execuções)', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{h}h' for h in hours], rotation=45, fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Destaca horas matinais
    for h in [6, 7, 8]:
        ax3.axvspan(h - 0.5, h + 0.5, alpha=0.15, color='yellow')
    
    # ===== GRÁFICO 4: Análise de Threshold =====
    ax4 = axes[1, 1]
    
    # Testa diferentes thresholds
    thresholds = range(5, 50, 5)
    accuracies = []
    precisions = []
    recalls = []
    
    for th in thresholds:
        tp = sum(1 for d in awake if d["action_performed"] and d["fbm_total"] >= th)
        tn = sum(1 for d in awake if not d["action_performed"] and d["fbm_total"] < th)
        fp = sum(1 for d in awake if not d["action_performed"] and d["fbm_total"] >= th)
        fn = sum(1 for d in awake if d["action_performed"] and d["fbm_total"] < th)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        accuracies.append(accuracy * 100)
        precisions.append(precision * 100)
        recalls.append(recall * 100)
    
    ax4.plot(thresholds, accuracies, 'b-o', label='Acurácia', linewidth=2)
    ax4.plot(thresholds, precisions, 'g-s', label='Precisão', linewidth=2)
    ax4.plot(thresholds, recalls, 'r-^', label='Recall', linewidth=2)
    
    # Marca threshold do paper
    ax4.axvline(x=ACTION_THRESHOLD, color='purple', linestyle='--', linewidth=2, label=f'Threshold Paper = {ACTION_THRESHOLD}')
    
    # Encontra melhor threshold
    best_th_idx = np.argmax(accuracies)
    best_th = list(thresholds)[best_th_idx]
    ax4.axvline(x=best_th, color='orange', linestyle=':', linewidth=2, label=f'Melhor Threshold = {best_th}')
    
    ax4.set_xlabel('Threshold (FBM Score)', fontsize=12)
    ax4.set_ylabel('Porcentagem (%)', fontsize=12)
    ax4.set_title('Métricas por Threshold\n(encontrando o threshold ideal)', fontsize=14)
    ax4.legend(loc='center right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 55)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # Salva
    output_path = Path("data/results")
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "fbm_total_threshold.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Gráfico salvo em: {fig_path}")
    
    plt.show()
    
    return best_th


def print_analysis(data: list, best_threshold: int):
    """Imprime análise detalhada."""
    
    awake = [d for d in data if not d["sleeping"]]
    exec_data = [d for d in awake if d["action_performed"]]
    no_exec_data = [d for d in awake if not d["action_performed"]]
    
    print("\n" + "=" * 70)
    print("📊 ANÁLISE DO FBM - COMPARAÇÃO COM PAPER")
    print("=" * 70)
    
    print(f"\n📈 Scores Médios:")
    print(f"\n   {'':^15} | {'Executou':^12} | {'Não Executou':^12} | {'Diferença':^10}")
    print(f"   {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    
    m_exec = np.mean([d["motivation"] for d in exec_data])
    m_no = np.mean([d["motivation"] for d in no_exec_data])
    print(f"   {'Motivação (M)':^15} | {m_exec:^12.2f} | {m_no:^12.2f} | {m_exec-m_no:^+10.2f}")
    
    a_exec = np.mean([d["ability"] for d in exec_data])
    a_no = np.mean([d["ability"] for d in no_exec_data])
    print(f"   {'Habilidade (A)':^15} | {a_exec:^12.2f} | {a_no:^12.2f} | {a_exec-a_no:^+10.2f}")
    
    t_exec = np.mean([d["trigger"] for d in exec_data])
    t_no = np.mean([d["trigger"] for d in no_exec_data])
    print(f"   {'Gatilho (T)':^15} | {t_exec:^12.2f} | {t_no:^12.2f} | {t_exec-t_no:^+10.2f}")
    
    fbm_exec = np.mean([d["fbm_total"] for d in exec_data])
    fbm_no = np.mean([d["fbm_total"] for d in no_exec_data])
    print(f"   {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    print(f"   {'FBM Total':^15} | {fbm_exec:^12.1f} | {fbm_no:^12.1f} | {fbm_exec-fbm_no:^+10.1f}")
    
    print(f"\n🎯 Thresholds:")
    print(f"   - Paper: {ACTION_THRESHOLD}")
    print(f"   - Ideal para nossos dados: {best_threshold}")
    
    # Análise de qual componente mais diferencia
    print(f"\n📊 Qual componente mais diferencia execução de não-execução?")
    diffs = [
        ("Motivação", m_exec - m_no, 4),
        ("Habilidade", a_exec - a_no, 3),
        ("Gatilho", t_exec - t_no, 5)
    ]
    
    # Normaliza pela escala
    for name, diff, max_val in sorted(diffs, key=lambda x: x[1]/x[2], reverse=True):
        pct = (diff / max_val) * 100
        bar = "█" * int(abs(pct) / 5)
        print(f"   - {name}: {diff:+.2f} ({pct:+.1f}% do max) {bar}")
    
    print(f"\n💡 CONCLUSÃO:")
    max_diff = max(diffs, key=lambda x: x[1]/x[2])
    print(f"   O componente que mais diferencia é: **{max_diff[0]}**")
    print(f"   Diferença normalizada: {(max_diff[1]/max_diff[2])*100:.1f}% do máximo possível")


if __name__ == "__main__":
    print("=" * 70)
    print("📊 ANÁLISE FBM TOTAL - Threshold do Paper")
    print("=" * 70)
    
    # Carrega dados
    data_file = Path("data/synthetic/user_matinal_30dias_all_days.json")
    
    if not data_file.exists():
        print(f"❌ Arquivo não encontrado: {data_file}")
        print("   Execute primeiro: python tests/generate_monthly_data.py")
        sys.exit(1)
    
    data = load_data_from_file(str(data_file))
    print(f"✅ {len(data)} amostras carregadas")
    
    # Plota
    best_th = plot_fbm_total_threshold(data)
    
    # Análise
    print_analysis(data, best_th)
