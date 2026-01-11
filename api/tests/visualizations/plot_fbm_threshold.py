"""
Visualização do FBM: Gráfico Motivação × Habilidade com threshold de execução.

Plota todos os pontos (M, A) do histórico do usuário, colorindo por:
- Verde: Executou a ação
- Vermelho: Não executou

Traça uma linha de threshold estimada onde ocorre a transição.
"""

import sys
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Adiciona path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))


API_URL = "http://localhost:8000"


def load_data_from_api(user_id: str) -> List[Dict]:
    """Carrega dados de treinamento da API."""
    response = requests.get(f"{API_URL}/treino/dados-treinamento/{user_id}")
    
    if response.status_code == 200:
        return response.json()["data"]
    else:
        print(f"❌ Erro ao carregar dados: {response.text}")
        return []


def load_data_from_file(file_path: str) -> List[Dict]:
    """Carrega dados de um arquivo JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        days = json.load(f)
    
    # Extrai dados de cada hora
    all_data = []
    for day in days:
        for hour_data in day["hours"]:
            mf = hour_data["motivation_factors"]
            af = hour_data["ability_factors"]
            tf = hour_data["trigger_factors"]
            ctx = hour_data["context"]
            fb = hour_data["feedback"]
            
            # Calcula scores FBM
            m = (1 if mf["valence"] == 1 else 0) + 1 + \
                (1 if mf["last_activity_score"] == 1 else 0) + \
                (1 if mf["hours_slept_last_night"] >= 7 else 0)
            
            a = (1 if af["cognitive_load"] == 0 else 0) + \
                (1 if af["activities_performed_today"] <= 1 else 0) + \
                (1 if af["time_since_last_activity_hours"] >= 1 else 0) + \
                (1 if af["confidence_score"] >= 4 else 0)
            
            if tf["sleeping"]:
                t = 0
            else:
                t = 1 + (1 if tf["arousal"] == 1 else 0) + \
                    (1 if tf["location"] == "home" else 0) + \
                    (1 if tf["motion_activity"] == "stationary" else 0) + \
                    (1 if ctx["day_period"] == 1 else 0) + \
                    (1 if ctx["is_weekend"] else 0)
            
            all_data.append({
                "hour": hour_data["hour"],
                "date": day["date"],
                "motivation": m,
                "ability": a,
                "trigger": t,
                "fbm_score": m * a * t,
                "sleeping": tf["sleeping"],
                "action_performed": fb["action_performed"],
                "notification_sent": fb["notification_sent"]
            })
    
    return all_data


def calculate_threshold_line(data: List[Dict]) -> Tuple[float, float]:
    """
    Calcula a linha de threshold M×A que separa execuções de não-execuções.
    
    Retorna coeficientes (slope, intercept) para a linha A = slope * M + intercept
    ou o valor de M×A threshold.
    """
    # Pontos com execução
    exec_points = [(d["motivation"], d["ability"]) for d in data if d["action_performed"] and not d["sleeping"]]
    
    # Pontos sem execução (acordado)
    no_exec_points = [(d["motivation"], d["ability"]) for d in data if not d["action_performed"] and not d["sleeping"]]
    
    if not exec_points:
        return 2.0, 0  # Threshold padrão
    
    # Calcula M×A para cada ponto
    exec_products = [m * a for m, a in exec_points]
    no_exec_products = [m * a for m, a in no_exec_points]
    
    # Threshold = média entre o mínimo das execuções e máximo das não-execuções
    min_exec = min(exec_products) if exec_products else 4
    max_no_exec = max(no_exec_products) if no_exec_products else 2
    
    threshold = (min_exec + max_no_exec) / 2
    
    return threshold, None


def plot_fbm_scatter(data: List[Dict], title: str = "FBM: Motivação × Habilidade"):
    """
    Plota gráfico de dispersão M vs A com cores por execução.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ===== GRÁFICO 1: Motivação vs Habilidade =====
    ax1 = axes[0]
    
    # Separa pontos
    exec_m = [d["motivation"] for d in data if d["action_performed"] and not d["sleeping"]]
    exec_a = [d["ability"] for d in data if d["action_performed"] and not d["sleeping"]]
    
    no_exec_m = [d["motivation"] for d in data if not d["action_performed"] and not d["sleeping"]]
    no_exec_a = [d["ability"] for d in data if not d["action_performed"] and not d["sleeping"]]
    
    sleep_m = [d["motivation"] for d in data if d["sleeping"]]
    sleep_a = [d["ability"] for d in data if d["sleeping"]]
    
    # Plota pontos
    ax1.scatter(no_exec_m, no_exec_a, c='red', alpha=0.4, s=50, label=f'Não executou (n={len(no_exec_m)})', marker='x')
    ax1.scatter(exec_m, exec_a, c='green', alpha=0.7, s=80, label=f'Executou (n={len(exec_m)})', marker='o')
    ax1.scatter(sleep_m, sleep_a, c='gray', alpha=0.2, s=30, label=f'Dormindo (n={len(sleep_m)})', marker='.')
    
    # Calcula e plota linha de threshold (M × A = threshold)
    threshold, _ = calculate_threshold_line(data)
    
    # Plota hipérbole M × A = threshold
    m_range = np.linspace(0.5, 4.5, 100)
    a_threshold = threshold / m_range
    a_threshold = np.clip(a_threshold, 0, 5)
    
    ax1.plot(m_range, a_threshold, 'b--', linewidth=2, label=f'Threshold: M×A = {threshold:.1f}')
    ax1.fill_between(m_range, a_threshold, 5, alpha=0.1, color='green')
    ax1.fill_between(m_range, 0, a_threshold, alpha=0.1, color='red')
    
    ax1.set_xlabel('Motivação (M)', fontsize=12)
    ax1.set_ylabel('Habilidade (A)', fontsize=12)
    ax1.set_title('Motivação vs Habilidade\n(linha = threshold de execução)', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # ===== GRÁFICO 2: FBM Score por Hora =====
    ax2 = axes[1]
    
    # Agrupa por hora
    hours_exec = defaultdict(list)
    hours_no_exec = defaultdict(list)
    
    for d in data:
        if d["sleeping"]:
            continue
        if d["action_performed"]:
            hours_exec[d["hour"]].append(d["fbm_score"])
        else:
            hours_no_exec[d["hour"]].append(d["fbm_score"])
    
    # Calcula médias por hora
    hours = range(24)
    exec_means = [np.mean(hours_exec[h]) if hours_exec[h] else 0 for h in hours]
    no_exec_means = [np.mean(hours_no_exec[h]) if hours_no_exec[h] else 0 for h in hours]
    exec_counts = [len(hours_exec[h]) for h in hours]
    
    # Plota barras
    x = np.arange(24)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, exec_means, width, label='Executou (média FBM)', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, no_exec_means, width, label='Não executou (média FBM)', color='red', alpha=0.4)
    
    # Adiciona contagem de execuções em cima das barras
    for i, count in enumerate(exec_counts):
        if count > 0:
            ax2.annotate(f'{count}', xy=(i - width/2, exec_means[i]), 
                        ha='center', va='bottom', fontsize=8, color='darkgreen')
    
    # Linha de threshold
    ax2.axhline(y=threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.1f}')
    
    ax2.set_xlabel('Hora do Dia', fontsize=12)
    ax2.set_ylabel('FBM Score (M × A × T)', fontsize=12)
    ax2.set_title('FBM Score Médio por Hora\n(números = qtd execuções)', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{h}h' for h in hours], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Destaca horas matinais (6-8h)
    for h in [6, 7, 8]:
        ax2.axvspan(h - 0.5, h + 0.5, alpha=0.1, color='yellow')
    
    plt.tight_layout()
    
    # Salva figura
    output_path = Path("data/results")
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "fbm_threshold_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Gráfico salvo em: {fig_path}")
    
    plt.show()
    
    return threshold


def plot_daily_evolution(data: List[Dict]):
    """
    Plota evolução diária dos scores FBM.
    """
    # Agrupa por data
    daily_data = defaultdict(lambda: {"m": [], "a": [], "t": [], "fbm": [], "exec": 0})
    
    for d in data:
        if d["sleeping"]:
            continue
        date = d["date"]
        daily_data[date]["m"].append(d["motivation"])
        daily_data[date]["a"].append(d["ability"])
        daily_data[date]["t"].append(d["trigger"])
        daily_data[date]["fbm"].append(d["fbm_score"])
        if d["action_performed"]:
            daily_data[date]["exec"] += 1
    
    # Ordena por data
    dates = sorted(daily_data.keys())
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Médias diárias de M, A, T
    ax1 = axes[0]
    
    m_means = [np.mean(daily_data[d]["m"]) for d in dates]
    a_means = [np.mean(daily_data[d]["a"]) for d in dates]
    t_means = [np.mean(daily_data[d]["t"]) for d in dates]
    
    x = range(len(dates))
    ax1.plot(x, m_means, 'b-o', label='Motivação (M)', linewidth=2, markersize=5)
    ax1.plot(x, a_means, 'g-s', label='Habilidade (A)', linewidth=2, markersize=5)
    ax1.plot(x, t_means, 'r-^', label='Gatilho (T)', linewidth=2, markersize=5)
    
    ax1.set_xlabel('Dia', fontsize=12)
    ax1.set_ylabel('Score Médio', fontsize=12)
    ax1.set_title('Evolução Diária dos Componentes FBM', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([dates[i][5:] for i in range(0, len(dates), 2)], rotation=45)
    
    # Gráfico 2: FBM total e execuções
    ax2 = axes[1]
    
    fbm_means = [np.mean(daily_data[d]["fbm"]) for d in dates]
    exec_counts = [daily_data[d]["exec"] for d in dates]
    
    ax2_twin = ax2.twinx()
    
    line1, = ax2.plot(x, fbm_means, 'purple', linewidth=2, label='FBM Médio')
    ax2.fill_between(x, fbm_means, alpha=0.2, color='purple')
    
    bars = ax2_twin.bar(x, exec_counts, alpha=0.5, color='green', label='Execuções')
    
    ax2.set_xlabel('Dia', fontsize=12)
    ax2.set_ylabel('FBM Score Médio', fontsize=12, color='purple')
    ax2_twin.set_ylabel('Nº Execuções', fontsize=12, color='green')
    ax2.set_title('FBM Score vs Execuções por Dia', fontsize=14)
    
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([dates[i][5:] for i in range(0, len(dates), 2)], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Legenda combinada
    lines = [line1, plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.5)]
    labels = ['FBM Médio', 'Execuções']
    ax2.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    # Salva figura
    output_path = Path("data/results")
    fig_path = output_path / "fbm_daily_evolution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"💾 Gráfico salvo em: {fig_path}")
    
    plt.show()


def print_summary(data: List[Dict], threshold: float):
    """Imprime resumo estatístico."""
    
    print("\n" + "=" * 60)
    print("📊 RESUMO ESTATÍSTICO DO FBM")
    print("=" * 60)
    
    # Filtra dados de quando estava acordado
    awake_data = [d for d in data if not d["sleeping"]]
    exec_data = [d for d in awake_data if d["action_performed"]]
    no_exec_data = [d for d in awake_data if not d["action_performed"]]
    
    print(f"\n📈 Total de amostras: {len(data)}")
    print(f"   - Acordado: {len(awake_data)}")
    print(f"   - Dormindo: {len(data) - len(awake_data)}")
    print(f"   - Com execução: {len(exec_data)}")
    print(f"   - Sem execução: {len(no_exec_data)}")
    
    if exec_data:
        print(f"\n✅ Quando EXECUTOU:")
        print(f"   - Motivação média: {np.mean([d['motivation'] for d in exec_data]):.2f}")
        print(f"   - Habilidade média: {np.mean([d['ability'] for d in exec_data]):.2f}")
        print(f"   - Gatilho médio: {np.mean([d['trigger'] for d in exec_data]):.2f}")
        print(f"   - FBM médio: {np.mean([d['fbm_score'] for d in exec_data]):.1f}")
        print(f"   - M×A médio: {np.mean([d['motivation']*d['ability'] for d in exec_data]):.2f}")
    
    if no_exec_data:
        print(f"\n❌ Quando NÃO EXECUTOU:")
        print(f"   - Motivação média: {np.mean([d['motivation'] for d in no_exec_data]):.2f}")
        print(f"   - Habilidade média: {np.mean([d['ability'] for d in no_exec_data]):.2f}")
        print(f"   - Gatilho médio: {np.mean([d['trigger'] for d in no_exec_data]):.2f}")
        print(f"   - FBM médio: {np.mean([d['fbm_score'] for d in no_exec_data]):.1f}")
        print(f"   - M×A médio: {np.mean([d['motivation']*d['ability'] for d in no_exec_data]):.2f}")
    
    print(f"\n🎯 Threshold calculado: M×A = {threshold:.2f}")
    print(f"   (Acima deste valor, maior chance de execução)")
    
    # Verifica eficácia do threshold
    correct_above = sum(1 for d in exec_data if d['motivation'] * d['ability'] >= threshold)
    correct_below = sum(1 for d in no_exec_data if d['motivation'] * d['ability'] < threshold)
    total = len(exec_data) + len(no_exec_data)
    
    accuracy = (correct_above + correct_below) / total * 100 if total > 0 else 0
    print(f"\n📊 Eficácia do threshold:")
    print(f"   - Acurácia: {accuracy:.1f}%")
    print(f"   - Execuções acima do threshold: {correct_above}/{len(exec_data)}")
    print(f"   - Não-execuções abaixo do threshold: {correct_below}/{len(no_exec_data)}")


if __name__ == "__main__":
    print("=" * 60)
    print("📊 ANÁLISE FBM - Threshold de Execução")
    print("=" * 60)
    
    # Tenta carregar da API primeiro
    user_id = "user_matinal_30dias"
    
    print(f"\n🔍 Carregando dados do usuário: {user_id}")
    
    # Tenta carregar do arquivo local
    data_file = Path("data/synthetic/user_matinal_30dias_all_days.json")
    
    if data_file.exists():
        print(f"   Fonte: {data_file}")
        data = load_data_from_file(str(data_file))
    else:
        print("   Fonte: API")
        data = load_data_from_api(user_id)
    
    if not data:
        print("❌ Nenhum dado encontrado!")
        sys.exit(1)
    
    print(f"✅ {len(data)} amostras carregadas")
    
    # Plota gráficos
    threshold = plot_fbm_scatter(data, title=f"FBM Analysis - {user_id}")
    
    # Imprime resumo
    print_summary(data, threshold)
    
    # Plota evolução diária
    print("\n📈 Gerando gráfico de evolução diária...")
    plot_daily_evolution(data)
    
    print("\n✅ Análise completa!")
