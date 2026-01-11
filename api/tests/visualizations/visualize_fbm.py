"""
Script para visualizar todos os FBM scores e seus componentes.
"""

import requests

API_URL = "http://localhost:8000"


def visualize_all_fbm_scores(user_id: str = "test_user_001"):
    """Mostra todos os FBM scores detalhados para cada hora."""
    
    response = requests.get(f"{API_URL}/treino/dados-treinamento/{user_id}")
    
    if response.status_code != 200:
        print(f"❌ Erro: {response.text}")
        return
    
    data = response.json()
    
    print("=" * 100)
    print(f"📊 FBM SCORES DETALHADOS - Usuário: {user_id}")
    print(f"   Total de amostras: {data['total_samples']}")
    print("=" * 100)
    print()
    print(f"{'Hora':^6} | {'M':^3} | {'A':^3} | {'T':^3} | {'FBM':^5} | {'Sleeping':^8} | {'Notif':^5} | {'Ação':^5} | {'Reward':^7}")
    print("-" * 100)
    
    for sample in data['data']:
        obs = sample['observation']
        hora = obs['hour']
        m = obs['motivation']
        a = obs['ability']
        t = obs['trigger']
        fbm = obs['fbm_score']
        sleeping = "💤" if obs['sleeping'] else ""
        notif = "📱" if sample['action'] == 1 else ""
        acao = "✅" if sample['action_performed'] else ""
        reward = sample['reward']
        
        print(f"  {hora:02d}h  |  {m}  |  {a}  |  {t}  |  {fbm:3d}  |   {sleeping:^6} |  {notif:^4} |  {acao:^4} |  {reward:>6.1f}")
    
    print("-" * 100)
    print()
    print("Legenda:")
    print("  M = Motivação (0-4), A = Habilidade (0-4), T = Gatilho (0-6)")
    print("  FBM = M × A × T (score final)")
    print("  💤 = Dormindo, 📱 = Notificação enviada, ✅ = Ação executada")
    print()
    
    # Análise
    fbm_scores = [s['observation']['fbm_score'] for s in data['data']]
    non_zero = [s for s in fbm_scores if s > 0]
    
    print("📈 ANÁLISE:")
    print(f"   - Horas com FBM > 0: {len(non_zero)}/24")
    print(f"   - Maior FBM: {max(fbm_scores)}")
    print(f"   - Média FBM (excluindo zeros): {sum(non_zero)/len(non_zero):.1f}" if non_zero else "   - Sem scores > 0")
    
    # Mostrar quais horas dorming
    sleeping_hours = [s['observation']['hour'] for s in data['data'] if s['observation']['sleeping']]
    print(f"   - Horas dormindo: {sleeping_hours}")


def visualize_day_details(user_id: str = "test_user_001"):
    """Mostra detalhes completos do histórico do usuário."""
    
    # Busca histórico direto do arquivo
    import json
    from pathlib import Path
    
    user_file = Path(f"data/users/{user_id}.json")
    
    if not user_file.exists():
        print(f"❌ Arquivo não encontrado: {user_file}")
        return
    
    with open(user_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    print("\n" + "=" * 100)
    print("📋 DETALHES COMPLETOS DO ÚLTIMO DIA")
    print("=" * 100)
    
    if not history["days"]:
        print("Nenhum dia registrado")
        return
    
    last_day = history["days"][-1]
    
    print(f"\nData: {last_day['date']}")
    print()
    print(f"{'Hora':^5} | {'Val':^3} | {'Act':^3} | {'Slp':^3} | {'Cog':^3} | {'ActT':^4} | {'Conf':^4} | {'Slpg':^4} | {'Aro':^3} | {'Loc':^5} | {'Mot':^5} | {'Per':^3} | {'WE':^3}")
    print("-" * 100)
    
    for hour in last_day["hours"]:
        h = hour["hour"]
        mf = hour["motivation_factors"]
        af = hour["ability_factors"]
        tf = hour["trigger_factors"]
        ctx = hour["context"]
        
        print(f"  {h:02d}  | {mf['valence']:^3} | {mf['last_activity_score']:^3} | {mf['hours_slept_last_night']:^3} | "
              f"{af['cognitive_load']:^3} | {af['activities_performed_today']:^4} | {af['confidence_score']:^4} | "
              f"{'Y' if tf['sleeping'] else 'N':^4} | {tf['arousal']:^3} | {tf['location']:^5} | {tf['motion_activity'][:5]:^5} | "
              f"{ctx['day_period']:^3} | {'Y' if ctx['is_weekend'] else 'N':^3}")
    
    print()
    print("Legenda:")
    print("  Val=Valence, Act=LastActivityScore, Slp=HoursSlept, Cog=CognitiveLoad")
    print("  ActT=ActivitiesToday, Conf=Confidence, Slpg=Sleeping, Aro=Arousal")
    print("  Loc=Location, Mot=Motion, Per=Period, WE=Weekend")


if __name__ == "__main__":
    visualize_all_fbm_scores("test_user_001")
    visualize_day_details("test_user_001")
