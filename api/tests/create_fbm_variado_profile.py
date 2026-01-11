"""
Cria perfil com FBM manipulado:
- Manhã (6-11h): FBM ALTO (60-96)
- Tarde (12-17h): FBM BAIXO (10-30)
- Noite (18-23h): FBM ALTO (60-96)
- Madrugada (0-5h): FBM BAIXO (0-10, dormindo)
"""

import json
from pathlib import Path
from copy import deepcopy

def manipulate_fbm_factors(hour, original_factors):
    """Ajusta fatores para gerar FBM alto/baixo conforme período."""
    
    # Copia fatores originais
    motivation = deepcopy(original_factors['motivation_factors'])
    ability = deepcopy(original_factors['ability_factors'])
    trigger = deepcopy(original_factors['trigger_factors'])
    
    # MANHÃ (6-11h): FBM ALTO (M=4, A=4, T=6)
    if 6 <= hour <= 11:
        motivation['valence'] = 4  # Alta motivação
        motivation['last_activity_score'] = 4
        
        ability['cognitive_load'] = 1  # Baixa carga cognitiva
        ability['confidence_score'] = 9
        
        trigger['sleeping'] = False
        trigger['arousal'] = 5  # Alto arousal
        trigger['location'] = 'home'
        trigger['motion_activity'] = 'walking'  # Corrigido: 'active' não existe
        # M=4, A=4, T=6 → FBM=96
    
    # TARDE (12-17h): FBM BAIXO (M=1, A=1, T=2)
    elif 12 <= hour <= 17:
        motivation['valence'] = 1  # Baixa motivação
        motivation['last_activity_score'] = 1
        
        ability['cognitive_load'] = 6  # Alta carga cognitiva (cansaço)
        ability['confidence_score'] = 4
        
        trigger['sleeping'] = False
        trigger['arousal'] = 1  # Baixo arousal
        trigger['location'] = 'work'
        trigger['motion_activity'] = 'stationary'
        # M=1, A=1, T=2 → FBM=2 a 18
    
    # NOITE (18-23h): FBM ALTO (M=3-4, A=3-4, T=5)
    elif 18 <= hour <= 23:
        motivation['valence'] = 3  # Boa motivação
        motivation['last_activity_score'] = 3
        
        ability['cognitive_load'] = 2  # Carga moderada
        ability['confidence_score'] = 8
        
        trigger['sleeping'] = False
        trigger['arousal'] = 4  # Bom arousal
        trigger['location'] = 'home'
        trigger['motion_activity'] = 'walking'
        # M=3-4, A=3-4, T=5 → FBM=45-80
    
    # MADRUGADA (0-5h): FBM BAIXO (dormindo)
    else:
        motivation['valence'] = 0
        motivation['last_activity_score'] = 0
        
        ability['cognitive_load'] = 0
        ability['confidence_score'] = 3
        
        trigger['sleeping'] = True
        trigger['arousal'] = 0
        trigger['location'] = 'home'
        trigger['motion_activity'] = 'stationary'
        # M=0, A=0, T=0 → FBM=0
    
    return {
        'motivation_factors': motivation,
        'ability_factors': ability,
        'trigger_factors': trigger
    }

def create_fbm_variado_profile():
    """Cria novo perfil baseado no matinal com FBM manipulado."""
    
    base_dir = Path(__file__).parent.parent
    original_path = base_dir / 'data' / 'users' / 'user_matinal_rl_v2.json'
    output_dir = base_dir / 'data' / 'simulation' / 'NovoPerfil'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'user_fbm_variado.json'
    
    print("📂 Carregando perfil original...")
    with open(original_path, 'r', encoding='utf-8') as f:
        original_profile = json.load(f)
    
    print("🔧 Manipulando fatores FBM...")
    
    # Cria novo perfil
    new_profile = deepcopy(original_profile)
    new_profile['user_id'] = 'user_fbm_variado'
    
    # Manipula cada hora de cada dia
    for day in new_profile['days']:
        for hour_data in day['hours']:
            hour = hour_data['hour']
            
            # Substitui fatores
            new_factors = manipulate_fbm_factors(hour, hour_data)
            hour_data['motivation_factors'] = new_factors['motivation_factors']
            hour_data['ability_factors'] = new_factors['ability_factors']
            hour_data['trigger_factors'] = new_factors['trigger_factors']
            
            # Zera feedback (vai ser recalculado na simulação)
            hour_data['feedback'] = {
                'notification_sent': False,
                'action_performed': False,
                'training_feedback': None
            }
    
    # Salva novo perfil
    print(f"💾 Salvando novo perfil: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_profile, f, indent=2, ensure_ascii=False)
    
    print("✅ Perfil criado com sucesso!")
    
    # Mostra estatísticas
    print("\n📊 Estatísticas esperadas:")
    print("  Manhã (6-11h): FBM ~60-96 (ALTO)")
    print("  Tarde (12-17h): FBM ~2-30 (BAIXO)")
    print("  Noite (18-23h): FBM ~45-80 (ALTO)")
    print("  Madrugada (0-5h): FBM ~0 (BAIXO, dormindo)")
    
    return output_path

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏗️  CRIADOR DE PERFIL FBM VARIADO")
    print("="*70 + "\n")
    
    output_path = create_fbm_variado_profile()
    
    print("\n" + "="*70)
    print(f"📁 Perfil salvo: {output_path}")
    print("="*70 + "\n")
