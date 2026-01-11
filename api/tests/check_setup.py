"""
Script de verificação: confirma que tudo está pronto para validação interativa.
"""

import sys
import os

def check_imports():
    """Verifica se todas as importações necessárias funcionam."""
    print("🔍 Verificando importações...")
    
    try:
        import random
        import json
        from datetime import date, timedelta
        print("   ✅ Bibliotecas padrão: OK")
    except Exception as e:
        print(f"   ❌ Erro em bibliotecas padrão: {e}")
        return False
    
    try:
        import gymnasium as gym
        import numpy as np
        print("   ✅ Gymnasium + NumPy: OK")
    except Exception as e:
        print(f"   ❌ Erro em Gymnasium/NumPy: {e}")
        print("   💡 Instale: pip install gymnasium numpy")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("   ✅ Stable-Baselines3: OK")
    except Exception as e:
        print(f"   ❌ Erro em Stable-Baselines3: {e}")
        print("   💡 Instale: pip install stable-baselines3")
        return False
    
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from tests.synthetic_data_generator import SyntheticDataGenerator, PERFIL_MATINAL
        from app.rl.environment import NotificationEnv
        from app.rl.trainer import RLTrainer
        print("   ✅ Módulos do projeto: OK")
    except Exception as e:
        print(f"   ❌ Erro em módulos do projeto: {e}")
        return False
    
    return True

def check_directories():
    """Verifica se diretórios necessários existem."""
    print("\n📁 Verificando diretórios...")
    
    dirs = [
        "tests",
        "tests/results",
        "tests/models",
        "tests/logs",
        "app/rl"
    ]
    
    all_ok = True
    for dir_path in dirs:
        full_path = os.path.join(os.path.dirname(__file__), '..', dir_path)
        if os.path.exists(full_path):
            print(f"   ✅ {dir_path}: OK")
        else:
            print(f"   ⚠️  {dir_path}: Não existe (será criado automaticamente)")
    
    return True

def check_files():
    """Verifica se arquivos necessários existem."""
    print("\n📄 Verificando arquivos...")
    
    files = [
        "tests/interactive_validation.py",
        "tests/synthetic_data_generator.py",
        "app/rl/environment.py",
        "app/rl/trainer.py"
    ]
    
    all_ok = True
    for file_path in files:
        full_path = os.path.join(os.path.dirname(__file__), '..', file_path)
        if os.path.exists(full_path):
            print(f"   ✅ {file_path}: OK")
        else:
            print(f"   ❌ {file_path}: NÃO ENCONTRADO")
            all_ok = False
    
    return all_ok

def test_fbm_calculation():
    """Testa cálculo de FBM."""
    print("\n🧮 Testando cálculo de FBM...")
    
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from tests.synthetic_data_generator import SyntheticDataGenerator, PERFIL_MATINAL
        from datetime import date
        
        generator = SyntheticDataGenerator(PERFIL_MATINAL, seed=42)
        day_data = generator.generate_day(
            user_id="test",
            target_date=date(2025, 11, 21),
            notification_strategy="never"
        )
        
        # Testa cálculo em algumas horas
        test_passed = True
        for hour_num in [6, 12, 22]:
            hour_data = day_data["hours"][hour_num]
            mf = hour_data["motivation_factors"]
            af = hour_data["ability_factors"]
            tf = hour_data["trigger_factors"]
            
            # Simples verificação
            if not isinstance(mf["valence"], int):
                test_passed = False
                break
        
        if test_passed:
            print("   ✅ Cálculo de FBM: OK")
            return True
        else:
            print("   ❌ Cálculo de FBM: ERRO")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro ao testar FBM: {e}")
        return False

def main():
    """Executa todas as verificações."""
    print("=" * 70)
    print("🔧 VERIFICAÇÃO DE SETUP - VALIDAÇÃO INTERATIVA")
    print("=" * 70)
    
    results = []
    
    # Verifica importações
    results.append(("Importações", check_imports()))
    
    # Verifica diretórios
    results.append(("Diretórios", check_directories()))
    
    # Verifica arquivos
    results.append(("Arquivos", check_files()))
    
    # Testa FBM
    results.append(("Cálculo FBM", test_fbm_calculation()))
    
    # Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO DA VERIFICAÇÃO")
    print("=" * 70)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"   {check_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 TUDO PRONTO! Você pode executar:")
        print("\n   python tests/interactive_validation.py")
        print("\n" + "=" * 70)
        return 0
    else:
        print("\n⚠️  ATENÇÃO: Alguns checks falharam.")
        print("   Corrija os problemas antes de prosseguir.")
        print("\n" + "=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
