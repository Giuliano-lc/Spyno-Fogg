"""
Verifica se o ambiente está pronto para executar simulação com RL.
"""

import sys
import requests
from pathlib import Path

print("="*70)
print("🔍 VERIFICAÇÃO DE AMBIENTE - Simulação RL")
print("="*70)
print()

checks_passed = 0
checks_total = 5

# 1. Verifica imports
print("1️⃣ Verificando dependências Python...")
try:
    import stable_baselines3
    print("   ✅ stable-baselines3 instalado")
    checks_passed += 1
except ImportError:
    print("   ❌ stable-baselines3 não encontrado")
    print("      Instale com: pip install stable-baselines3")

try:
    import gymnasium
    print("   ✅ gymnasium instalado")
    checks_passed += 1
except ImportError:
    print("   ❌ gymnasium não encontrado")
    print("      Instale com: pip install gymnasium")

try:
    import requests
    print("   ✅ requests instalado")
    checks_passed += 1
except ImportError:
    print("   ❌ requests não encontrado")
    print("      Instale com: pip install requests")

# 2. Verifica API
print("\n2️⃣ Verificando API...")
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    if response.status_code == 200:
        print("   ✅ API está rodando em http://localhost:8000")
        checks_passed += 1
    else:
        print(f"   ⚠️ API respondeu com status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ❌ API NÃO está rodando")
    print("      Inicie com: python start.py")
except requests.exceptions.Timeout:
    print("   ❌ API timeout")

# 3. Verifica endpoints necessários
print("\n3️⃣ Verificando endpoints da API...")
try:
    # Testa /treino
    test_data = {
        "user_id": "test",
        "date": "2024-01-01",
        "hours": []
    }
    response = requests.post(
        "http://localhost:8000/treino",
        json=test_data,
        timeout=5
    )
    if response.status_code in [200, 400]:  # 400 também ok (dados inválidos esperado)
        print("   ✅ Endpoint /treino disponível")
        checks_passed += 1
    else:
        print(f"   ⚠️ Endpoint /treino status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Erro ao testar /treino: {e}")

# Resultado final
print("\n" + "="*70)
print(f"📊 RESULTADO: {checks_passed}/{checks_total} verificações passaram")
print("="*70)

if checks_passed == checks_total:
    print("\n✅ TUDO PRONTO! Você pode executar:")
    print("   python tests\\run_simulation_with_rl.py")
else:
    print(f"\n⚠️ {checks_total - checks_passed} problema(s) encontrado(s)")
    print("\n🔧 AÇÕES NECESSÁRIAS:")
    
    if checks_passed < 3:
        print("   1. Instale dependências:")
        print("      pip install stable-baselines3 gymnasium requests")
    
    if checks_passed < 4:
        print("   2. Inicie a API:")
        print("      python start.py")
    
    print("\n   Depois execute novamente: python tests\\check_rl_ready.py")

print()
