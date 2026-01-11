"""
Demo do script de validação interativa - mostra como o sistema funciona.
Este é um exemplo automatizado para demonstração.
"""

import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.interactive_validation import InteractiveValidator
from datetime import date


def demo_notification_logic():
    """Demonstra a lógica de notificação baseada em FBM."""
    
    print("=" * 100)
    print("🎯 DEMONSTRAÇÃO: SISTEMA DE NOTIFICAÇÕES BASEADO EM FBM")
    print("=" * 100)
    
    # Cria validador
    validator = InteractiveValidator(
        fbm_threshold=30,
        max_notifications_per_day=5,
        seed=42
    )
    
    print("\n📊 Vamos simular um dia e ver quando o sistema decide notificar...")
    print(f"   Threshold: FBM > {validator.fbm_threshold}")
    print(f"   Máx notif/dia: {validator.max_notifications_per_day}")
    
    # Gera dados de um dia
    target_date = date(2025, 11, 21)
    day_data = validator.generate_base_day_data(target_date)
    
    print(f"\n📅 Analisando: {target_date.strftime('%Y-%m-%d')}")
    print("\n" + "=" * 100)
    print(f"{'Hora':^6} | {'Dormindo':^10} | {'M':^4} | {'A':^4} | {'T':^4} | {'FBM':^5} | {'Notificar?':^12} | {'Motivo':^30}")
    print("-" * 100)
    
    notifications_today = 0
    notification_hours = []
    
    for hour_data in day_data["hours"]:
        hour = hour_data["hour"]
        sleeping = hour_data["trigger_factors"]["sleeping"]
        
        # Calcula FBM
        fbm_info = validator.calculate_fbm(hour_data)
        m = fbm_info["motivation"]
        a = fbm_info["ability"]
        t = fbm_info["trigger"]
        fbm = fbm_info["fbm_score"]
        
        # Verifica se deve notificar
        should_notify = validator.should_notify(hour, fbm, sleeping, notifications_today)
        
        # Determina motivo
        if sleeping:
            motivo = "❌ Dormindo"
            notif = "NÃO"
        elif fbm < validator.fbm_threshold:
            motivo = f"❌ FBM baixo (<{validator.fbm_threshold})"
            notif = "NÃO"
        elif notifications_today >= validator.max_notifications_per_day:
            motivo = f"❌ Limite diário ({validator.max_notifications_per_day})"
            notif = "NÃO"
        elif should_notify:
            motivo = "✅ Condições ideais"
            notif = "SIM 🔔"
            notifications_today += 1
            notification_hours.append(hour)
        else:
            motivo = "❓ Outro motivo"
            notif = "NÃO"
        
        sleep_icon = "💤" if sleeping else "👁️"
        
        print(f"{hour:02d}h   | {sleep_icon:^10} | {m:^4} | {a:^4} | {t:^4} | {fbm:^5} | {notif:^12} | {motivo:^30}")
    
    print("-" * 100)
    
    # Resumo
    print(f"\n📊 RESUMO:")
    print(f"   ✅ Total de notificações que seriam enviadas: {notifications_today}")
    print(f"   🕐 Horários: {notification_hours}")
    print(f"   📈 Taxa de notificação: {notifications_today/24*100:.1f}% das horas")
    
    # Análise FBM
    print(f"\n🔍 ANÁLISE DE FBM:")
    fbm_high = sum(1 for h in day_data["hours"] if validator.calculate_fbm(h)["fbm_score"] >= validator.fbm_threshold)
    fbm_sleeping = sum(1 for h in day_data["hours"] if h["trigger_factors"]["sleeping"])
    fbm_awake_high = fbm_high - fbm_sleeping
    
    print(f"   - Horas com FBM alto (>{validator.fbm_threshold}): {fbm_high}")
    print(f"   - Horas dormindo: {fbm_sleeping}")
    print(f"   - Horas acordado com FBM alto: {fbm_awake_high}")
    print(f"   - Notificações enviadas: {notifications_today} (limitado a {validator.max_notifications_per_day})")
    
    print("\n" + "=" * 100)
    print("✅ DEMONSTRAÇÃO CONCLUÍDA!")
    print("=" * 100)
    print("\n💡 Observações:")
    print("   1. Sistema NUNCA notifica quando dormindo")
    print("   2. Sistema SÓ notifica quando FBM > threshold")
    print("   3. Sistema respeita limite diário de notificações")
    print("   4. Notificações são baseadas em CÁLCULO REAL, não horários fixos")
    print("\n🚀 Para validação interativa completa, execute:")
    print("   python tests/interactive_validation.py")
    print("=" * 100)


if __name__ == "__main__":
    demo_notification_logic()
