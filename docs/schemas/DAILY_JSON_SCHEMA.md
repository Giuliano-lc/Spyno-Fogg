# Schema do JSON Diário — API de Notificações com RL + FBM

## Visão Geral

Este documento define formalmente a estrutura do JSON que será recebido diariamente pela API. O JSON representa os dados comportamentais do usuário coletados ao longo das 24 horas do dia anterior, enviado pelo front-end à meia-noite (00h).

Os dados são organizados de acordo com os componentes do **Fogg Behavior Model (FBM)**:
- **Motivação (M)**: Fatores que influenciam a vontade do usuário de agir
- **Habilidade (A)**: Fatores que indicam capacidade do usuário de agir
- **Gatilho (T)**: Fatores que indicam se o momento é propício para notificação

**Fórmula**: `FBM_score = M × A × T`

---

## Estrutura Completa do JSON

```json
{
  "user_id": "usr_123456",
  "date": "2025-01-15",
  "timezone": "America/Sao_Paulo",
  
  "user_profile": {
    "has_family": true
  },
  
  "hours": [
    {
      "hour": 0,
      
      "motivation_factors": {
        "valence": 1,
        "last_activity_score": 1,
        "hours_slept_last_night": 7
      },
      
      "ability_factors": {
        "cognitive_load": 0,
        "activities_performed_today": 0,
        "time_since_last_activity_hours": 12,
        "confidence_score": 7
      },
      
      "trigger_factors": {
        "sleeping": true,
        "arousal": 0,
        "location": "home",
        "motion_activity": "stationary"
      },
      
      "context": {
        "day_period": 3,
        "is_weekend": false
      },
      
      "feedback": {
        "notification_sent": false,
        "action_performed": false,
        "training_feedback": null
      }
    }
  ]
}
```

---

## Descrição dos Campos

### Campos Raiz

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `user_id` | string | Sim | Identificador único do usuário |
| `date` | string | Sim | Data no formato YYYY-MM-DD |
| `timezone` | string | Sim | Fuso horário do usuário (ex: America/Sao_Paulo) |
| `user_profile` | object | Sim | Perfil estático do usuário |
| `hours` | array[24] | Sim | Array com exatamente 24 objetos (um por hora) |

### user_profile

| Campo | Tipo | Valores | Descrição | Base Teórica |
|-------|------|---------|-----------|--------------|
| `has_family` | boolean | true/false | Se o usuário tem suporte familiar | Jowsey et al (2014): presença de família/amigos é fator externo de motivação |

### hours[].motivation_factors

Fatores que influenciam a **vontade** do usuário de realizar a ação.

| Campo | Tipo | Valores | Descrição | Base Teórica |
|-------|------|---------|-----------|--------------|
| `valence` | integer | 0, 1 | Estado emocional: 0=negativo, 1=positivo | Jowsey et al (2014): "Remaining positive was one of the most important strategies for optimizing health" |
| `last_activity_score` | integer | 0, 1 | Percepção de benefício da última atividade: 0=baixo, 1=alto | Paper: "perceiving self-management behaviour as having limited benefit" desmotiva |
| `hours_slept_last_night` | integer | 0-12 | Horas de sono na noite anterior | Dolsen et al (2017): sono insuficiente (<7h) prejudica motivação |

### hours[].ability_factors

Fatores que indicam a **capacidade** do usuário de agir naquele momento.

| Campo | Tipo | Valores | Descrição | Base Teórica |
|-------|------|---------|-----------|--------------|
| `cognitive_load` | integer | 0, 1 | Carga cognitiva: 0=baixa, 1=alta | Chan et al (2020): "users were more receptive to prompts under low cognitive load" |
| `activities_performed_today` | integer | 0-N | Quantidade de atividades já realizadas no dia | Paper: se >2, usuário pode estar "cansado de repetir" (strained) |
| `time_since_last_activity_hours` | integer | 0-48 | Horas desde a última atividade | Influencia prontidão para nova atividade |
| `confidence_score` | integer | 0-10 | Autoeficácia baseada em histórico de sucesso | Bandura (1997): "belief in one's capabilities to organize and execute courses of action" |

### hours[].trigger_factors

Fatores que indicam se o **momento é propício** para enviar notificação.

| Campo | Tipo | Valores | Descrição | Base Teórica |
|-------|------|---------|-----------|--------------|
| `sleeping` | boolean | true/false | Se o usuário está dormindo | **NUNCA** enviar notificação se true |
| `arousal` | integer | 0, 1, 2 | Nível de ativação: 0=baixo, 1=médio, 2=alto | Goyal et al (2017): "users are likely to pay attention to notifications at times of increasing arousal". Médio (1) é ideal |
| `location` | string | home, work, other | Localização atual | Ho et al (2018): usuários preferem notificações em casa |
| `motion_activity` | string | stationary, walking, running | Atividade física atual | Aminikhanghahi (2017): "stationary" é ideal para receptividade |

### hours[].context

Informações contextuais de tempo.

| Campo | Tipo | Valores | Descrição | Base Teórica |
|-------|------|---------|-----------|--------------|
| `day_period` | integer | 0, 1, 2, 3 | Período: 0=manhã(6-10h), 1=meio-dia(10-18h), 2=noite(18-22h), 3=madrugada(22-6h) | Bidargadi et al (2018): timing importa |
| `is_weekend` | boolean | true/false | Se é fim de semana | Bidargadi et al (2018): "users are more likely to engage when notifications are sent at mid-day on weekends" |

### hours[].feedback

Dados de feedback para aprendizado do modelo.

| Campo | Tipo | Valores | Descrição |
|-------|------|---------|-----------|
| `notification_sent` | boolean | true/false | Se uma notificação foi enviada nesta hora |
| `action_performed` | boolean | true/false | Se o usuário executou a ação desejada nesta hora |
| `training_feedback` | object/null | objeto ou null | Feedback pós-treino (apenas se `action_performed=true`) |

### hours[].feedback.training_feedback

**Preenchido apenas quando `action_performed=true`**. Contém as respostas do usuário após realizar o treino.

| Campo | Tipo | Valores | Descrição |
|-------|------|---------|-----------|
| `difficulty_level` | integer | 1-5 | "Qual nível de dificuldade para realizar o treino?" (1=muito fácil, 5=muito difícil) |
| `familiarity_level` | integer | 1-5 | "Qual seu nível de familiaridade com esses exercícios?" (1=nenhuma, 5=muito alta) |
| `completed_fully` | boolean | true/false | Se completou o treino inteiro |
| `duration_minutes` | integer | 0-N | Duração real do treino em minutos |

---

## Mapeamento para Cálculo do FBM

### MOTIVAÇÃO (M) — Range: 0 a 4

```
M = valence_score + family_score + benefit_score + sleep_score

Onde:
- valence_score = 1 se valence == 1, senão 0
- family_score = 1 se has_family == true, senão 0
- benefit_score = 1 se last_activity_score == 1, senão 0
- sleep_score = 1 se hours_slept_last_night >= 7, senão 0
```

### HABILIDADE (A) — Range: 0 a 4

```
A = load_score + strain_score + ready_score + confidence_score

Onde:
- load_score = 1 se cognitive_load == 0 (baixa), senão 0
- strain_score = 1 se activities_performed_today <= 1, senão 0
- ready_score = 1 se time_since_last_activity_hours >= 1, senão 0
- confidence_score = 1 se confidence_score >= 4, senão 0
```

### GATILHO (T) — Range: 0 a 6

```
Se sleeping == true:
    T = 0  # NUNCA notificar dormindo

Senão:
    T = awake + arousal_score + location_score + motion_score + time_score + day_score

Onde:
- awake = 1 (pois sleeping == false)
- arousal_score = 1 se arousal == 1 (médio), senão 0
- location_score = 1 se location == "home", senão 0
- motion_score = 1 se motion_activity == "stationary", senão 0
- time_score = 1 se day_period coincide com preferência do usuário
- day_score = 1 se is_weekend == true, senão 0
```

### SCORE FINAL

```
FBM_score = M × A × T

Range teórico: 0 a 96 (4 × 4 × 6)
```

---

## Matriz de Confusão para Aprendizado

Os campos `notification_sent` e `action_performed` permitem calcular:

| Cenário | notification_sent | action_performed | Interpretação | Recompensa RL |
|---------|-------------------|------------------|---------------|---------------|
| **VP** (Verdadeiro Positivo) | true | true | Notificou e usuário executou ✅ | +reward |
| **VN** (Verdadeiro Negativo) | false | false | Não notificou e usuário não faria | 0 |
| **FP** (Falso Positivo) | true | false | Notificou mas usuário ignorou ❌ | -penalty |
| **FN** (Falso Negativo) | false | true | Não notificou mas usuário fez sozinho | Informativo |

---

## Exemplo Completo de JSON Diário

```json
{
  "user_id": "usr_abc123",
  "date": "2025-01-15",
  "timezone": "America/Sao_Paulo",
  
  "user_profile": {
    "has_family": true
  },
  
  "hours": [
    {
      "hour": 0,
      "motivation_factors": {
        "valence": 0,
        "last_activity_score": 1,
        "hours_slept_last_night": 0
      },
      "ability_factors": {
        "cognitive_load": 0,
        "activities_performed_today": 0,
        "time_since_last_activity_hours": 8,
        "confidence_score": 6
      },
      "trigger_factors": {
        "sleeping": true,
        "arousal": 0,
        "location": "home",
        "motion_activity": "stationary"
      },
      "context": {
        "day_period": 3,
        "is_weekend": false
      },
      "feedback": {
        "notification_sent": false,
        "action_performed": false,
        "training_feedback": null
      }
    },
    {
      "hour": 10,
      "motivation_factors": {
        "valence": 1,
        "last_activity_score": 1,
        "hours_slept_last_night": 7
      },
      "ability_factors": {
        "cognitive_load": 0,
        "activities_performed_today": 0,
        "time_since_last_activity_hours": 18,
        "confidence_score": 7
      },
      "trigger_factors": {
        "sleeping": false,
        "arousal": 1,
        "location": "home",
        "motion_activity": "stationary"
      },
      "context": {
        "day_period": 1,
        "is_weekend": true
      },
      "feedback": {
        "notification_sent": true,
        "action_performed": true,
        "training_feedback": {
          "difficulty_level": 3,
          "familiarity_level": 4,
          "completed_fully": true,
          "duration_minutes": 25
        }
      }
    }
  ]
}
```

---

## Impacto do TrainingFeedback nos Cálculos Futuros do FBM

O `training_feedback` coletado após cada treino realizado **influencia os cálculos do FBM nas próximas horas e dias**. Isso cria um ciclo de aprendizado onde o comportamento passado afeta as previsões futuras.

### Mapeamento TrainingFeedback → Componentes FBM

| Campo | Componente Afetado | Lógica de Atualização | Base Teórica |
|-------|-------------------|----------------------|--------------|
| `difficulty_level` | **HABILIDADE** → `confidence_score` | Se dificuldade alta (4-5): diminui confidence_score futuro em -1 ou -2. Se baixa (1-2): aumenta em +1 | Paper: "Self-efficacy... perception of one's capability to execute the target activity, and whether the person has performed this activity **successfully** before" |
| `familiarity_level` | **HABILIDADE** → `cognitive_load` | Alta familiaridade (4-5): reduz probabilidade de carga cognitiva alta em treinos similares | Chan et al (2020): "users were more receptive to prompts under low cognitive load" |
| `completed_fully` | **MOTIVAÇÃO** → `last_activity_score` | Se `true`: próximo `last_activity_score = 1` (percepção de benefício alta). Se `false`: `last_activity_score = 0` | Paper: "During-behaviour affect is predictive of concurrent and **future** physical activity behaviour" |
| `duration_minutes` | **HABILIDADE** → `activities_performed_today`, `time_since_last_activity` | Treino longo (>30min) pode aumentar "strain" (cansaço). Afeta `unstrained` score | Paper: "patient's ability to perform the target behaviour may be affected by the patient being **tired** or **bored** of repeating the same activity" |

### Fórmulas de Atualização Propostas

**1. Atualização de `confidence_score` (autoeficácia):**
```python
# Após cada treino completado
if completed_fully:
    if difficulty_level <= 2:  # Fácil
        confidence_score = min(10, confidence_score + 1)
    elif difficulty_level >= 4:  # Difícil mas completou
        confidence_score = min(10, confidence_score + 0.5)  # Pequeno ganho
else:  # Não completou
    if difficulty_level >= 4:  # Muito difícil
        confidence_score = max(0, confidence_score - 2)
    else:
        confidence_score = max(0, confidence_score - 1)
```

**2. Atualização de `last_activity_score` (percepção de benefício):**
```python
# Para a próxima hora/dia
if completed_fully and difficulty_level <= 3:
    last_activity_score = 1  # Experiência positiva → motiva repetição
else:
    last_activity_score = 0  # Experiência negativa ou incompleta
```

**3. Influência de `familiarity_level` na carga cognitiva:**
```python
# Reduz probabilidade de high cognitive_load em horários similares
if familiarity_level >= 4:
    # Usuário conhece os exercícios → menor esforço mental
    cognitive_load_modifier = -0.3  # Aplica em previsões futuras
```

**4. Influência de `duration_minutes` no strain:**
```python
# Afeta "unstrained" score nas próximas horas
if duration_minutes > 45:
    strain_hours = 3  # Precisa de mais tempo para recuperar
elif duration_minutes > 30:
    strain_hours = 2
else:
    strain_hours = 1
```

### Ciclo de Retroalimentação

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   JSON Dia N                                                    │
│   └── training_feedback (dificuldade=3, familiaridade=4,        │
│                          completed=true, duration=25min)        │
│                                                                 │
│                    ▼                                            │
│                                                                 │
│   Atualização de Estado do Usuário                              │
│   └── confidence_score: 6 → 7 (completou com dificuldade média) │
│   └── last_activity_score: 0 → 1 (experiência positiva)         │
│   └── strain_window: próximas 2h com unstrained=0               │
│                                                                 │
│                    ▼                                            │
│                                                                 │
│   JSON Dia N+1                                                  │
│   └── Novos valores base refletem o feedback anterior           │
│   └── Modelo RL usa histórico para melhorar predições           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Por que isso importa para o RL?

O modelo PPO vai aprender que:
1. **Notificar após treino fácil bem-sucedido** → usuário com alta motivação/habilidade → maior chance de VP
2. **Notificar logo após treino longo** → usuário cansado (strained) → maior chance de FN
3. **Padrões de familiaridade** → se usuário tem alta familiaridade, pode aceitar treinos em momentos de maior carga cognitiva

Isso permite ao RL **personalizar** as recomendações baseado no histórico real de cada usuário.

---

## Notas de Implementação

1. **Array hours**: Deve conter **exatamente 24 objetos**, um para cada hora (0-23)
2. **training_feedback**: Apenas preenchido quando `action_performed=true`
3. **sleeping=true**: Zera automaticamente o Gatilho (T), independente dos outros fatores
4. **Threshold dinâmico**: Será ajustado baseado na matriz de confusão VP/VN/FP/FN ao longo do tempo
5. **Retroalimentação**: Os valores de `training_feedback` de um dia afetam os cálculos base do dia seguinte
