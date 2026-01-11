# API de Notificações com RL + FBM

API REST para treinamento de modelo PPO (Reinforcement Learning) usando Fogg Behavior Model para otimização de notificações.

## Instalação

```bash
cd api
python -m pip install -r requirements.txt
```

## Comandos Principais

### Iniciar API

```bash
python run.py
# ou
python -m uvicorn main:app --port 8000 --reload
```

API disponível em: http://localhost:8000

## Gerador de Dados Sintéticos

O `synthetic_data_generator.py` é o componente central para criação de perfis de usuário realistas:

**Características:**
- **Perfis comportamentais**: Define padrões (matinal, noturno) com horários preferidos, sono, trabalho
- **Cálculo FBM completo**: Motivação (valência, sono, atividades) × Habilidade (carga cognitiva, confiança) × Gatilho (arousal, localização, momento)
- **Variação realista**: Simula fim de semana, variações de humor, fadiga, confiança que evolui com sucesso
- **Estratégias de notificação**: Suporta fbm_based, smart, random, always, never
- **Dados detalhados**: Gera feedback de treino (dificuldade, familiaridade, duração)

**Uso:**
```bash
python tests/synthetic_data_generator.py 
```

Este gerador foi usado para criar os perfis JSON que alimentam a simulação principal de 100 epochs × 30 dias.
o script tests/simulations/run_simulation_with_rl_fbm_enhanced.py chama o tests/synthetic_data_generator.py para gerar o perfil definido antes de executar a simulação.

## Scripts de Simulação e Análise

### 🔄 Simulações (`tests/simulations/`)

**Simulação Principal - RL**
```bash
# Terminal 1: API rodando
python run.py

# Terminal 2: Executar simulação (escolhe tests presentes na pasta, modificar conforme necessidade)
python tests/simulations/run_simulation_with_rl_fbm_enhanced.py
```


**Pipeline de Treinamento (30 dias)**
```bash
python tests/simulations/training_pipeline.py
```

**Outras simulações disponíveis:**
```bash
python tests/simulations/run_simulation_shift_behavior.py      # Mudança de comportamento
python tests/simulations/run_simulation_hybrid_rl.py           # RL híbrido
python tests/simulations/run_simulation_incremental_rl.py      # RL incremental
```

### 📊 Visualizações (`tests/visualizations/`)

**Gráficos FBM**
```bash
python tests/visualizations/plot_fbm_threshold.py              # M×A com threshold
python tests/visualizations/plot_fbm_total.py                  # FBM total
python tests/visualizations/plot_threshold_evolution.py        # Evolução threshold
```

**Geração de Dados**
```bash
python tests/visualizations/generate_monthly_data.py           # 30 dias sintéticos
python tests/synthetic_data_generator.py                       # Gerador de perfis FBM realistas (usado para criar dados de treinamento)
```

**Análises**
```bash
python tests/visualizations/analyze_simulation_data.py         # Análise geral
python tests/visualizations/generate_rl_learning_analysis.py   # Análise aprendizado RL
```

### Gerar Dados Sintéticos

```bash
python tests/simulations/training_pipeline.py
```

Fluxo: Envia dados → Treina PPO → Gera previsões → Avalia acurácia

## Endpoints da API

### Documentação Interativa
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Rotas de Treino (`/treino`)

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/treino` | Recebe JSON diário (24h) do usuário |
| `GET` | `/treino/historico/{user_id}` | Retorna histórico completo |
| `GET` | `/treino/dados-treinamento/{user_id}` | Dados formatados para RL |
| `GET` | `/treino/usuarios` | Lista todos os usuários |
| `DELETE` | `/treino/historico/{user_id}` | Remove histórico do usuário |

### Rotas de Previsão (`/previsao`)

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/previsao/{user_id}` | Previsão completa (filtrada por threshold) |
| `GET` | `/previsao/{user_id}/simples` | Apenas top 3 horas recomendadas |
| `POST` | `/previsao/{user_id}/custom` | Previsão com dados customizados |

### Rotas de Threshold Dinâmico (`/threshold`)

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/threshold/{user_id}` | Threshold atual do usuário |
| `POST` | `/threshold/{user_id}/feedback` | Registra feedback (VP/VN/FP/FN) |
| `POST` | `/threshold/{user_id}/check` | Verifica se deve notificar dado FBM |
| `GET` | `/threshold/{user_id}/stats` | Estatísticas (contagens, taxas) |
| `GET` | `/threshold/{user_id}/history` | Histórico de ajustes |
| `GET` | `/threshold/{user_id}/decision/{fbm}` | Decisão rápida: notificar? |
| `POST` | `/threshold/{user_id}/config` | Configura parâmetros |
| `POST` | `/threshold/{user_id}/reset` | Reseta para valor inicial |

### Exemplos de Uso (curl)

```bash
# Health check
curl http://localhost:8000/health

# Listar usuários
curl http://localhost:8000/treino/usuarios

# Obter previsão simplificada
curl http://localhost:8000/previsao/user_matinal_30dias/simples

# Obter histórico
curl http://localhost:8000/treino/historico/user_matinal_30dias

# Threshold dinâmico - ver atual
curl http://localhost:8000/threshold/user_matinal_30dias

# Threshold - decisão rápida (FBM=25)
curl http://localhost:8000/threshold/user_matinal_30dias/decision/25

# Threshold - registrar feedback
curl -X POST http://localhost:8000/threshold/user_matinal_30dias/feedback \
  -H "Content-Type: application/json" \
  -d '{"hour": 7, "notified": true, "executed": true, "fbm_score": 30}'

# Threshold - ver estatísticas
curl http://localhost:8000/threshold/user_matinal_30dias/stats
```

## Scripts

| Script | Função |
|--------|--------|
| `generate_monthly_data.py` | Gera 30 dias sintéticos |
| `training_pipeline.py` | Pipeline: enviar → treinar → prever |
| `plot_fbm_threshold.py` | Gráfico M×A com threshold |
| `plot_fbm_total.py` | Gráfico FBM total |
| `plot_threshold_evolution.py` | Evolução threshold dinâmico |

## Estrutura

```
api/
├── main.py                    # FastAPI app
├── run.py                     # Server starter
├── app/
│   ├── models/                # Pydantic models
│   ├── routers/               # API routes
│   ├── services/              # Storage, threshold
│   └── rl/                    # Environment, trainer
├── tests/
│   ├── simulations/           # Scripts de simulação
│   ├── visualizations/        # Scripts de gráficos/análise
│   └── data/                  # Dados de teste
├── data/
│   ├── users/                 # Histórico JSON
│   ├── simulation/            # Dados de simulação
│   ├── synthetic/             # Dados gerados
│   ├── results/               # Resultados
│   └── visualizations/        # Gráficos
└── models/                    # Modelos PPO treinados
```

## Fogg Behavior Model

```
Behavior = (M × A × T) > threshold
```

- **M (Motivação)**: 0-4
- **A (Habilidade)**: 0-4  
- **T (Gatilho)**: 0-6
- **Threshold**: Dinâmico (ajustado por VP/VN/FP/FN)

## Uso Rápido

### Exemplo 1: Pipeline Completo
```bash
python run.py                                               # 1. Iniciar API
python tests/visualizations/generate_monthly_data.py        # 2. Gerar dados
python tests/simulations/training_pipeline.py               # 3. Treinar
curl http://localhost:8000/previsao/user_matinal_30dias/simples  # 4. Prever
```

### Exemplo 2: Simulação RL + Threshold Dinâmico (60 dias)
```bash
# Terminal 1: Subir API
python run.py

# Terminal 2: Executar simulação
python tests/simulations/run_simulation_with_rl_fbm_enhanced.py
```

**Saída da simulação:**
- `data/simulation/` - Dados por dia
- `data/reports/` - Relatórios de métricas (VP/VN/FP/FN)
- `data/visualizations/` - Gráficos de evolução
- Modelo PPO treinado incrementalmente

## Referência

Projeto CAPABLE: *"From personalized timely notification to healthy habit formation"*  
https://github.com/Capable-project/capable-rl4vc
