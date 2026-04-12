<p align="center">
  <h1 align="center">EvoAdAgent</h1>
  <p align="center">
    <em>A Self-Evolving Agent That Learns to Recommend by Doing</em>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> |
    <a href="#architecture">Architecture</a> |
    <a href="#features">Features</a> |
    <a href="#usage">Usage</a> |
    <a href="#roadmap">Roadmap</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/LangGraph-0.4+-green.svg" alt="LangGraph">
    <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="MIT License">
    <img src="https://img.shields.io/badge/status-active%20development-brightgreen.svg" alt="Status">
  </p>
</p>

---

**EvoAdAgent** is a self-evolving agent for ad/recommendation optimization. It executes campaigns in a simulated environment, reflects on outcomes using [Reflexion](https://arxiv.org/abs/2303.11366), distills reusable strategies, and continuously improves -- all without human intervention.

Unlike static recommendation systems, EvoAdAgent **learns from its own experience**: each campaign round produces reflections, reflections crystallize into strategies, and strategies guide future decisions. The strategy library grows autonomously over time.

## Features

- **Dual-Layer LangGraph Architecture** -- Outer graph orchestrates the evolution cycle; inner graph runs a ReAct tool-calling agent for recommendation decisions.
- **Reflexion-Based Self-Improvement** -- Structured post-campaign analysis inspired by *Reflexion* (NeurIPS 2023), producing actionable insights that feed strategy distillation.
- **Autonomous Strategy Distillation** -- Three modes (NEW / REFINE / MERGE) for extracting, improving, and consolidating reusable recommendation strategies.
- **LLM-Simulated User Behavior** -- A three-stage behavior chain (Click? Watch? Engage?) simulates realistic user feedback, inspired by [RecAgent](https://arxiv.org/abs/2306.02552). Zero-cost learning without real traffic.
- **Four-Layer Memory System** -- Campaign logs (SQLite), user profile vectors (FAISS), strategy library (Markdown + index), and evolution metrics work together to provide the agent with full context.
- **Multi-LLM Support** -- Separate LLM configs for executor, reflector, distiller, and simulator. Supports OpenAI, Qwen (DashScope), and DeepSeek out of the box.
- **Real Dataset** -- Built on [KuaiRec](https://kuairec.com/) (Kuaishou recommendation dataset) with Chinese captions, three-level categories, and user profiles.

## Architecture

### Dual-Layer LangGraph Design

```
                        OUTER GRAPH (Evolution Cycle)
 ┌─────────────────────────────────────────────────────────────────────┐
 │                                                                     │
 │   ┌──────────┐      ┌──────────┐      ┌──────────┐                 │
 │   │ Execute  │─────>│ Reflect  │─────>│ Distill  │────> END        │
 │   │   Node   │      │   Node   │      │   Node   │                 │
 │   └────┬─────┘      └──────────┘      └─────┬────┘                 │
 │        │            Reflexion-based          │                      │
 │        │            analysis of what         │  NEW / REFINE /      │
 │        │            worked & failed          │  MERGE strategy      │
 │        │                                     │                      │
 │   ┌────┴──────────────────┐            ┌─────┴──────────────┐      │
 │   │   INNER GRAPH         │            │  Strategy Library   │      │
 │   │   (ReAct Agent)       │            │  (Markdown + Index) │      │
 │   │                       │            └─────────────────────┘      │
 │   │  ┌───────┐  ┌──────┐ │                                         │
 │   │  │ Agent │─>│ Tool │ │  ReAct Loop                             │
 │   │  │ Node  │<─│ Node │ │  (think → act → observe → ...)          │
 │   │  └───┬───┘  └──────┘ │                                         │
 │   │      │                │                                         │
 │   │  ┌───┴─────┐         │                                         │
 │   │  │ Extract │──> END  │                                         │
 │   │  │  Node   │         │                                         │
 │   │  └─────────┘         │                                         │
 │   └──────────────────────┘                                         │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘
```

### Core Evolution Loop

```
 ┌───────────────────────────────────────────────────────────────────┐
 │                                                                   │
 │  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌─────────┐        │
 │  │ Execute │──>│ Simulate │──>│ Reflect │──>│ Distill │         │
 │  │  (ReAct │   │  (LLM    │   │ (Verbal │   │ (NEW /  │         │
 │  │  Agent) │   │  Users)  │   │  RL)    │   │ REFINE) │         │
 │  └─────────┘   └──────────┘   └─────────┘   └────┬────┘         │
 │       ^                                           │               │
 │       │         ┌──────────────────┐               │               │
 │       └─────────│ Strategy Library │<──────────────┘               │
 │                 │ (grows each      │                               │
 │                 │  round)          │                               │
 │                 └──────────────────┘                               │
 │                                                                   │
 └───────────────────────────────────────────────────────────────────┘
```

### Four-Layer Memory System

| Layer | Name | Storage | Purpose |
|-------|------|---------|---------|
| L1 | Campaign Log | SQLite | Full decision traces, metrics per round |
| L2 | User Profiles | FAISS (WIP) | User embedding vectors for similarity search |
| L3 | Strategy Library | Markdown + JSON Index | Distilled reusable strategies |
| L4 | Evolution Log | (WIP) | Cross-round performance trends |

### Ad Tools (Inner ReAct Agent)

| Tool | Description |
|------|-------------|
| `analyze_audience` | Analyze user group demographics (gender, age, region, interests) |
| `analyze_content_pool` | Analyze content distribution by category and tags |
| `match_user_content` | Score and rank content for a specific user based on interest overlap |
| `load_strategy` | Load a distilled strategy into the agent's decision process |
| `generate_recommendation` | Produce a final user-content recommendation with reasoning |

## Quick Start

### Prerequisites

- Python 3.10+
- An API key for OpenAI, Qwen (DashScope), or DeepSeek

### Installation

```bash
git clone https://github.com/your-org/evo-ad-agent.git
cd evo-ad-agent
pip install -e .
```

### Run

```bash
# Using OpenAI
OPENAI_API_KEY=sk-xxx python -m examples.demo_evolution --rounds 5

# Using Qwen (DashScope)
DASHSCOPE_API_KEY=sk-xxx python -m examples.demo_evolution --rounds 5 --provider qwen

# Using DeepSeek
DEEPSEEK_API_KEY=sk-xxx python -m examples.demo_evolution --rounds 5 --provider deepseek
```

## Usage

### Basic Evolution Run

```bash
python -m examples.demo_evolution \
    --rounds 10 \
    --provider qwen \
    --model qwen-plus \
    --users 15 \
    --scenario 宠物
```

**Parameters:**

| Arg | Default | Description |
|-----|---------|-------------|
| `--rounds` | 5 | Number of evolution rounds |
| `--provider` | openai | LLM provider (`openai` / `qwen` / `deepseek`) |
| `--model` | gpt-4o-mini | Model name (auto-mapped per provider) |
| `--users` | 10 | Number of users sampled per round |
| `--scenario` | (all) | Scenario filter (e.g., `宠物`, `美食`) |

### Programmatic Usage

```python
from src.agent.evo_ad_agent import EvoAdAgent
from src.config import LLMConfig, ProjectConfig
from src.simulation.scenarios import create_full_scenario

config = ProjectConfig(
    executor_llm=LLMConfig(provider="qwen", model="qwen-plus"),
    reflector_llm=LLMConfig(provider="qwen", model="qwen-plus"),
    distiller_llm=LLMConfig(provider="qwen", model="qwen-plus"),
    simulator_llm=LLMConfig(provider="qwen", model="qwen-plus"),
)

agent = EvoAdAgent(config)
users, contents = create_full_scenario()
agent.setup_environment(users, contents)
agent.run_evolution(rounds=10, scenario="宠物", n_users=15)
```

### Example Output

```
============================================================
  EvoAdAgent Evolution -- 10 rounds
  Scenario: 宠物
  Initial strategies: 0
============================================================

--- Round 1/10 ---
  CTR: 40.00%
  Completion: 30.00%
  Engagement: 10.00%
  Insight: Interest-based matching works for pet content lovers
  Strategy count: 0

--- Round 2/10 ---
  CTR: 55.00%
  Completion: 40.00%
  Engagement: 20.00%
  Insight: Users in tier-2 cities prefer practical pet care content
  NEW STRATEGY: pet-care-practical-match
  Strategy count: 1

  ...

============================================================
  Evolution Summary
  CTR: 40.00% → 72.00% (+32.0pp)
  Strategies learned: 3
============================================================
```

## Project Structure

```
evo-ad-agent/
├── src/
│   ├── agent/                  # Core agent modules
│   │   ├── executor.py             # LangGraph ReAct executor (inner graph)
│   │   ├── reflector.py            # Reflexion-based reflector
│   │   ├── distiller.py            # Strategy distiller (NEW/REFINE/MERGE)
│   │   └── evo_ad_agent.py         # Main evolution loop (outer graph)
│   ├── memory/                 # Four-layer memory system
│   │   ├── campaign_log.py         # L1: SQLite campaign log
│   │   └── strategy_lib.py         # L3: Strategy library
│   ├── simulation/             # LLM-based simulation environment
│   │   ├── ad_environment.py       # Environment wrapper
│   │   ├── user_simulator.py       # Three-stage behavior chain simulator
│   │   └── scenarios.py            # Pre-built scenarios (pet, food, etc.)
│   ├── tools/                  # Ad-specific LangChain tools
│   │   └── ad_tools.py             # 5 tools for the ReAct agent
│   ├── config.py               # Multi-LLM configuration
│   ├── llm_factory.py          # LLM instantiation factory
│   └── models.py               # Pydantic/dataclass data models
├── examples/
│   └── demo_evolution.py       # CLI demo script
├── data/kuairec/               # KuaiRec dataset files
├── strategies/                 # Auto-generated strategy library
├── tests/                      # Test suite
└── pyproject.toml              # Project metadata & dependencies
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph + ToolNode) |
| LLM Integration | [LangChain](https://github.com/langchain-ai/langchain) (ChatOpenAI, multi-provider) |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) (user profile embeddings) |
| API Layer | [FastAPI](https://fastapi.tiangolo.com/) |
| Database | SQLite (campaign logs) |
| Data Processing | Pandas, NumPy |
| LLM Providers | OpenAI, Qwen (DashScope), DeepSeek |

## Dataset

This project uses [KuaiRec](https://kuairec.com/), a real-world recommendation dataset from Kuaishou (Chinese short-video platform), featuring:

- **User profiles** with demographics (gender, age, region, device, activity level)
- **Video items** with Chinese captions, topic tags, and three-level hierarchical categories
- **Interaction logs** for grounding the simulation

## Academic References

This project draws inspiration from the following research:

| Paper | Venue | How It's Used |
|-------|-------|---------------|
| [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | NeurIPS 2023 | Core reflection mechanism -- the agent verbally analyzes what worked and what failed after each round |
| [RecAgent: A Novel Simulation Paradigm for Recommender Systems](https://arxiv.org/abs/2306.02552) | -- | Inspires the LLM-based user behavior simulation with persona-grounded responses |
| [InteRecAgent: Recommender Agent with LLM Tools](https://arxiv.org/abs/2308.16505) | -- | Inspires the tool-augmented recommendation approach |
| [A Survey on Self-Evolving Agents](https://github.com/AgiBot-Hack/Awesome-Self-Evolving-Agents) | Survey | General framework reference for autonomous agent self-improvement |

## Roadmap

### Week 1 -- Core Framework [DONE]
- [x] Dual-layer LangGraph architecture (inner ReAct + outer evolution)
- [x] 5 ad-specific tools for the ReAct executor
- [x] Reflexion-based reflector with structured output
- [x] Strategy distiller with NEW / REFINE modes
- [x] LLM user behavior simulator (three-stage behavior chain)
- [x] SQLite campaign log (L1 memory)
- [x] Markdown strategy library (L3 memory)
- [x] Multi-LLM support (OpenAI / Qwen / DeepSeek)
- [x] KuaiRec data integration with scenario builders
- [x] CLI demo with configurable rounds and providers

### Week 2 -- Evaluation & Verification [IN PROGRESS]
- [ ] A/B strategy verifier (StrategyVerifier)
- [ ] Quantitative evaluation metrics (CTR lift, strategy reuse rate)
- [ ] Automated benchmark against baseline (random / popularity-based)
- [ ] MERGE mode for strategy consolidation

### Week 3 -- Memory & Retrieval [IN PROGRESS]
- [ ] L2 User Profile vectors (FAISS embeddings)
- [ ] L4 Evolution Log with trend analysis
- [ ] Strategy retrieval by scenario similarity (vector search)
- [ ] Cross-round knowledge transfer

### Week 4 -- Production & Visualization [PLANNED]
- [ ] FastAPI serving endpoint
- [ ] Streamlit dashboard for evolution visualization
- [ ] Strategy diff viewer (version history)
- [ ] Multi-scenario parallel evolution

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with LangGraph, powered by Reflexion.</sub>
</p>
