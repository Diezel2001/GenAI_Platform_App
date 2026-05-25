---
tags:
  - ai
  - agents
  - multi-agent
  - architecture
---
```mermaid
flowchart TD

    A["Task / Objective
    ---
    Incoming user request
    or autonomous goal"]

    subgraph Planning Layer

        B["Planner Agent
        ---
        - Understand intent
        - Determine complexity
        - Create execution plan"]

        C["Task Decomposer
        ---
        Breaks objective into
        executable subtasks"]

        D["Task Orchestrator
        ---
        Assigns skills/tools
        and coordinates flow"]

    end

    subgraph Capability Layer

        E["Skill Retriever
        ---
        Hybrid retrieval:
        - semantic search
        - BM25
        - metadata"]

        F["Tool Registry
        ---
        Stores:
        - tool schemas
        - validators
        - executors"]

    end

    subgraph Execution Layer

        G["Execution Blueprint
        ---
        Maps:
        task → skills → tools"]

        H["Worker Spawner
        ---
        Creates isolated
        worker contexts"]

        I["Worker Agents
        ---
        Specialized agents:
        - research
        - analysis
        - coding"]

        J["Execution Loop
        ---
        Thought
        → Action
        → Observation"]

    end

    subgraph Synthesis Layer

        K["Result Collector
        ---
        Aggregates worker
        outputs and failures"]

        L["Synthesis Engine
        ---
        Merges outputs
        and validates results"]

        M["Final Response
        ---
        Final structured
        output to user"]

    end

    A --> B
    B --> C
    C --> D

    D --> E
    D --> F

    E --> G
    F --> G

    G --> H
    H --> I
    I --> J

    J --> K
    K --> L
    L --> M
```



### Agents:
- [[Planner Agent]] 
	- Planning, Decomposition, Routing
	- Task and Skill Delegation
	- Agent Spawning
- [[Retrieval Agent]]
	- Given Skill and Task
	- Retrieves Infromation from Documents
- [[Worker Agent]]
	- ReAct type agent
	- Skill Executor

### Services:
- [[Agent Manager]]
- [[Skill Manager]]
- [[Tool Manager]]
- [[RAG Manager]]
- 
