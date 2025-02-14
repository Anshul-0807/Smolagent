# SmolAgent-based AI Assistants

This repository contains implementations of different AI agents built using the SmolAgent framework. The project demonstrates various use cases of AI agents, from simple conversational bots to multi-agent systems and dataset-driven question-answering agents.

## Project Components

### 1. Conversational Agent
A basic conversational AI agent that can engage in natural dialogue with users.

Key features:
- Natural language processing capabilities
- Context-aware responses
- General-purpose conversation handling

### 2. Dataset-Driven QA Agent
An AI agent specifically designed to answer questions based on loaded datasets.

Key features:
- Dataset loading and preprocessing
- Query processing against loaded data
- Accurate information retrieval and response generation

### 3. Multi-Agent System
A system implementing multiple coordinating AI agents that work together to accomplish tasks.

Key features:
- Inter-agent communication
- Task distribution and coordination
- Parallel processing capabilities

## Installation

```bash
pip install smolAgent
```

## Usage Examples

### Conversational Agent
```python
from smolAgent import ConversationalAgent

agent = ConversationalAgent()
response = agent.chat("Hello, how are you?")
```

### Dataset QA Agent
```python
from smolAgent import DatasetAgent

agent = DatasetAgent()
agent.load_dataset("path/to/your/dataset.csv")
response = agent.query("What insights can you provide about the data?")
```

### Multi-Agent System
```python
from smolAgent import MultiAgentSystem

system = MultiAgentSystem()
system.add_agent("conversation", ConversationalAgent())
system.add_agent("data", DatasetAgent())
system.coordinate_task("Analyze this conversation and provide data-driven insights")
```

## Requirements
- Python 3.8+
- SmolAgent library
- Additional dependencies listed in requirements.txt

## Project Structure
```
.
├── agents/
│   ├── conversational.py
│   ├── dataset_qa.py
│   └── multi_agent.py
├── data/
│   └── sample_datasets/
├── tests/
├── requirements.txt
└── README.md
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- SmolAgent framework developers
- Contributors to the project
- Open-source AI community
