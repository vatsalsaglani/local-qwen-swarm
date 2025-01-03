# Local Qwen Swarm

A Python implementation of an agent swarm system that works with local LLM servers. The system allows you to create multiple agents that can work together to solve complex tasks using the Qwen 2.5 Coder model.

> **Note:** This implementation is a work in progress and is not production ready.

_**Read the full blog post [here](https://pub.towardsai.net/i-built-an-openai-style-swarm-that-runs-entirely-on-my-laptop-heres-how-ac606ba739f3).**_


## Prerequisites

- Python 3.12+
- Poetry for dependency management
- One of the following LLM servers running locally:
  - LM Studio Server (recommended)
  - llama.cpp server
  - Ollama server

## Setup


1. Clone the repository:

```bash
git clone https://github.com/vatsalsaglani/local-qwen-swarm.git
cd local-qwen-swarm
```

2. Install dependencies:

```bash
poetry install
```

3. Set up your LLM server:

Download the **LM Studio Server** from [here](https://lmstudio.ai/) and follow the instructions to run the server.

Search for **Qwen-2.5-Coder** in the model list and download the model. Load the model and click on **Start Server**.

> **Note:** You can also use the **llama.cpp server** or **Ollama server** by following their respective setup instructions.

## Usage

Here's a basic example of how to use the agent swarm:

```python
import asyncio
from pydantic import BaseModel, Field
from typing import List
from agent.localswarm import LocalSwarm
from agent._types import Agent

# define agents
class Multiply(BaseModel):
    numbers: List[int] = Field(description="List of numbers to multiply")


def multiplication_agent(numbers: List[int]):
    result = 1
    for num in numbers:
        result *= num
    return result


class Add(BaseModel):
    numbers: List[int] = Field(description="List of numbers to add")


def addition_agent(numbers: List[int]):
    return sum(numbers)


local_multiply_agent = Agent(name="multiply",
                             instruction="Multiply the given numbers",
                             functions=[
                                 dict(name="multiply",
                                      description="Multiply the given numbers",
                                      parameters=Multiply.model_json_schema(),
                                      _callable=multiplication_agent)
                             ])

local_add_agent = Agent(name="add",
                        instruction="Add the given numbers",
                        functions=[
                            dict(name="add",
                                 description="Add the given numbers",
                                 parameters=Add.model_json_schema(),
                                 _callable=addition_agent)
                        ])

# create swarm
local_swarm = LocalSwarm(orchestrator_agent=Agent(
    name="orchestrator",
    instruction="Orchestrate the given numbers",
    functions=[]),
                         agents=[local_multiply_agent, local_add_agent],
                         messages=[{
                             "role":
                             "user",
                             "content":
                             "Multiply 2 and 3 and then add 4 to the result"
                         }],
                         max_iterations=10)

# run the swarm
print(
    asyncio.run(
        local_swarm.run_swarm(model="qwen2.5-coder-3b-instruct-q4_k_m",
                              llm_args={})))
```

> To test the swarm, you can use the following command:
> ```bash
> poetry run python -m tests.test_localswarm
> ```

## Configuration

The system uses environment variables for configuration:
- `LOCAL_API_BASE_URL`: Default is "http://localhost:1234/v1"
- `LOCAL_API_KEY`: Default is "your-api"

## Model Information

The system is designed to work with [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/lmstudio-community/Qwen2.5-Coder-3B-Instruct-GGUF), which offers:
- Context support up to 32K tokens
- Optimized for code-related tasks
- Multiple quantization options (Q3_K_L, Q4_K_M, Q6_K, Q8_0)
- Trained on 5.5 trillion tokens including source code

## Demo

Below is a demo of the agent swarm in action:

![Demo](./demo.gif)