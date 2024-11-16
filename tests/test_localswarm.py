import asyncio
from pydantic import BaseModel, Field
from typing import List
from agent.localswarm import LocalSwarm
from agent._types import Agent


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

# print("local_add_agent\n", local_add_agent)

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

print(
    asyncio.run(
        local_swarm.run_swarm(model="qwen2.5-coder-3b-instruct-q4_k_m",
                              llm_args={})))
