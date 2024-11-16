import json
from agent._types import *
from llm.local import LocalLLM
from copy import deepcopy
import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalSwarm:

    def __init__(self,
                 orchestrator_agent: Agent,
                 agents: List[Agent],
                 messages: List[Dict],
                 local_api_base_url: str = "http://localhost:1234/v1",
                 local_api_key: str = "your-api",
                 max_iterations: int = 10):
        self.orchestrator_agent = orchestrator_agent
        self.client = LocalLLM(local_api_base_url, local_api_key)
        self.agents = agents.copy()
        self.orchestrator_agent.functions = [
            AgentFunction(name=a.name,
                          description=a.instruction,
                          parameters={
                              "properties": {
                                  "agent_name": {
                                      "type":
                                      "string",
                                      "description":
                                      "The name of the agent to call"
                                  }
                              }
                          },
                          _callable=a.name) for a in agents
        ]
        self.orchestrator_agent.functions += [
            AgentFunction(name="end",
                          description="End the agent once the task is done",
                          parameters=End.model_json_schema(),
                          _callable="end")
        ]
        self.function_map = {
            f.name: f
            for f in self.orchestrator_agent.functions
        }
        self.agent_map = {a.name: a for a in self.agents}
        self.messages = messages
        self.max_iteration = max_iterations
        self.curr_iteration = 0
        self.selected_agents = []

    async def _iterate(self, model, llm_args):
        curr_messages = deepcopy(self.messages)
        functions = list(
            map(
                lambda x: {
                    "name": x.name,
                    "description": x.description,
                    "parameters": x.parameters,
                }, self.orchestrator_agent.functions))
        logger.info("[iterate functions]\n%s\n\n=====", functions)
        response_functions = await self.client.function_call(
            model, curr_messages, functions, validate_params=False, **llm_args)
        logger.info("[iterate response_functions]\n%s\n\n=====",
                    response_functions)
        return response_functions

    async def _function_args(self,
                             model,
                             agent_name: str,
                             no_func_call: bool = False,
                             max_rec_depth: int = 3,
                             curr_depth: int = 0):
        functions = self.agent_map[agent_name].functions
        logger.info("[function_args functions]\n%s\n\n=====", functions)
        functions_list = list(
            map(
                lambda x: {
                    "name": x.get("name"),
                    "description": x.get("description"),
                    "parameters": x.get("parameters"),
                }, functions))
        curr_messages = deepcopy(self.messages)
        if no_func_call:
            curr_messages[-1][
                "content"] += "\n\nDidn't receive a function. Please return only one appropriate function."
        response = await self.client.function_call(model, curr_messages,
                                                   functions_list, **{})
        logger.info("[function_args response]\n%s\n\n=====", response)
        if len(response) == 0:
            curr_depth += 1
            if curr_depth > max_rec_depth:
                raise Exception(
                    f"Max recursion depth reached while waiting for function call from {agent_name}"
                )
            return await self._function_args(model, agent_name, no_func_call,
                                             max_rec_depth, curr_depth)
        return response[-1]

    async def _return_final_answer(self, model: str, end_output: Dict):
        final_answer_system_prompt = """You are a helpful AI assistant that explains the problem-solving process to users. You will receive:
            1. The original user request
            2. A list of steps taken by different agents
            3. The final output

            Your task is to create a clear, conversational explanation that helps users understand:
            - What their request was
            - Which agents were involved
            - What functions were used and why
            - The intermediate results
            - The final answer

            Format your response as follows:
            1. Brief restatement of what the user asked for
            2. Step-by-step breakdown of what happened
            3. Final result with explanation

            Example:

            Input:
            User Request: "Check temperature in London and notify if above 20C"
            Steps: [
                {"agent": "weather_agent", "function": "getWeather", "params": {"city": "London"}, "output": "22C"},
                {"agent": "notification_agent", "function": "sendNotification", "params": {"message": "Temperature alert"}, "output": "Notification sent"}
            ]
            Final Output: {"why": "Temperature check and notification complete", "final_answer": "Temperature was 22C, notification sent"}

            Response:
            I helped you check London's temperature and send a notification since it was above 20C. Here's what happened:

            1. Temperature Check:
            • Weather_agent checked London's current temperature
            • The temperature was 22°C

            2. Notification:
            • Since 22°C was above your threshold of 20°C
            • Notification_agent sent out an alert

            The process is complete - London is at 22°C and you've been notified.

            Remember to:
            - Use clear, conversational language
            - Break down complex steps into digestible pieces
            - Show the logical flow from one step to the next
            - Highlight key numbers and results
            - Explain why each step was necessary
            - Make technical processes understandable to non-technical users"""
        self.messages[-1]["content"] += f'\n\n{end_output.get("content")}'
        final_messages = [{
            "role": "system",
            "content": final_answer_system_prompt
        }] + [{
            "role":
            "user",
            "content":
            f'The following is the list of agents and functions used to solve the problem:\n{self.selected_agents}'
        }]
        print("=====[ANSWER]=====\n")
        async for chunk in self.client.stream(model, final_messages):
            yield chunk

    async def run_swarm(self, model: str, llm_args: Dict):
        done = False
        while self.curr_iteration < self.max_iteration + 1 and not done:
            # print("[iteration messages]\n", self.messages, "\n\n=====")
            response_functions = await self._iterate(model, llm_args)
            self.selected_agents.append(
                {"agent": response_functions[0].get('name')})
            func_op = {"role": "assistant", "content": ""}
            for functions_ in response_functions:
                if functions_.get('name') == "end":
                    print("End of the conversation")
                    print("[end]\n", functions_, "\n\n=====")
                    print("[end]\n",
                          json.dumps(functions_.get('parameters'), indent=4),
                          "\n\n=====")
                    done = functions_.get('parameters')
                    break
                if not functions_.get('name') in self.agent_map:

                    raise Exception(
                        f"Function {functions_.get('name')} not found in the orchestrator agent"
                    )
                else:
                    func_op[
                        "content"] += f"Selected Agent: {functions_.get('name')}\n"
                    function_args = await self._function_args(
                        model, functions_.get('name'), curr_depth=0)
                    if not "functions" in self.selected_agents[-1]:
                        self.selected_agents[-1]["functions"] = []
                    self.selected_agents[-1]["functions"].append({
                        "name":
                        function_args.get('name'),
                        "parameters":
                        function_args.get('parameters')
                    })
                    # print("[function_args]\n", function_args, "\n\n=====")
                    func_op[
                        "content"] += f"Function Name: {function_args.get('name')}\n"
                    func_op[
                        "content"] += f"Function Args: {function_args.get('parameters')}\n"
                    agent_functions = self.agent_map[functions_.get(
                        'name')].functions
                    # print("[agent_functions]\n", agent_functions, "\n\n=====")
                    agent_function = next(
                        (f for f in agent_functions
                         if f.get("name") == function_args.get('name')), None)
                    # print("[agent_function]\n", agent_function, "\n\n=====")
                    if not agent_function:
                        raise Exception(
                            f"Function {function_args.get('name')} not found in the agent {functions_.get('name')}"
                        )
                    else:
                        if "_callable" in agent_function and callable(
                                agent_function.get('_callable')):
                            results = agent_function.get('_callable')(
                                **function_args.get('parameters', {}))
                            self.selected_agents[-1]["functions"][-1][
                                "result"] = results
                            logger.info("[results callable]\n%s\n\n=====",
                                        results)
                        else:
                            results = str(agent_function.get('_callable'))
                            self.selected_agents[-1]["functions"][-1][
                                "result"] = results
                            logger.info("[results else]\n%s\n\n=====", results)
                    func_op["content"] += f"Function Results: {results}\n"
                    self.messages[-1]["content"] += f'\n\n{func_op["content"]}'
            self.curr_iteration += 1
            # self.messages.append(func_op)
        if done:
            async for chunk in self._return_final_answer(model, done):
                print(chunk, end="", flush=True)
        print("\n=====[END]=====\n")
        return self.messages
