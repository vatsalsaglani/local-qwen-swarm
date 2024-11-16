import os
import json
from openai import AsyncOpenAI
from typing import List, Dict, AsyncGenerator, Optional
import re
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)


class LocalLLM:

    def __init__(self, base_url: str, api_key: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def _extract_function_calls(self, response: str) -> List[Dict]:
        pattern = r'<functioncall>\s*(.*?)\s*</functioncall>'
        function_calls = re.findall(pattern, response)

        parsed_calls = []
        for call in function_calls:
            try:
                parsed_calls.append(json.loads(call))
            except json.JSONDecodeError:
                continue
        return parsed_calls

    async def completion(self, model: str, messages: List[Dict], **kwargs):
        response = await self.client.chat.completions.create(model=model,
                                                             messages=messages,
                                                             **kwargs)
        response = response.choices[0].message.content
        # print("completion response\n", response)
        return response

    def _validate_parameters(self, provided_params: Dict,
                             expected_params: Dict) -> List[str]:
        """Validate parameters against schema and return list of validation errors."""
        validation_errors = []

        required_params = expected_params.get('required', [])
        properties = expected_params.get('properties', {})

        missing_params = [
            p for p in required_params if p not in provided_params
        ]
        if missing_params:
            validation_errors.append(
                f"Missing required parameters: {missing_params}")

        for param_name, param_value in provided_params.items():
            if param_name not in properties:
                validation_errors.append(f"Unexpected parameter: {param_name}")
                continue

            expected_type = properties[param_name].get('type')
            if expected_type == 'array' and not isinstance(param_value, list):
                validation_errors.append(
                    f"Parameter {param_name} should be an array")
            elif expected_type == 'integer' and not isinstance(
                    param_value, int):
                validation_errors.append(
                    f"Parameter {param_name} should be an integer")
            elif expected_type == 'string' and not isinstance(
                    param_value, str):
                validation_errors.append(
                    f"Parameter {param_name} should be a string")
            elif expected_type == 'number' and not isinstance(
                    param_value, (int, float)):
                validation_errors.append(
                    f"Parameter {param_name} should be a number")
            elif expected_type == 'boolean' and not isinstance(
                    param_value, bool):
                validation_errors.append(
                    f"Parameter {param_name} should be a boolean")

        return validation_errors

    async def _recursive_function_call(self,
                                       model: str,
                                       messages: List[Dict],
                                       functions: List[Dict],
                                       validate_params: bool = True,
                                       depth: int = 0,
                                       max_depth: int = 5) -> List[Dict]:
        """Recursive helper for function_call with depth limit."""
        if depth >= max_depth:
            logger.warning("Max depth %d reached, returning empty list",
                           max_depth)
            return []

        response = await self.completion(model=model,
                                         messages=messages,
                                         temperature=0.2)
        extracted_functions = self._extract_function_calls(response)
        logger.debug("Extracted functions: %s", extracted_functions)

        if not extracted_functions:
            logger.info("No function calls found, trying again")
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. You did not select any function. Please select one appropriate function."
            return await self._recursive_function_call(model, messages,
                                                       functions, depth + 1,
                                                       max_depth)

        if len(extracted_functions) > 1:
            logger.info("Multiple function calls found, trying again")
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. You selected multiple functions. Please select only one appropriate function."
            return await self._recursive_function_call(model, messages,
                                                       functions, depth + 1,
                                                       max_depth)

        function_call = extracted_functions[0]
        function_name = function_call.get('name')

        target_function = next(
            (f for f in functions if f['name'] == function_name), None)
        if not target_function:
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. The function name is not valid. Please choose the most appropriate function."
            return await self._recursive_function_call(model, messages,
                                                       functions, depth + 1,
                                                       max_depth)

        validation_errors = None
        if validate_params:
            validation_errors = self._validate_parameters(
                function_call.get('parameters', {}),
                target_function.get('parameters', {}))

        if validation_errors:
            logger.warning("Validation errors: %s", validation_errors)
            error_message = "\n".join(validation_errors)
            messages[-1][
                "content"] += f"\n\nThe following is your output: {response}. The parameters are not valid: {error_message}. Please select the most appropriate function with correct parameters."
            return await self._recursive_function_call(model, messages,
                                                       functions, depth + 1,
                                                       max_depth)

        return extracted_functions

    async def function_call(self,
                            model: str,
                            messages: List[Dict],
                            functions: List[Dict],
                            validate_params: bool = True) -> List[Dict]:
        """Main function call method with validation and recursion."""
        try:
            function_call_prompt = """You are a helpful AI assistant with access to a set of functions. Your task is to assist users by utilizing these functions to respond to their requests. Here are your instructions:

            1. Available Functions:
            <functions>
            {functions}
            </functions>

            2. Using Functions:
            - You MUST provide exactly ONE function call per response
            - Your response MUST be enclosed in <functioncall> tags
            - Parameters MUST match the function's schema exactly
            - Do not provide any additional text or explanations
            - For multi-step tasks, determine which step is currently needed based on the conversation history
            - When all steps are complete, use the 'end' function

            3. Format:
            <functioncall> {{"name": "functionName", "parameters": {{"param1": "value1", "param2": "value2"}} }} </functioncall>

            4. Few-Shot Examples:

            User: "First check the weather in London, then book a taxi"
            Assistant: <functioncall> {{"name": "getWeather", "parameters": {{"city": "London"}} }} </functioncall>

            System: Weather in London is 15°C and sunny
            User: "First check the weather in London, then book a taxi"
            Assistant: <functioncall> {{"name": "bookTaxi", "parameters": {{"pickup": "London"}} }} </functioncall>

            User: "Convert 100F to Celsius and then send it to admin@example.com"
            Assistant: <functioncall> {{"name": "convertTemperature", "parameters": {{"value": 100, "from": "F", "to": "C"}} }} </functioncall>

            System: Temperature converted to 37.8C
            User: "Convert 100F to Celsius and then send it to admin@example.com"
            Assistant: <functioncall> {{"name": "sendEmail", "parameters": {{"to": "admin@example.com", "body": "Temperature is 37.8C"}} }} </functioncall>

            5. Important:
            - Examine the conversation history to determine which step needs to be executed next
            - Return exactly ONE function call that matches the current step
            - If all steps are complete, use the 'end' function with appropriate parameters
            - Never include multiple function calls in one response
            - Return the function call in <functioncall> tags as shown in the examples above

            Remember: Only ONE function call is allowed per response. Determine the current step from the conversation history and return only one function call.
            """.format(functions=functions)

            _messages = [{
                "role": "system",
                "content": function_call_prompt
            }] + messages
            logger.debug("Function call messages: %s", _messages)

            return await self._recursive_function_call(
                model,
                _messages,
                functions,
                validate_params=validate_params,
                depth=0)

        except Exception as e:
            logger.error("Error during function call: %s", str(e))
            return []

    async def stream(self, model: str,
                     messages: List[Dict]) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, stream=True)
            async for chunk in response:
                if chunk.choices[0].delta and chunk.choices[
                        0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error("Error during streaming: %s", str(e))
            yield f"Error during streaming: {str(e)}"


if __name__ == "__main__":
    import asyncio
    llm = LocalLLM("http://localhost:1234/v1", "your-api")
    functions = [
        {
            "name": "getWeather",
            "description": "Get the weather for a specific city."
        },
        {
            "name": "getNews",
            "description": "Get the latest news headlines."
        },
        {
            "name": "getStockPrice",
            "description": "Get the stock price for a specific company."
        },
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather in New York today?"
        },
        # {"role": "user", "content": "Can you tell me the latest news headlines?"},
    ]
    response = asyncio.run(
        llm.function_call(model="qwen2.5-coder-3b-instruct-q4_k_m",
                          messages=messages,
                          functions=functions))
    print(response)
