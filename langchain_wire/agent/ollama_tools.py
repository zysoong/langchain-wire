from typing import Sequence
from langchain.agents.output_parsers.ollama_tools import OllamaToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.language_models import BaseLanguageModel
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

from langchain_wire.injector.injector import PromptInjector


def create_ollama_tools_agent(
        llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    """Create an agent that uses Ollama tools.
    The prompts will be injected to the tools
    automatically.
    Check the documentation of 'create_openai_tools_agent'
    for detailed instructions.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to. Prompts will be injected as 'prompt'
            attribute automatically.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                )
            )
            | prompt
            | llm_with_tools
            | OllamaToolsAgentOutputParser()
    )
    return agent


def create_ollama_tools_agent_and_inject_prompts(
        llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:
    """Create an agent that uses Ollama tools.
    The prompts will be injected to the tools
    automatically.
    Check the documentation of 'create_openai_tools_agent'
    for detailed instructions.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to. Prompts will be injected as 'prompt'
            attribute automatically.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.
    """
    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                )
            )
            | prompt
            | PromptInjector(inject_objects=tools, pass_on_injection_fail=True)
            | llm_with_tools
            | OllamaToolsAgentOutputParser()
    )
    return agent
