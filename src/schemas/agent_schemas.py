"""Defines Pydantic models for agent configurations."""

from pydantic import BaseModel, Field
from typing import Annotated, Literal


class AgentModelConfig(BaseModel):
    """
    Configuration parameters for the LLM model used by the agent.
    """

    provider: Annotated[Literal["google"], Field(description="The model provider.")]
    name: Annotated[str, Field(description="The name of the model, e.g., 'gemini-3-pro-preview'.")]
    temperature: Annotated[
        float,
        Field(
            ge=0.0,
            le=2.0,
            description="The temperature setting for the model. 0.0 is deterministic, 1.0 is considered medium, higher values increase randomness.",
        ),
    ]


class AgentMetadata(BaseModel):
    """
    Descriptive metadata about the agent.
    """

    name: Annotated[str, Field(description="The name of the agent.")]
    description: Annotated[str, Field(description="A brief description of the agent.")]


# TODO: Expand with additional agent configuration schemas as needed (Tools, etc.)


class AgentConfig(BaseModel):
    """
    Root coniguration model merging metadata, model config, and other settings.
    """

    version: Annotated[str, Field(description="Agent configuration version for compatibility checks.")] = "1.0"
    agent_metadata: Annotated[AgentMetadata, Field(description="Metadata about the agent.")]
    model: Annotated[
        AgentModelConfig,
        Field(description="LLM technical configuration for the agent."),
    ]
    system_prompt: Annotated[str, Field(description="The system prompt for the agent.")]
