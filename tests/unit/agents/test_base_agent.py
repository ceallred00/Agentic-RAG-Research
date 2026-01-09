# """Unit tests for base agent. Example provided by AI assistant."""

# import pytest

# from src.agents.base_agent import BaseAgent


# class ConcreteAgent(BaseAgent):
#     """Concrete implementation of BaseAgent for testing."""

#     def think(self, input_data):
#         """Simple thinking logic."""
#         return f"Thought: {input_data}"

#     def act(self, decision):
#         """Simple action logic."""
#         return f"Action: {decision}"


# class TestBaseAgent:
#     """Test suite for BaseAgent class."""

#     def test_agent_initialization(self, sample_config):
#         """Test agent initialization."""
#         agent = ConcreteAgent(name="test_agent", config=sample_config)
#         assert agent.name == "test_agent"
#         assert agent.config == sample_config

#     def test_think_method(self):
#         """Test think method."""
#         agent = ConcreteAgent(name="test_agent")
#         result = agent.think("input_data")
#         assert "Thought" in result
#         assert "input_data" in result

#     def test_act_method(self):
#         """Test act method."""
#         agent = ConcreteAgent(name="test_agent")
#         result = agent.act("decision")
#         assert "Action" in result
#         assert "decision" in result

#     def test_run_cycle(self):
#         """Test complete think-act cycle."""
#         agent = ConcreteAgent(name="test_agent")
#         result = agent.run("test_input")
#         assert "Action" in result
#         assert "Thought" in result
