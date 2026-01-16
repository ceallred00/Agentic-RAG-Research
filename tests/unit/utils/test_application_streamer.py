import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from utils.application_streamer import application_streamer

@pytest.fixture
def mock_langgraph_app():
    app = MagicMock()

    fake_event = {
        'node_name': {
            'messages': [AIMessage(content="Test Response")]
        }
    }

    app.stream.return_value = [fake_event]

    return app

@pytest.fixture
def run_config():
    return {"configurable": {"thread_id": "1"}}


class TestApplicationStreamer:
    """ Test suite for the application_streamer utility function."""
    def test_happy_path(self, mock_langgraph_app, run_config):
        input_text = "Hello World"

        # Execute test

        results = application_streamer(
            application = mock_langgraph_app,
            user_input = input_text,
            configuration = run_config, #type:ignore
            stream_mode = "updates"
        )

        expected_data = mock_langgraph_app.stream.return_value
        assert list(results) == expected_data

        mock_langgraph_app.stream.assert_called_once_with(
            {"messages": [HumanMessage(content=input_text)]},
            config = run_config, 
            stream_mode = "updates"
        )
    
    def test_error_handling(self, mock_langgraph_app, run_config, caplog):
        mock_langgraph_app.stream.side_effect = Exception("Test Error")
        input_text = "Hello World"

        results = application_streamer(
            application = mock_langgraph_app,
            user_input = input_text,
            configuration = run_config, #type:ignore
            stream_mode = "updates"
        )
        assert list(results) == [{"error": "Test Error"}]
        assert "Error in application_streamer:" in caplog.text






