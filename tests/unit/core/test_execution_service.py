
import pytest

class TestExecutionService:
    """ Test suite for ExecutionService class. """
    class TestGetGeminiClient:
        """ Test cases for get_gemini_client method. """

        def test_happy_path(self, gemini_api_key_env, instance_agent_config_execution_service, mock_gemini_client):
            """ Test creating Gemini client with valid agent configuration.

            Happy path test. Test should succeed with valid agent configuration.
            """

            # Attempt to create Gemini client for a valid agent
            gemini_client = instance_agent_config_execution_service.get_gemini_client(agent_name="good_test_agent_2")

            mock_gemini_client.assert_called_once_with(
                model="gemini-3-small",
                temperature=0.7,
                api_key=gemini_api_key_env
            )
            assert gemini_client == mock_gemini_client.return_value
        
        def test_invalid_agent_name(self, gemini_api_key_env, instance_agent_config_execution_service):
            """ Test behavior when an invalid agent name is provided.
            
            The method should raise a ValueError indicating the agent is not found.
            """
            # Attempt to create Gemini client for an invalid agent
            with pytest.raises(ValueError) as ve:
                instance_agent_config_execution_service.get_gemini_client(agent_name="non_existent_agent")

            assert "Agent 'non_existent_agent' not found in loaded configurations" in str(ve.value)
        
        def test_missing_gemini_api_key(self, instance_agent_config_execution_service, monkeypatch):
            """ Test behavior when GEMINI_API_KEY environment variable is missing.
            
            The method should raise a ValueError indicating the missing API key.
            """

            # Remove GEMINI_API_KEY if it exists
            monkeypatch.delenv("GEMINI_API_KEY", raising=False)

            # Attempt to create Gemini client without GEMINI_API_KEY
            with pytest.raises(ValueError) as ve:
                instance_agent_config_execution_service.get_gemini_client(agent_name="good_test_agent_1")

            assert "GEMINI_API_KEY not found in environment variables." in str(ve.value)

        def test_gemini_client_creation_failure(self, gemini_api_key_env, instance_agent_config_execution_service, mock_gemini_client):
            """ Test behavior when Gemini client creation fails.
            
            The method should log an error and raise the exception.
            """

            # Configure the mock to raise an exception when instantiated
            mock_gemini_client.side_effect = Exception("Client creation failed")

            # Attempt to create Gemini client, expecting an exception
            with pytest.raises(Exception) as excinfo:
                instance_agent_config_execution_service.get_gemini_client(agent_name="good_test_agent_1")

            assert "Client creation failed" in str(excinfo.value)

        def test_missing_config_file(self, instance_execution_service, mock_gemini_client):
            """ Test behavior when no agent configurations are loaded.
            
            Agent configurations are required to create a Gemini LLM client. The method should raise a ValueError.
            """
            with pytest.raises(ValueError) as ve:
                instance_execution_service.get_gemini_client(agent_name="any_agent")

            assert "No agent configurations were loaded into ExecutionService." in str(ve.value)
    
    class TestGetEmbeddingClient:
        """ Test cases for get_embedding_client method. """
        
        MODEL_NAME = "gemini-embedding-001"

        @pytest.mark.parametrize("task_type", ["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"])
        def test_get_embedding_client_happy_paths(self, instance_execution_service, mock_gemini_dense_embedding_client, secret_gemini_api_key_env, task_type):
            dense_embedding_client = instance_execution_service.get_embedding_client(
                model_name=self.MODEL_NAME,
                task_type=task_type
            )
            assert dense_embedding_client == mock_gemini_dense_embedding_client.return_value

            mock_gemini_dense_embedding_client.assert_called_once_with(
                model=self.MODEL_NAME,
                task_type=task_type,
                api_key = secret_gemini_api_key_env,
                output_dimensionality = 768
            )
        def test_gemini_embedding_client_creation_failure(self, instance_execution_service, mock_gemini_dense_embedding_client, gemini_api_key_env):
            """ Test behavior when Gemini Dense Embedding client creation fails.
            
            The method should log an error and raise the exception.

            Passing gemini_api_key_env to ensure the API key is set in the environment. Needed for _validate_api_key method.
            """

            # Configure the mock to raise an exception when instantiated
            mock_gemini_dense_embedding_client.side_effect = Exception("Dense Embedding Client creation failed")

            # Attempt to create Dense Embedding client, expecting an exception
            with pytest.raises(Exception) as excinfo:
                instance_execution_service.get_embedding_client(
                    model_name="gemini-embedding-001",
                    task_type="RETRIEVAL_QUERY"
                )

            assert "Dense Embedding Client creation failed" in str(excinfo.value)
    class TestGetPineconeClient:
        """ Test cases for get_pinecone_client method. """
        def test_get_pinecone_client_happy_path(self, instance_execution_service, mock_pinecone_client, pinecone_api_key_env):
            """ Test creating Pinecone client with valid configuration.

            Happy path test. Test should succeed with valid configuration.
            """

            # Attempt to create Pinecone client
            pinecone_client = instance_execution_service.get_pinecone_client()

            mock_pinecone_client.assert_called_once_with(api_key = pinecone_api_key_env)
            assert pinecone_client == mock_pinecone_client.return_value
        def test_missing_pinecone_api_key(self, instance_execution_service, monkeypatch):
            """ Test behavior when PINECONE_API_KEY environment variable is missing."""
            
            # Remove PINECONE_API_KEY if it exists
            monkeypatch.delenv("PINECONE_API_KEY", raising=False)

            # Attempt to create Pinecone client without PINECONE_API_KEY
            with pytest.raises(ValueError) as ve:
                instance_execution_service.get_pinecone_client()
            
            assert "PINECONE_API_KEY not found in environment variables." in str(ve.value)
        def test_pinecone_client_creation_failure(self, instance_execution_service, mock_pinecone_client, pinecone_api_key_env):
            """ Test behavior when Pinecone client creation fails.
            
            The method should log an error and raise the exception.
            """

            # Configure the mock to raise an exception when instantiated
            mock_pinecone_client.side_effect = Exception("Pinecone Client creation failed")

            # Attempt to create Pinecone client, expecting an exception
            with pytest.raises(Exception) as excinfo:
                instance_execution_service.get_pinecone_client()

            assert "Pinecone Client creation failed" in str(excinfo.value)





            