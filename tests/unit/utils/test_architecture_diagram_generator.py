import pytest
from utils.architecture_diagram_generator import ArchitectureDiagramGenerator
from unittest.mock import MagicMock

@pytest.fixture
def valid_diagrams_dir(tmp_path):
    """ Creates a real temporary directory for the test."""
    valid_dir = tmp_path / "diagrams"
    valid_dir.mkdir()

    return valid_dir

@pytest.fixture
def mocked_application_graph():
    """ Creates a 'fake' graph object.
    When the production code calls: app.get_graph().draw_mermaid_png()
    This mock will return: b"fake_image_bytes"
    """
    mock_app = MagicMock()
    mock_app.get_graph.return_value.draw_mermaid_png.return_value = b"fake_image_bytes"
    return mock_app

class TestArchitectureDiagramGenerator:
    """ Test suite for generating architectural diagrams."""
    class TestGenerateGraphDiagram:
        """ Test suite for generating architectural diagrams from Compiled LangGraph app."""
        def test_happy_path(self, valid_diagrams_dir, mocked_application_graph):
            """
            Test creating graph diagram with valid directory path and valid function calls.

            Test should complete successfully, creating a "test_diagram.png" and storing it in the valid diagrams directory.

            Happy path test.
            """
            generator = ArchitectureDiagramGenerator(valid_diagrams_dir)
            filename = "test_diagram.png"

            generator.generate_graph_diagram(diagram_name = filename, application_graph = mocked_application_graph)

            expected_file = valid_diagrams_dir / filename

            assert expected_file.exists()

            assert expected_file.read_bytes() == b"fake_image_bytes"

        def test_nonexistent_directory(self, valid_diagrams_dir, mocked_application_graph):
            """ Verifies function behavior if the specified directory does not exist."""
            # Point to a path which does NOT exist.
            ghost_dir = valid_diagrams_dir / "ghost_folder"
            generator = ArchitectureDiagramGenerator(ghost_dir)

            with pytest.raises(FileNotFoundError) as excinfo:
                generator.generate_graph_diagram(diagram_name = "test_diagram.png", application_graph= mocked_application_graph)
            
            assert "Diagrams directory not found" in str(excinfo.value)
        def test_image_error(self, valid_diagrams_dir, mocked_application_graph):
            """ 
            Verifies function behavior if .get_graph().draw_mermaid_png() were to fail.

            .draw_mermaid_png() relies on an external API call, thus generating a potential failure point.

            The test should raise a RuntimeError.
            """
            generator = ArchitectureDiagramGenerator(valid_diagrams_dir)

            mocked_application_graph.get_graph.return_value.draw_mermaid_png.side_effect = Exception("API call failed")

            with pytest.raises(RuntimeError) as excinfo:
                generator.generate_graph_diagram(diagram_name = "test_diagram.png", application_graph= mocked_application_graph)
            
            assert "API call failed" in str(excinfo.value)
        def test_incorrect_file_extension(self, valid_diagrams_dir, mocked_application_graph):
            """
            The filename passed by the user should include the .png extension
            
            This test checks the output if the user passes an incorrect file extension (e.g., .txt).

            The function should remove the incorrect extension and create the file with the correct extension.

            The test should succeed. 
            """
            generator = ArchitectureDiagramGenerator(valid_diagrams_dir)

            incorrect_filename = "test_diagram.txt"
            correct_filename = "test_diagram.png"

            expected_file = valid_diagrams_dir / correct_filename

            generator.generate_graph_diagram(diagram_name=incorrect_filename, application_graph=mocked_application_graph)

            assert expected_file.exists()
            assert expected_file.read_bytes() == b"fake_image_bytes"
        def test_no_file_extension(self, valid_diagrams_dir, mocked_application_graph):
            """
            The filename passed by the user should include the .png extension
            
            This test checks the output if the user passes a diagram_name with no file extension.

            The function should attached the .png extension to the diagram_name and create the correct file.

            The test should succeed. 
            """
            generator = ArchitectureDiagramGenerator(valid_diagrams_dir)

            incorrect_filename = "test_diagram"
            correct_filename = "test_diagram.png"

            expected_file = valid_diagrams_dir / correct_filename

            generator.generate_graph_diagram(diagram_name=incorrect_filename, application_graph=mocked_application_graph)

            assert expected_file.exists()
            assert expected_file.read_bytes() == b"fake_image_bytes"
