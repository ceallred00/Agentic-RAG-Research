from langgraph.graph.state import CompiledStateGraph
from constants import PROD_DIAGRAM_DIR
from pathlib import Path
from typing import Union

class ArchitectureDiagramGenerator():
    """ 
    This class generates architecture diagrams from compiled LangGraph applications.

    Class created for the possibility to extend and generate other diagrams in the future.
    """
    
    def __init__(self, base_path: Union[str, Path] = PROD_DIAGRAM_DIR):
        """ 
        Initialize the class with the file path to the desired directory.
        Defaults to the Production Diagram Directory
        """
        self.base_path = base_path
    
    def generate_graph_diagram(self, diagram_name: str, application_graph: CompiledStateGraph):
        """
        Generates and saves a mermaid PNG diagram of the graph.

        Args:
            diagram_name (str): The filename for the image (e.g. 'agent_v1.png'). Filename should include the extension.
            application_graph (CompiledStateGraph): The actual compiled LangGraph app.

        Raises: 
            FileNotFoundError: Diagram directory does not exist.
            RuntimeError: Error when generating the diagram. Most likely culprint is an API failure.
        """
        # Convert base_path to Path object. Check for str input.
        diagram_dir = Path(self.base_path)

        if not diagram_dir.exists():
            raise FileNotFoundError(f"Diagrams directory not found: {diagram_dir}")
        
        # Forces file extension to be .png (Prevents confusing file names)
        filename_path = Path(diagram_name).with_suffix(".png")
        output_path = diagram_dir / filename_path
        
        try:
            # .draw_mermaid_png() is an external API call. 
            png_bytes = application_graph.get_graph().draw_mermaid_png()
        except Exception as e:
            raise RuntimeError(f"Failed to generate diagram: {e}")
        
        # Overwrites existing image
        output_path.write_bytes(png_bytes)




