from src.tools.registry import registry

# Import all tool modules to ensure their registration in the registry.
import src.tools.search_hn_db
import src.tools.search_web


def get_tool_schemas() -> list[dict]:
    """For LLM to get the schema of all registered tools for planning and decision-making."""
    return registry.get_schemas()


def run_tool_call(tool_name: str, arguments: dict) -> str:
    """For LLM to route and execute specific tools based on the provided arguments."""
    return registry.execute(tool_name, arguments)
