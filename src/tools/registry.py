import copy
import logging
from functools import cache
from typing import Callable, Dict, Any, List


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, name: str, description: str, parameters: dict):
        """Decorator to register a tool function with its metadata."""

        def decorator(func: Callable):
            if name in self._tools:
                logging.warning(f'Tool {name} is already registered. Overwriting.')
            
            self._tools[name] = func
            self._schemas.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': description,
                    'parameters': parameters
                }
            })
            self.get_schemas.cache_clear()  # Clear cache to update schemas
            return func
        return decorator

    @cache
    def get_schemas(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._schemas)

    def execute(self, name: str, kwargs: dict):
        if name not in self._tools:
            error_msg = f'未注册的工具调用: {name}'
            logging.error(error_msg)
            return error_msg

        try:
            return self._tools[name](**kwargs)

        except Exception as e:
            logging.exception(f'执行工具 {name} 时发生异常: {e}')
            return f'工具执行异常: {e}'


registry = ToolRegistry()
