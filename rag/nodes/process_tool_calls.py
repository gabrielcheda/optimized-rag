"""
Process Tool Calls Node
Executes tool calls requested by the LLM
"""

import logging
from typing import Dict, Any
from agent.state import MemGPTState

logger = logging.getLogger(__name__)


def process_tool_calls_node(state: MemGPTState, agent) -> Dict[str, Any]:
    """
    Process and execute tool calls from LLM response
    
    Executes tools requested by the LLM and appends results to messages.
    """
    if not state.tool_calls or len(state.tool_calls) == 0:
        logger.debug("No tool calls to process")
        return {}
    
    logger.info(f"Processing {len(state.tool_calls)} tool calls")
    
    tool_results = []
    
    # Create tool map for fast lookup
    tool_map = {tool.name: tool for tool in agent.all_tools}
    
    for tool_call in state.tool_calls:
        tool_name = tool_call.get('name', '')
        tool_args = tool_call.get('args', {})
        tool_id = tool_call.get('id', '')
        
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        try:
            if tool_name not in tool_map:
                result = f"Error: Tool '{tool_name}' not found"
                logger.error(result)
            else:
                # Execute the tool
                tool = tool_map[tool_name]
                result = tool.invoke(tool_args)
                logger.info(f"Tool {tool_name} executed successfully")
            
            tool_results.append({
                'tool_call_id': tool_id,
                'tool_name': tool_name,
                'result': result
            })
            
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            tool_results.append({
                'tool_call_id': tool_id,
                'tool_name': tool_name,
                'result': error_msg
            })
    
    # Append tool results to messages for context
    tool_messages = []
    for tr in tool_results:
        tool_messages.append({
            'role': 'tool',
            'tool_call_id': tr['tool_call_id'],
            'name': tr['tool_name'],
            'content': str(tr['result'])
        })
    
    return {
        'tool_results': tool_results,
        'messages': state.messages + tool_messages
    }
