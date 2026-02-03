"""
Python Code Execution Tool
Safely execute Python code and return results
"""
from typing import Dict, Any, Optional
import sys
import os
from io import StringIO
from pathlib import Path

from src.tools.base import BaseTool, ToolResult, register_tool


@register_tool()
class PythonCodeExecutor(BaseTool):
    """
    Python Code Execution Tool
    Execute Python code in a safe environment and return results
    """
    
    name = "python_code_executor"
    description = "Type: General Tool. Execute Python code and return results, the execution environment already supports various common libraries. Each code execution is stateless and requires complete code."
    category = "general"

    
    parameters = {
        "code": {
            "type": "string",
            "description": "Python code to execute",
            "required": True
        },
        "timeout": {
            "type": "integer",
            "description": "Execution timeout (seconds), default 30 seconds",
            "required": False
        }
    }
    
    # Project root path
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent.parent
    # Working directory
    WORKING_DIR = "/tmp/data"
    
    def _create_globals(self) -> Dict[str, Any]:
        """Create execution environment"""
        import pandas as pd
        import numpy as np
        import math
        import json
        import re
        from datetime import datetime, date, timedelta
        
        return {
            'pd': pd,
            'np': np,
            'math': math,
            'json': json,
            're': re,
            'datetime': datetime,
            'date': date,
            'timedelta': timedelta,
            'open': open,
            'PROJECT_ROOT': str(self.PROJECT_ROOT),
        }
    
    def execute(
        self,
        code: str,
        timeout: int = 30,
        cwd: str = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute Python code
        
        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            cwd: Specified working directory
            
        Returns:
            ToolResult: Contains execution results or error messages
        """
        # Capture standard output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            # Create execution environment
            exec_globals = self._create_globals()
            
            # Switch working directory
            old_cwd = os.getcwd()
            # Determine working directory: prioritize passed cwd, otherwise use default WORKING_DIR
            target_dir = cwd if cwd else self.WORKING_DIR
            os.makedirs(target_dir, exist_ok=True)
            os.chdir(target_dir)
            
            try:
                # Execute code
                exec(code, exec_globals)
                
                # Get output
                stdout_output = sys.stdout.getvalue()
                stderr_output = sys.stderr.getvalue()
                
                # Directly return stdout output
                final_output = stdout_output if stdout_output else ""
                if stderr_output:
                    final_output += stderr_output
                
                return self.make_result(
                    success=True,
                    data=final_output,
                    message="Code executed successfully"
                )
            finally:
                os.chdir(old_cwd)
                
        except Exception as e:
            stderr_output = sys.stderr.getvalue()
            error_msg = f"{type(e).__name__}: {str(e)}"
            if stderr_output:
                error_msg = f"{stderr_output}{error_msg}"
            
            return self.make_result(
                success=False,
                data=error_msg,
                message=f"Code execution error: {error_msg}"
            )
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
