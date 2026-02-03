"""
Command Line Execution Tool
Executes commands in the current environment and returns results
"""
import subprocess
import re
from typing import Optional, List

from src.tools.base import BaseTool, ToolResult, register_tool


# Dangerous command blacklist (regex patterns)
DANGEROUS_PATTERNS = [
    # System destruction
    r'\brm\s+(-rf?|--recursive)\s+[/\\]',  # rm -rf /
    r'\bdel\s+[/\\]\*',                      # del /*
    r'\bformat\s+[a-zA-Z]:',                 # format C:
    r'\bmkfs\b',                             # mkfs
    r'\bdd\s+.*of=/dev/',                    # dd of=/dev/
    # Privilege escalation
    r'\bsudo\s+',                            # sudo
    r'\bsu\s+',                              # su
    r'\brunas\b',                            # runas (Windows)
    # Network attack
    r'\bnc\s+-[el]',                         # netcat reverse shell
    r'\bcurl\s+.*\|\s*(bash|sh)',           # curl | bash
    r'\bwget\s+.*\|\s*(bash|sh)',           # wget | bash
    # Sensitive information
    r'\bpasswd\b',                           # passwd
    r'\bshadow\b',                           # /etc/shadow
    # Process/Service
    r'\bkill\s+-9\s+1\b',                   # kill -9 1 (init)
    r'\bshutdown\b',                         # shutdown
    r'\breboot\b',                           # reboot
    r'\bhalt\b',                             # halt
    # Registry (Windows)
    r'\breg\s+(delete|add)\s+HKLM',         # reg delete HKLM
    # Fork bomb
    r':\s*\(\s*\)\s*\{',                    # :() { 
]


def is_dangerous_command(command: str) -> "tuple[bool, str]":
    """
    Check if a command is dangerous
    
    Returns:
        (is_dangerous, reason)
    """
    cmd_lower = command.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True, f"Matches dangerous pattern: {pattern}"
    # Extra check: delete root directory
    if re.search(r'(rm|del|rmdir)\s+.*[/\\]$', cmd_lower):
        return True, "Attempting to delete root directory"
    return False, ""


@register_tool()
class CmdExecutor(BaseTool):
    """
    Command Line Execution Tool
    Executes command line commands in the current environment and returns results
    """
    
    name = "cmd_executor"
    description = "Type: General Tool. Executes command line commands. Can be used for file browsing, searching, and file operations within the environment directory (specified in system prompt). Returns standard output and standard error. Dangerous commands will be rejected."
    category = "general"
    
    parameters = {
        "command": {
            "type": "string",
            "description": "The command line command to execute",
            "required": True
        },
        "cwd": {
            "type": "string",
            "description": "Working directory for command execution. Must be the environment directory or its subdirectory.",
            "required": True
        },
        "timeout": {
            "type": "integer",
            "description": "Command timeout in seconds, defaults to 30",
            "required": False
        }
    }
    
    def execute(
        self, 
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 30,
        **kwargs
    ) -> ToolResult:
        """
        Execute command line command
        
        Args:
            command: The command to execute
            cwd: Working directory
            timeout: Timeout in seconds
            
        Returns:
            ToolResult: Contains command execution results
        """
        # Safety check: reject dangerous commands
        is_dangerous, reason = is_dangerous_command(command)
        if is_dangerous:
            return ToolResult(
                success=False,
                message=f"⚠️ Dangerous command rejected: {reason}"
            )
        
        try:
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            # Build output (stdout, stderr)
            output_parts = []
            if result.stdout.strip():
                output_parts.append(result.stdout.strip())
            if result.stderr.strip():
                output_parts.append(f"[stderr]\n{result.stderr.strip()}")
            output = "\n".join(output_parts) if output_parts else "No output"
            
            # Construct a complete record including command and output, simulating terminal display
            terminal_like_output = f"{output}"

            # Determine execution status
            if result.returncode == 0:
                return self.make_result(
                    success=True,
                    data=terminal_like_output,
                    message=f"Command executed successfully (exit code: {result.returncode})"
                )
            else:
                return self.make_result(
                    success=False,
                    data=f"{terminal_like_output}",
                    message=f"Command execution failed (exit code: {result.returncode})\n{terminal_like_output}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Command execution error: {str(e)}"
            )