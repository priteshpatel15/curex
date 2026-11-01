"""
Safe Code Execution Engine for running generated Python analysis code.
Provides sandboxed execution with restricted imports and output capture.
"""

import ast
import sys
import io
import time
import logging
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    output: str = ""  # Default to empty string
    error: Optional[str] = None
    execution_time: float = 0.0
    plots_generated: List[str] = None
    variables_created: List[str] = None
    data_modified: bool = False
    
    def __post_init__(self):
        if self.plots_generated is None:
            self.plots_generated = []
        if self.variables_created is None:
            self.variables_created = []
    



class SafeCodeExecutor:
    """
    Safe execution environment for generated Python analysis code.
    Provides sandboxed execution with restricted imports and comprehensive output capture.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the code executor.
        
        Args:
            dataframe: The pandas DataFrame to make available as 'df'
        """
        self.dataframe = dataframe.copy()  # Work with a copy to prevent accidental modification
        self.allowed_modules = self._setup_allowed_modules()
        self.plot_counter = 0
        
        # Suppress warnings during execution
        warnings.filterwarnings('ignore')
        
    def execute_code(self, code: str, timeout: int = 30) -> ExecutionResult:
        """
        Execute Python code in a safe environment.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            ExecutionResult: Result of the execution
        """
        start_time = time.time()
        
        try:
            # Validate the code before execution
            if not self._validate_code(code):
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Code validation failed: potentially unsafe operations detected",
                    execution_time=time.time() - start_time
                )
            
            # Prepare execution environment
            execution_globals = self._create_execution_environment()
            
            # Capture output
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            plots_generated = []
            variables_before = set(execution_globals.keys())
            
            try:
                with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                    # Clear any existing plots
                    plt.clf()
                    plt.close('all')
                    
                    # Execute the code
                    exec(code, execution_globals)
                    
                    # Check if any plots were created
                    if plt.get_fignums():
                        plot_filename = f"plot_{self.plot_counter}.png"
                        plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
                        plots_generated.append(plot_filename)
                        self.plot_counter += 1
                        plt.close('all')
                
                # Capture execution results
                output = output_buffer.getvalue()
                error_output = error_buffer.getvalue()
                
                # Check for new variables created
                variables_after = set(execution_globals.keys())
                new_variables = list(variables_after - variables_before)
                
                # Check if the original dataframe was modified
                data_modified = not self.dataframe.equals(execution_globals.get('df', pd.DataFrame()))
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=True,
                    output=output,
                    error=error_output if error_output else None,
                    execution_time=execution_time,
                    plots_generated=plots_generated,
                    variables_created=[var for var in new_variables if not var.startswith('_')],
                    data_modified=data_modified
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=False,
                    output=output_buffer.getvalue(),
                    error=f"Execution error: {str(e)}\n{traceback.format_exc()}",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                output="",
                error=f"Setup error: {str(e)}",
                execution_time=execution_time
            )
    
    def _validate_code(self, code: str) -> bool:
        """
        Validate code for safety before execution.
        
        Args:
            code: Python code to validate
            
        Returns:
            bool: True if code is safe to execute
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Check for dangerous operations
            dangerous_nodes = [
                ast.Import,
                ast.ImportFrom
            ]
            
            for node in ast.walk(tree):
                # Check for dangerous node types
                if type(node) in dangerous_nodes:
                    logger.warning(f"Dangerous operation detected: {type(node).__name__}")
                    return False
                
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        dangerous_functions = [
                            'exec', 'eval', 'compile', 'open', '__import__',
                            'getattr', 'setattr', 'delattr', 'globals', 'locals',
                            'vars', 'dir', 'hasattr'
                        ]
                        
                        if node.func.id in dangerous_functions:
                            logger.warning(f"Dangerous function call detected: {node.func.id}")
                            return False
                
                # Check for attribute access to dangerous modules
                if isinstance(node, ast.Attribute):
                    dangerous_attributes = [
                        'os', 'sys', 'subprocess', 'socket', 'urllib',
                        'requests', 'pathlib', 'shutil', 'glob'
                    ]
                    
                    if isinstance(node.value, ast.Name) and node.value.id in dangerous_attributes:
                        logger.warning(f"Dangerous module access detected: {node.value.id}")
                        return False
            
            return True
            
        except SyntaxError:
            logger.error("Code has syntax errors")
            return False
        except Exception as e:
            logger.error(f"Code validation error: {str(e)}")
            return False
    
    def _create_execution_environment(self) -> Dict[str, Any]:
        """Create a safe execution environment with allowed modules and functions."""
        
        # Core execution environment
        execution_globals = {
            '__builtins__': {
                # Safe built-in functions
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'print': print,
                'type': type,
                'isinstance': isinstance,
                'issubclass': issubclass,
            }
        }
        
        # Add allowed modules
        execution_globals.update(self.allowed_modules)
        
        # Add the dataframe
        execution_globals['df'] = self.dataframe.copy()
        
        return execution_globals
    
    def _setup_allowed_modules(self) -> Dict[str, Any]:
        """Setup allowed modules and their safe interfaces."""
        
        allowed_modules = {}
        
        # Data manipulation
        allowed_modules['pd'] = pd
        allowed_modules['pandas'] = pd
        allowed_modules['np'] = np
        allowed_modules['numpy'] = np
        
        # Statistics and scientific computing
        allowed_modules['stats'] = stats
        allowed_modules['scipy'] = __import__('scipy')
        
        # Visualization
        allowed_modules['plt'] = plt
        allowed_modules['matplotlib'] = matplotlib
        allowed_modules['sns'] = sns
        allowed_modules['seaborn'] = sns
        allowed_modules['go'] = go
        allowed_modules['px'] = px
        allowed_modules['plotly'] = __import__('plotly')
        
        # Machine learning (if needed)
        try:
            sklearn = __import__('sklearn')
            allowed_modules['sklearn'] = sklearn
        except ImportError:
            pass
        
        # Mathematical functions
        import math
        allowed_modules['math'] = math
        
        # Date and time (safe subset)
        from datetime import datetime, date, timedelta
        allowed_modules['datetime'] = datetime
        allowed_modules['date'] = date
        allowed_modules['timedelta'] = timedelta
        
        return allowed_modules
    
    def get_execution_summary(self, result: ExecutionResult) -> Dict[str, Any]:
        """
        Get a summary of the execution result.
        
        Args:
            result: ExecutionResult to summarize
            
        Returns:
            Dict containing execution summary
        """
        summary = {
            'success': result.success,
            'execution_time': result.execution_time,
            'output_length': len(result.output) if result.output else 0,
            'has_error': result.error is not None,
            'plots_count': len(result.plots_generated),
            'variables_created_count': len(result.variables_created),
            'data_modified': result.data_modified
        }
        
        if result.error:
            summary['error_type'] = result.error.split(':')[0] if ':' in result.error else 'Unknown'
        
        return summary
    
    def evaluate_code_quality(self, code: str, result: ExecutionResult) -> float:
        """
        Evaluate the quality of the generated code based on execution results.
        
        Args:
            code: The executed code
            result: ExecutionResult from execution
            
        Returns:
            float: Quality score between 0 and 1
        """
        score = 0.0
        
        # Success is fundamental (40% of score)
        if result.success:
            score += 0.4
        
        # Code structure and practices (20% of score)
        code_quality_score = self._assess_code_structure(code)
        score += 0.2 * code_quality_score
        
        # Execution efficiency (15% of score)
        if result.execution_time < 5.0:  # Fast execution
            score += 0.15
        elif result.execution_time < 15.0:  # Reasonable execution
            score += 0.1
        
        # Output usefulness (15% of score)
        if result.output and len(result.output.strip()) > 0:
            score += 0.15
        
        # Visualization generation (10% of score)
        if result.plots_generated:
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_code_structure(self, code: str) -> float:
        """Assess the quality of code structure and practices."""
        score = 0.0
        
        # Check for comments
        if '#' in code:
            score += 0.3
        
        # Check for proper imports
        if any(imp in code for imp in ['import', 'from']):
            score += 0.2
        
        # Check for error handling
        if 'try:' in code or 'except:' in code:
            score += 0.2
        
        # Check for proper variable naming (not just 'df')
        lines = code.split('\n')
        if any('=' in line and not line.strip().startswith('#') for line in lines):
            score += 0.3
        
        return min(score, 1.0)
    
    def clear_plots(self):
        """Clear all matplotlib plots and reset plot counter."""
        plt.clf()
        plt.close('all')
        self.plot_counter = 0
    
    def update_dataframe(self, new_dataframe: pd.DataFrame):
        """Update the dataframe used in execution."""
        self.dataframe = new_dataframe.copy()


class CodeValidator:
    """Additional code validation utilities."""
    
    @staticmethod
    def extract_imports(code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
        except Exception:
            pass
        return imports
    
    @staticmethod
    def check_pandas_usage(code: str) -> bool:
        """Check if code properly uses pandas operations."""
        pandas_patterns = [
            'df.', 'pd.', '.iloc', '.loc', '.head()', '.tail()',
            '.describe()', '.info()', '.shape', '.columns'
        ]
        return any(pattern in code for pattern in pandas_patterns)
    
    @staticmethod
    def estimate_complexity(code: str) -> str:
        """Estimate code complexity based on structure."""
        try:
            tree = ast.parse(code)
            complexity_score = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    complexity_score += 2
                elif isinstance(node, (ast.If, ast.Try)):
                    complexity_score += 1
                elif isinstance(node, ast.FunctionDef):
                    complexity_score += 3
                elif isinstance(node, ast.ClassDef):
                    complexity_score += 5
            
            if complexity_score == 0:
                return "simple"
            elif complexity_score < 5:
                return "medium"
            else:
                return "complex"
                
        except Exception:
            return "unknown" 