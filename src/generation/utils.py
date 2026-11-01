"""
Utility functions for the code generation engine
"""

import ast
import re
from typing import Dict, List, Optional, Any, Set
import logging

logger = logging.getLogger(__name__)


class CodeOptimizer:
    """Optimizes generated code for performance and readability"""
    
    @staticmethod
    def optimize_pandas_operations(code: str) -> str:
        """Optimize pandas operations in generated code"""
        optimizations = [
            # Use vectorized operations instead of loops
            (r'for\s+\w+\s+in\s+df\.iterrows\(\):', 
             '# Vectorized operation (optimized):'),
            
            # Use .loc for better performance
            (r'df\[([\'"][^\'"]+ == [^]]+)\]', 
             r'df.loc[\1]'),
            
            # Use .query() for complex filtering
            (r'df\[(df\[[\'"][^\'"]+[\'"]]\s*[><=!]+\s*[^]]+)\s*&\s*(df\[[\'"][^\'"]+[\'"]]\s*[><=!]+\s*[^]]+)\]',
             r'df.query("\1 and \2")'),
        ]
        
        optimized_code = code
        for pattern, replacement in optimizations:
            optimized_code = re.sub(pattern, replacement, optimized_code)
        
        return optimized_code
    
    @staticmethod
    def add_memory_optimization(code: str, dataset_size_mb: float) -> str:
        """Add memory optimization hints for large datasets"""
        if dataset_size_mb > 100:  # > 100MB
            memory_hints = """
# Memory optimization for large dataset
import gc
gc.collect()  # Free up memory before processing
"""
            return memory_hints + code
        return code
    
    @staticmethod
    def add_progress_indicators(code: str, complexity: str) -> str:
        """Add progress indicators for complex operations"""
        if complexity == 'high':
            progress_code = """
from tqdm import tqdm
import sys
print("Processing... This may take a moment for large datasets.")
"""
            return progress_code + code
        return code


class TemplateValidator:
    """Validates code templates for correctness and security"""
    
    @staticmethod
    def validate_template_syntax(template: str) -> List[str]:
        """Validate template syntax and return any issues"""
        issues = []
        
        # Check for unmatched braces
        open_braces = template.count('{')
        close_braces = template.count('}')
        if open_braces != close_braces:
            issues.append(f"Unmatched braces: {open_braces} open, {close_braces} close")
        
        # Check for Python syntax in template placeholders
        placeholders = re.findall(r'\{([^}]+)\}', template)
        for placeholder in placeholders:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', placeholder):
                issues.append(f"Invalid placeholder name: {placeholder}")
        
        return issues
    
    @staticmethod
    def check_security_risks(template: str) -> List[str]:
        """Check for potential security risks in templates"""
        risks = []
        
        dangerous_patterns = [
            (r'import\s+os', "OS module import"),
            (r'import\s+subprocess', "Subprocess module import"),
            (r'exec\s*\(', "Dynamic code execution"),
            (r'eval\s*\(', "Dynamic evaluation"),
            (r'__import__', "Dynamic import"),
            (r'open\s*\(', "File operations"),
            (r'with\s+open', "File operations"),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, template, re.IGNORECASE):
                risks.append(f"Security risk: {description}")
        
        return risks


class CodeAnalyzer:
    """Analyzes generated code for various metrics"""
    
    @staticmethod
    def count_operations(code: str) -> Dict[str, int]:
        """Count different types of operations in the code"""
        operations = {
            'dataframe_operations': len(re.findall(r'df\.', code)),
            'plotting_operations': len(re.findall(r'plt\.', code)),
            'statistical_operations': len(re.findall(r'(mean|std|var|corr|describe)', code)),
            'groupby_operations': len(re.findall(r'groupby', code)),
            'filter_operations': len(re.findall(r'\[df\[', code)),
        }
        return operations
    
    @staticmethod
    def estimate_execution_time(code: str, dataset_rows: int) -> str:
        """Estimate execution time category based on code complexity and dataset size"""
        operations = CodeAnalyzer.count_operations(code)
        total_ops = sum(operations.values())
        
        # Base time estimation
        if total_ops <= 2 and dataset_rows < 10000:
            return "< 1 second"
        elif total_ops <= 5 and dataset_rows < 100000:
            return "1-5 seconds"
        elif operations['plotting_operations'] > 0:
            return "5-30 seconds"
        elif operations['groupby_operations'] > 0 and dataset_rows > 100000:
            return "30 seconds - 2 minutes"
        else:
            return "< 30 seconds"
    
    @staticmethod
    def identify_dependencies(code: str) -> Set[str]:
        """Identify all library dependencies in the code"""
        dependencies = set()
        
        # Find import statements
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            dependencies.update(matches)
        
        # Common implicit dependencies
        if 'df.' in code:
            dependencies.add('pandas')
        if 'np.' in code:
            dependencies.add('numpy')
        if 'plt.' in code:
            dependencies.add('matplotlib')
        if 'sns.' in code:
            dependencies.add('seaborn')
        if 'stats.' in code:
            dependencies.add('scipy')
        
        return dependencies


class CodeFormatter:
    """Formats and beautifies generated code"""
    
    @staticmethod
    def add_comments(code: str, intent: str, columns: List[str]) -> str:
        """Add descriptive comments to the code"""
        header_comment = f"""
# Generated analysis code
# Intent: {intent}
# Columns involved: {', '.join(columns) if columns else 'N/A'}
# Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        return header_comment + code
    
    @staticmethod
    def format_output_statements(code: str) -> str:
        """Improve output formatting in the code"""
        # Add separators for cleaner output
        formatted_code = code.replace(
            'print(result)',
            'print("\\n" + "="*50)\nprint(result)\nprint("="*50 + "\\n")'
        )
        
        return formatted_code
    
    @staticmethod
    def add_error_handling(code: str) -> str:
        """Add basic error handling to the code"""
        wrapped_code = f"""
try:
{__import__('textwrap').indent(code, '    ')}
except Exception as e:
    print(f"Error during analysis: {{e}}")
    print("Please check your data and try again.")
    raise
"""
        return wrapped_code


class PerformanceProfiler:
    """Profile code performance for optimization suggestions"""
    
    @staticmethod
    def suggest_optimizations(code: str, dataset_info: Dict[str, Any]) -> List[str]:
        """Suggest performance optimizations based on code and dataset characteristics"""
        suggestions = []
        
        dataset_rows = dataset_info.get('rows', 0)
        dataset_cols = dataset_info.get('columns', 0)
        
        # Large dataset optimizations
        if dataset_rows > 1000000:
            if 'groupby' in code:
                suggestions.append("Consider using chunking for groupby operations on large datasets")
            if 'df[df[' in code:
                suggestions.append("Use df.query() for better performance with large datasets")
            if 'plt.' in code:
                suggestions.append("Consider sampling data for plotting to improve performance")
        
        # Memory optimizations
        if dataset_cols > 50:
            suggestions.append("Consider selecting only necessary columns to reduce memory usage")
        
        # Computation optimizations
        if code.count('df.') > 10:
            suggestions.append("Consider chaining operations to reduce intermediate DataFrame creation")
        
        return suggestions
    
    @staticmethod
    def estimate_memory_usage(code: str, dataset_info: Dict[str, Any]) -> str:
        """Estimate memory usage of the generated code"""
        base_memory = dataset_info.get('memory_mb', 0)
        
        # Count operations that create copies
        copy_operations = len(re.findall(r'(sort_values|groupby|merge|join)', code))
        plot_operations = len(re.findall(r'plt\.', code))
        
        estimated_memory = base_memory * (1 + copy_operations * 0.5 + plot_operations * 0.2)
        
        if estimated_memory < 100:
            return f"Low (~{estimated_memory:.0f}MB)"
        elif estimated_memory < 500:
            return f"Medium (~{estimated_memory:.0f}MB)"
        else:
            return f"High (~{estimated_memory:.0f}MB)" 