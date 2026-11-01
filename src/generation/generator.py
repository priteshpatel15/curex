"""
Code Generation Engine for Natural Language Data Analysis

This module provides the core code generation functionality that converts
parsed natural language intents into executable Python code for data analysis.
"""

import ast
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of data analysis operations"""
    DESCRIPTIVE_STATS = "descriptive_stats"
    VISUALIZATION = "visualization"
    FILTERING = "filtering"
    GROUPING = "grouping"
    CORRELATION = "correlation"
    TIME_SERIES = "time_series"
    STATISTICAL_TEST = "statistical_test"
    CUSTOM = "custom"


@dataclass
class GeneratedCode:
    """Container for generated code and metadata"""
    code: str
    analysis_type: AnalysisType
    required_columns: List[str]
    imports: List[str]
    description: str
    estimated_complexity: str  # 'low', 'medium', 'high'


class CodeTemplate:
    """Base class for code templates"""
    
    def __init__(self, template: str, required_params: List[str]):
        self.template = template
        self.required_params = required_params
    
    def generate(self, params: Dict[str, Any]) -> str:
        """Generate code from template with parameters"""
        missing_params = [p for p in self.required_params if p not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        try:
            return self.template.format(**params)
        except KeyError as e:
            raise ValueError(f"Template parameter error: {e}")


class CodeGenerator:
    """
    Main code generation engine that converts NLP analysis results
    into executable Python code for data analysis.
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.imports_map = self._initialize_imports()
    
    def generate_code(self, intent: str, entities: Dict[str, Any], 
                     available_columns: List[str]) -> GeneratedCode:
        """
        Generate Python code based on parsed intent and entities.
        
        Args:
            intent: The classified intent (e.g., 'show_statistics', 'create_plot')
            entities: Extracted entities (columns, operations, parameters)
            available_columns: List of available columns in the dataset
            
        Returns:
            GeneratedCode object with the generated code and metadata
        """
        logger.info(f"Generating code for intent: {intent}")
        
        # Resolve column names with fuzzy matching
        resolved_columns = self._resolve_columns(entities.get('columns', []), available_columns)
        
        # Determine analysis type
        analysis_type = self._classify_analysis_type(intent, entities)
        
        # Select appropriate template
        template_key = self._select_template(intent, analysis_type)
        
        if template_key not in self.templates:
            raise ValueError(f"No template found for intent: {intent}")
        
        template = self.templates[template_key]
        
        # Prepare template parameters
        params = self._prepare_template_params(
            intent, entities, resolved_columns, available_columns
        )
        
        # Generate code
        code = template.generate(params)
        
        # Validate generated code
        self._validate_code(code)
        
        # Determine required imports
        required_imports = self._determine_imports(analysis_type, entities)
        
        # Estimate complexity
        complexity = self._estimate_complexity(code, analysis_type)
        
        return GeneratedCode(
            code=code,
            analysis_type=analysis_type,
            required_columns=resolved_columns,
            imports=required_imports,
            description=self._generate_description(intent, entities),
            estimated_complexity=complexity
        )
    
    def _initialize_templates(self) -> Dict[str, CodeTemplate]:
        """Initialize code templates for different analysis types"""
        templates = {}
        
        # Descriptive Statistics Templates
        templates['basic_stats'] = CodeTemplate(
            """# Basic descriptive statistics
result = df[{columns}].describe()
print("Descriptive Statistics:")
print(result)
result""",
            ['columns']
        )
        
        templates['column_stats'] = CodeTemplate(
            """# Statistics for specific column: {column_name}
column_data = df['{column_name}']
result = {{
    'count': len(column_data),
    'mean': column_data.mean() if column_data.dtype in ['int64', 'float64'] else None,
    'median': column_data.median() if column_data.dtype in ['int64', 'float64'] else None,
    'std': column_data.std() if column_data.dtype in ['int64', 'float64'] else None,
    'min': column_data.min(),
    'max': column_data.max(),
    'unique_values': column_data.nunique(),
    'missing_values': column_data.isnull().sum()
}}
print(f"Statistics for '{column_name}':")
for key, value in result.items():
    if value is not None:
        print(f"  {{key}}: {{value}}")
result""",
            ['column_name']
        )
        
        # Visualization Templates
        templates['histogram'] = CodeTemplate(
            """# Histogram for {column_name}
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
plt.hist(df['{column_name}'].dropna(), bins={bins}, alpha=0.7, edgecolor='black')
plt.title('Histogram of {column_name}')
plt.xlabel('{column_name}')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""",
            ['column_name', 'bins']
        )
        
        templates['scatter_plot'] = CodeTemplate(
            """# Scatter plot: {x_column} vs {y_column}
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
plt.scatter(df['{x_column}'], df['{y_column}'], alpha=0.7)
plt.title('Scatter Plot: {x_column} vs {y_column}')
plt.xlabel('{x_column}')
plt.ylabel('{y_column}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""",
            ['x_column', 'y_column']
        )
        
        templates['bar_chart'] = CodeTemplate(
            """# Bar chart for {column_name}
import matplotlib.pyplot as plt
import seaborn as sns

value_counts = df['{column_name}'].value_counts().head({top_n})
plt.figure(figsize=(10, 6))
plt.bar(range(len(value_counts)), value_counts.values)
plt.title('Top {top_n} Values in {column_name}')
plt.xlabel('{column_name}')
plt.ylabel('Count')
plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
plt.tight_layout()
plt.show()""",
            ['column_name', 'top_n']
        )
        
        templates['line_plot'] = CodeTemplate(
            """# Line plot: {x_column} vs {y_column}
import matplotlib.pyplot as plt
import seaborn as sns

# Sort by x_column for proper line plot
sorted_df = df.sort_values('{x_column}')
plt.figure(figsize=(12, 6))
plt.plot(sorted_df['{x_column}'], sorted_df['{y_column}'], marker='o', linewidth=2, markersize=4)
plt.title('Line Plot: {x_column} vs {y_column}')
plt.xlabel('{x_column}')
plt.ylabel('{y_column}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""",
            ['x_column', 'y_column']
        )
        
        templates['correlation_heatmap'] = CodeTemplate(
            """# Correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

correlation_matrix""",
            []
        )
        
        # Filtering Templates
        templates['filter_data'] = CodeTemplate(
            """# Filter data: {filter_description}
filtered_df = df[{filter_condition}]
print(f"Original dataset: {{len(df)}} rows")
print(f"Filtered dataset: {{len(filtered_df)}} rows")
print(f"Filtered out: {{len(df) - len(filtered_df)}} rows ({{((len(df) - len(filtered_df))/len(df)*100):.1f}}%)")
filtered_df""",
            ['filter_condition', 'filter_description']
        )
        
        # Grouping Templates
        templates['group_by'] = CodeTemplate(
            """# Group by {group_column} and {operation} {target_column}
grouped_result = df.groupby('{group_column}')['{target_column}'].{operation}()
print(f"Results grouped by '{group_column}':")
print(grouped_result)
grouped_result""",
            ['group_column', 'target_column', 'operation']
        )
        
        templates['group_by_multiple'] = CodeTemplate(
            """# Group by multiple columns: {group_columns}
grouped_result = df.groupby({group_columns})[{target_columns}].{operation}()
print("Results grouped by {group_columns_str}:")
print(grouped_result)
grouped_result""",
            ['group_columns', 'target_columns', 'operation', 'group_columns_str']
        )
        
        # Time Series Templates
        templates['time_series_plot'] = CodeTemplate(
            """# Time series plot
import matplotlib.pyplot as plt
import pandas as pd

# Convert date column to datetime if needed
if df['{date_column}'].dtype == 'object':
    df['{date_column}'] = pd.to_datetime(df['{date_column}'])

# Sort by date
sorted_df = df.sort_values('{date_column}')

plt.figure(figsize=(12, 6))
plt.plot(sorted_df['{date_column}'], sorted_df['{value_column}'], linewidth=2)
plt.title('Time Series: {value_column} over time')
plt.xlabel('{date_column}')
plt.ylabel('{value_column}')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()""",
            ['date_column', 'value_column']
        )
        
        # Statistical Tests Templates
        templates['t_test'] = CodeTemplate(
            """# T-test between groups
from scipy import stats
import numpy as np

group1_data = df[df['{group_column}'] == '{group1_value}']['{value_column}'].dropna()
group2_data = df[df['{group_column}'] == '{group2_value}']['{value_column}'].dropna()

t_stat, p_value = stats.ttest_ind(group1_data, group2_data)

print(f"T-test results:")
print(f"Group 1 ({{'{group1_value}'}}) - Mean: {{group1_data.mean():.3f}}, N: {{len(group1_data)}}")
print(f"Group 2 ({{'{group2_value}'}}) - Mean: {{group2_data.mean():.3f}}, N: {{len(group2_data)}}")
print(f"T-statistic: {{t_stat:.3f}}")
print(f"P-value: {{p_value:.3f}}")
print(f"Significant at α=0.05: {{p_value < 0.05}}")

result = {{
    'group1_mean': group1_data.mean(),
    'group2_mean': group2_data.mean(),
    't_statistic': t_stat,
    'p_value': p_value,
    'significant': p_value < 0.05
}}
result""",
            ['group_column', 'group1_value', 'group2_value', 'value_column']
        )
        
        return templates
    
    def _initialize_imports(self) -> Dict[AnalysisType, List[str]]:
        """Initialize import requirements for different analysis types"""
        return {
            AnalysisType.DESCRIPTIVE_STATS: ['pandas as pd', 'numpy as np'],
            AnalysisType.VISUALIZATION: ['matplotlib.pyplot as plt', 'seaborn as sns', 'pandas as pd'],
            AnalysisType.FILTERING: ['pandas as pd'],
            AnalysisType.GROUPING: ['pandas as pd', 'numpy as np'],
            AnalysisType.CORRELATION: ['pandas as pd', 'numpy as np', 'matplotlib.pyplot as plt', 'seaborn as sns'],
            AnalysisType.TIME_SERIES: ['pandas as pd', 'matplotlib.pyplot as plt', 'numpy as np'],
            AnalysisType.STATISTICAL_TEST: ['scipy.stats as stats', 'numpy as np', 'pandas as pd'],
            AnalysisType.CUSTOM: ['pandas as pd', 'numpy as np']
        }
    
    def _resolve_columns(self, mentioned_columns: List[str], 
                        available_columns: List[str]) -> List[str]:
        """Resolve column names using fuzzy matching"""
        from difflib import SequenceMatcher
        
        resolved = []
        for mentioned in mentioned_columns:
            if mentioned in available_columns:
                resolved.append(mentioned)
            else:
                # Fuzzy matching
                best_match = None
                best_score = 0
                for available in available_columns:
                    score = SequenceMatcher(None, mentioned.lower(), available.lower()).ratio()
                    if score > best_score and score > 0.6:  # 60% similarity threshold
                        best_match = available
                        best_score = score
                
                if best_match:
                    resolved.append(best_match)
                    logger.info(f"Fuzzy matched '{mentioned}' to '{best_match}' (score: {best_score:.2f})")
                else:
                    logger.warning(f"Could not resolve column '{mentioned}'")
        
        return resolved
    
    def _classify_analysis_type(self, intent: str, entities: Dict[str, Any]) -> AnalysisType:
        """Classify the type of analysis based on intent and entities"""
        intent_lower = intent.lower()
        
        if any(word in intent_lower for word in ['plot', 'chart', 'graph', 'visualize', 'histogram', 'scatter']):
            return AnalysisType.VISUALIZATION
        elif any(word in intent_lower for word in ['filter', 'where', 'condition']):
            return AnalysisType.FILTERING
        elif any(word in intent_lower for word in ['group', 'aggregate', 'sum', 'mean', 'count']):
            return AnalysisType.GROUPING
        elif any(word in intent_lower for word in ['correlation', 'corr', 'relationship']):
            return AnalysisType.CORRELATION
        elif any(word in intent_lower for word in ['time', 'date', 'trend', 'series']):
            return AnalysisType.TIME_SERIES
        elif any(word in intent_lower for word in ['test', 'significance', 'compare']):
            return AnalysisType.STATISTICAL_TEST
        elif any(word in intent_lower for word in ['statistics', 'stats', 'summary', 'describe']):
            return AnalysisType.DESCRIPTIVE_STATS
        else:
            return AnalysisType.CUSTOM
    
    def _select_template(self, intent: str, analysis_type: AnalysisType) -> str:
        """Select the appropriate template based on intent and analysis type"""
        intent_lower = intent.lower()
        
        # Visualization templates
        if 'histogram' in intent_lower:
            return 'histogram'
        elif any(word in intent_lower for word in ['scatter', 'relationship']):
            return 'scatter_plot'
        elif any(word in intent_lower for word in ['bar', 'count', 'frequency']):
            return 'bar_chart'
        elif any(word in intent_lower for word in ['line', 'trend', 'time']):
            return 'line_plot'
        elif 'correlation' in intent_lower and 'heatmap' in intent_lower:
            return 'correlation_heatmap'
        
        # Statistical templates
        elif any(word in intent_lower for word in ['t-test', 'ttest', 'compare groups']):
            return 't_test'
        
        # Time series templates
        elif analysis_type == AnalysisType.TIME_SERIES:
            return 'time_series_plot'
        
        # Filtering templates
        elif analysis_type == AnalysisType.FILTERING:
            return 'filter_data'
        
        # Grouping templates
        elif analysis_type == AnalysisType.GROUPING:
            return 'group_by'
        
        # Default to basic stats for descriptive analysis
        elif analysis_type == AnalysisType.DESCRIPTIVE_STATS:
            if len([col for col in intent_lower.split() if col in ['column', 'field']]) == 1:
                return 'column_stats'
            else:
                return 'basic_stats'
        
        # Default fallback
        return 'basic_stats'
    
    def _prepare_template_params(self, intent: str, entities: Dict[str, Any], 
                               resolved_columns: List[str], 
                               available_columns: List[str]) -> Dict[str, Any]:
        """Prepare parameters for template substitution"""
        params = {}
        
        # Column parameters
        if resolved_columns:
            if len(resolved_columns) == 1:
                params['column_name'] = resolved_columns[0]
                params['x_column'] = resolved_columns[0]
            elif len(resolved_columns) >= 2:
                params['x_column'] = resolved_columns[0]
                params['y_column'] = resolved_columns[1]
                params['column_name'] = resolved_columns[0]
        
        # Multi-column parameters
        params['columns'] = str(resolved_columns) if resolved_columns else str(available_columns[:5])
        
        # Grouping parameters
        if 'group_by' in entities:
            params['group_column'] = entities['group_by']
        elif resolved_columns:
            params['group_column'] = resolved_columns[0]
        
        if len(resolved_columns) >= 2:
            params['target_column'] = resolved_columns[1]
        elif resolved_columns:
            params['target_column'] = resolved_columns[0]
        
        # Operation parameters
        params['operation'] = entities.get('operation', 'mean')
        
        # Visualization parameters
        params['bins'] = entities.get('bins', 30)
        params['top_n'] = entities.get('top_n', 10)
        
        # Filtering parameters
        if 'filter_condition' in entities:
            params['filter_condition'] = entities['filter_condition']
            params['filter_description'] = entities.get('filter_description', 'custom condition')
        
        # Time series parameters
        date_columns = [col for col in available_columns if any(word in col.lower() for word in ['date', 'time', 'year', 'month'])]
        if date_columns:
            params['date_column'] = date_columns[0]
        
        if resolved_columns:
            params['value_column'] = resolved_columns[0]
        
        # Statistical test parameters
        if 'groups' in entities and len(entities['groups']) >= 2:
            params['group1_value'] = entities['groups'][0]
            params['group2_value'] = entities['groups'][1]
        
        # Group by multiple columns
        if len(resolved_columns) >= 2:
            params['group_columns'] = str(resolved_columns[:-1])
            params['target_columns'] = str([resolved_columns[-1]])
            params['group_columns_str'] = ', '.join(resolved_columns[:-1])
        
        return params
    
    def _validate_code(self, code: str) -> None:
        """Validate generated code using AST parsing"""
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Generated code has syntax error: {e}")
        
        # Check for dangerous operations
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'exec\(',
            r'eval\(',
            r'__import__',
            r'open\(',
            r'file\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ValueError(f"Generated code contains potentially dangerous operation: {pattern}")
    
    def _determine_imports(self, analysis_type: AnalysisType, entities: Dict[str, Any]) -> List[str]:
        """Determine required imports based on analysis type"""
        base_imports = self.imports_map.get(analysis_type, ['pandas as pd'])
        
        # Add additional imports based on entities
        additional_imports = []
        if 'statistical_test' in entities:
            additional_imports.append('scipy.stats as stats')
        
        return list(set(base_imports + additional_imports))
    
    def _estimate_complexity(self, code: str, analysis_type: AnalysisType) -> str:
        """Estimate computational complexity of the generated code"""
        line_count = len(code.split('\n'))
        
        if analysis_type in [AnalysisType.STATISTICAL_TEST, AnalysisType.CORRELATION]:
            return 'high'
        elif line_count > 20 or analysis_type == AnalysisType.VISUALIZATION:
            return 'medium'
        else:
            return 'low'
    
    def _generate_description(self, intent: str, entities: Dict[str, Any]) -> str:
        """Generate a human-readable description of what the code does"""
        if entities.get('columns'):
            columns_str = ', '.join(entities['columns'])
            return f"Analysis of columns: {columns_str} - {intent}"
        else:
            return f"Data analysis: {intent}" 