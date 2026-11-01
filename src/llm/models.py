"""
Core data models and interfaces for the LLM abstraction layer.
These models define the structure for LLM configurations, responses, and provider interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time


class ProviderType(Enum):
    """Enumeration of supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class TaskComplexity(Enum):
    """Enumeration of task complexity levels for model selection"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    provider: ProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 30
    cost_per_token: float = 0.0
    context_window: int = 4000
    capabilities: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.provider == ProviderType.OPENAI and not self.api_key:
            raise ValueError("OpenAI provider requires an API key")
        if self.provider == ProviderType.ANTHROPIC and not self.api_key:
            raise ValueError("Anthropic provider requires an API key")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")


@dataclass
class LLMResponse:
    """Response from an LLM provider"""
    content: str
    model_used: str
    tokens_used: int
    cost: float
    execution_time: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_valid_python_code(self) -> bool:
        """Check if the response contains valid Python code"""
        if not self.success or not self.content:
            return False
        
        # Basic check for Python code structure
        import ast
        try:
            # Extract code from markdown if present
            code = self._extract_code_from_markdown(self.content)
            ast.parse(code)
            return True
        except (SyntaxError, ValueError):
            return False
    
    def _extract_code_from_markdown(self, content: str) -> str:
        """Extract Python code from markdown code blocks"""
        import re
        
        # Look for ```python code blocks
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', content, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        # Look for generic ``` code blocks
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', content, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, return the content as-is
        return content.strip()


@dataclass
class ModelUsageStats:
    """Statistics for model usage and performance"""
    total_requests: int = 0
    successful_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_execution_time: float = 0.0
    avg_code_quality_score: float = 0.0
    last_used: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def update_with_response(self, response: LLMResponse, code_quality_score: float = 0.0):
        """Update statistics with a new response"""
        self.total_requests += 1
        if response.success:
            self.successful_requests += 1
        
        self.total_tokens += response.tokens_used
        self.total_cost += response.cost
        
        # Update rolling average for execution time
        if self.total_requests == 1:
            self.avg_execution_time = response.execution_time
        else:
            self.avg_execution_time = (
                (self.avg_execution_time * (self.total_requests - 1) + response.execution_time)
                / self.total_requests
            )
        
        # Update rolling average for code quality
        if response.success and code_quality_score > 0:
            if self.successful_requests == 1:
                self.avg_code_quality_score = code_quality_score
            else:
                self.avg_code_quality_score = (
                    (self.avg_code_quality_score * (self.successful_requests - 1) + code_quality_score)
                    / self.successful_requests
                )
        
        self.last_used = response.timestamp


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.stats = ModelUsageStats()
    
    @abstractmethod
    async def generate_code(self, prompt: str, context: Dict[str, Any]) -> LLMResponse:
        """Generate Python code based on natural language prompt and data context"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        pass
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for code generation"""
        return """You are an expert Python data analyst. Your task is to generate clean, efficient Python code for data analysis tasks.

Rules:
1. Always assume the dataframe is named 'df'
2. Use pandas, numpy, scipy, matplotlib, seaborn, and plotly as needed
3. Include appropriate error handling
4. Add clear, explanatory comments
5. Generate visualizations when appropriate
6. Return only executable Python code without markdown formatting
7. Ensure all imports are included at the top
8. Handle edge cases gracefully

Focus on producing code that is:
- Correct and executable
- Well-commented and readable
- Efficient and follows best practices
- Appropriate for the given data context"""


@dataclass
class CodeGenerationContext:
    """Context information for code generation"""
    columns: List[str]
    dtypes: Dict[str, str]
    shape: tuple
    sample_data: Dict[str, Any]
    user_intent: str
    task_complexity: TaskComplexity = TaskComplexity.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for LLM processing"""
        return {
            'columns': self.columns,
            'dtypes': self.dtypes,
            'shape': self.shape,
            'sample_data': self.sample_data,
            'user_intent': self.user_intent,
            'task_complexity': self.task_complexity.value
        } 