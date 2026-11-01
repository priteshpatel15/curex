"""
Anthropic provider implementation for the LLM abstraction layer.
Handles communication with Anthropic's Claude models for code generation.
"""

import time
from typing import Dict, List, Any

import anthropic
from anthropic.types import Message

from ..models import LLMProvider, LLMResponse, ModelConfig, ProviderType


class AnthropicProvider(LLMProvider):
    """Anthropic provider for Claude models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if config.provider != ProviderType.ANTHROPIC:
            raise ValueError(f"Config provider must be ANTHROPIC, got {config.provider}")
        
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
        self._validate_model_name()
    
    def _validate_model_name(self):
        """Validate that the model name is supported by Anthropic"""
        supported_models = {
            'claude-3-opus-20240229': {'cost_per_token': 0.000015, 'context_window': 200000},
            'claude-3-sonnet-20240229': {'cost_per_token': 0.000003, 'context_window': 200000},
            'claude-3-haiku-20240307': {'cost_per_token': 0.00000025, 'context_window': 200000},
            'claude-instant-1.2': {'cost_per_token': 0.00000163, 'context_window': 100000}
        }
        
        if self.config.model_name not in supported_models:
            raise ValueError(f"Unsupported Anthropic model: {self.config.model_name}")
        
        # Update config with model-specific defaults if not set
        model_info = supported_models[self.config.model_name]
        if self.config.cost_per_token == 0.0:
            self.config.cost_per_token = model_info['cost_per_token']
        if self.config.context_window == 4000:
            self.config.context_window = model_info['context_window']
    
    async def generate_code(self, prompt: str, context: Dict[str, Any]) -> LLMResponse:
        """Generate Python code using Anthropic's Claude models"""
        start_time = time.time()
        
        try:
            # Prepare the optimized prompt for Claude
            optimized_prompt = self._create_optimized_prompt(prompt, context)
            
            # Make the API call
            response: Message = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": optimized_prompt}
                ]
            )
            
            execution_time = time.time() - start_time
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = tokens_used * self.config.cost_per_token
            
            # Extract the generated code
            generated_content = response.content[0].text if response.content else ""
            
            llm_response = LLMResponse(
                content=generated_content,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                cost=cost,
                execution_time=execution_time,
                success=True
            )
            
            # Update statistics
            self.stats.update_with_response(llm_response)
            
            return llm_response
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_response = LLMResponse(
                content="",
                model_used=self.config.model_name,
                tokens_used=0,
                cost=0,
                execution_time=execution_time,
                success=False,
                error=f"Anthropic API error: {str(e)}"
            )
            
            # Update statistics even for failed requests
            self.stats.update_with_response(error_response)
            
            return error_response
    
    def _create_optimized_prompt(self, user_prompt: str, context: Dict[str, Any]) -> str:
        """Create an optimized prompt specifically for Claude models"""
        columns_info = self._format_columns_info(context)
        sample_data_info = self._format_sample_data(context)
        
        return f"""I need you to generate Python code for this data analysis request: "{user_prompt}"

<data_context>
{columns_info}

Dataset Shape: {context.get('shape', 'Unknown')}

Sample Data (first few rows):
{sample_data_info}

Task Complexity: {context.get('task_complexity', 'medium')}
</data_context>

<requirements>
1. Generate executable Python code that assumes the dataframe is named 'df'
2. Include all necessary imports at the beginning of the code
3. Add clear, explanatory comments for each major step
4. Handle potential errors and edge cases gracefully
5. Use pandas, numpy, scipy, matplotlib, seaborn, or plotly as appropriate
6. If creating visualizations, make them informative and well-labeled
7. Return ONLY the executable Python code
8. Do NOT include markdown code blocks or additional explanations
9. Ensure the code follows Python best practices
</requirements>

<analysis_guidelines>
- Focus on the specific analysis requested by the user
- Use appropriate statistical methods and visualizations
- Consider data types and handle missing values if relevant
- Optimize for both correctness and readability
- Include error handling where appropriate
</analysis_guidelines>

Generate the Python code:"""
    
    def _format_columns_info(self, context: Dict[str, Any]) -> str:
        """Format column information for the prompt"""
        columns = context.get('columns', [])
        dtypes = context.get('dtypes', {})
        
        if not columns:
            return "No column information available"
        
        column_lines = []
        for col in columns:
            dtype = dtypes.get(col, 'unknown')
            column_lines.append(f"  - {col}: {dtype}")
        
        return "Available Columns:\n" + "\n".join(column_lines)
    
    def _format_sample_data(self, context: Dict[str, Any]) -> str:
        """Format sample data for the prompt"""
        sample_data = context.get('sample_data', {})
        
        if not sample_data:
            return "No sample data available"
        
        # Convert sample data to a readable format
        formatted_lines = []
        for col, values in sample_data.items():
            if isinstance(values, (list, tuple)):
                values_str = ", ".join(str(v)[:50] for v in values[:3])  # Limit to first 3 values
                formatted_lines.append(f"  {col}: {values_str}")
            else:
                formatted_lines.append(f"  {col}: {str(values)[:50]}")
        
        return "\n".join(formatted_lines)
    
    def validate_config(self) -> bool:
        """Validate the Anthropic provider configuration"""
        try:
            # Check if API key is present
            if not self.config.api_key:
                return False
            
            # Basic sanity check on API key format
            # Anthropic API keys typically start with 'sk-ant-'
            return self.config.api_key.startswith('sk-ant-') and len(self.config.api_key) > 20
            
        except Exception:
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get capabilities of the Anthropic provider"""
        base_capabilities = [
            "code_generation",
            "data_analysis",
            "statistical_analysis",
            "visualization",
            "data_transformation",
            "natural_language_understanding",
            "complex_reasoning"
        ]
        
        # Add model-specific capabilities
        if 'opus' in self.config.model_name:
            base_capabilities.extend([
                "advanced_mathematics",
                "multi_step_analysis",
                "complex_problem_solving",
                "detailed_explanations"
            ])
        elif 'sonnet' in self.config.model_name:
            base_capabilities.extend([
                "balanced_performance",
                "efficient_reasoning",
                "good_code_quality"
            ])
        elif 'haiku' in self.config.model_name:
            base_capabilities.extend([
                "fast_generation",
                "simple_tasks",
                "cost_effective"
            ])
        
        return base_capabilities
    
    def get_system_prompt(self) -> str:
        """Get Anthropic-optimized system prompt (Note: Claude doesn't use system prompts in the same way)"""
        # Claude models don't use system prompts like OpenAI, but we include this for consistency
        return """You are an expert Python data analyst specializing in generating clean, efficient code for data analysis tasks.

Your expertise includes:
- Pandas for data manipulation and analysis
- NumPy for numerical computing
- SciPy for statistical analysis
- Matplotlib and Seaborn for visualization
- Plotly for interactive visualizations
- Statistical modeling and hypothesis testing

Key principles:
1. Generate code that is immediately executable
2. Assume the dataframe variable is named 'df'
3. Include comprehensive error handling
4. Add clear, explanatory comments
5. Follow Python best practices
6. Create informative visualizations when appropriate
7. Handle edge cases gracefully
8. Optimize for both performance and readability""" 