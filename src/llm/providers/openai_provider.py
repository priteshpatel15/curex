"""
OpenAI provider implementation for the LLM abstraction layer.
Handles communication with OpenAI's GPT models for code generation.
"""

import time
import asyncio
from typing import Dict, List, Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from ..models import LLMProvider, LLMResponse, ModelConfig, ProviderType, CodeGenerationContext


class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if config.provider != ProviderType.OPENAI:
            raise ValueError(f"Config provider must be OPENAI, got {config.provider}")
        
        self.client = AsyncOpenAI(api_key=config.api_key)
        self._validate_model_name()
    
    def _validate_model_name(self):
        """Validate that the model name is supported by OpenAI"""
        supported_models = {
            'gpt-4': {'cost_per_token': 0.00003, 'context_window': 8192},
            'gpt-4-turbo': {'cost_per_token': 0.00001, 'context_window': 128000},
            'gpt-4-turbo-preview': {'cost_per_token': 0.00001, 'context_window': 128000},
            'gpt-3.5-turbo': {'cost_per_token': 0.0000015, 'context_window': 16385},
            'gpt-3.5-turbo-16k': {'cost_per_token': 0.000003, 'context_window': 16385}
        }
        
        if self.config.model_name not in supported_models:
            raise ValueError(f"Unsupported OpenAI model: {self.config.model_name}")
        
        # Update config with model-specific defaults if not set
        model_info = supported_models[self.config.model_name]
        if self.config.cost_per_token == 0.0:
            self.config.cost_per_token = model_info['cost_per_token']
        if self.config.context_window == 4000:
            self.config.context_window = model_info['context_window']
    
    async def generate_code(self, prompt: str, context: Dict[str, Any]) -> LLMResponse:
        """Generate Python code using OpenAI's GPT models"""
        start_time = time.time()
        
        try:
            # Prepare the optimized prompt for OpenAI
            optimized_prompt = self._create_optimized_prompt(prompt, context)
            
            # Make the API call
            response: ChatCompletion = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": optimized_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            execution_time = time.time() - start_time
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost = tokens_used * self.config.cost_per_token
            
            # Extract the generated code
            generated_content = response.choices[0].message.content or ""
            
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
                error=f"OpenAI API error: {str(e)}"
            )
            
            # Update statistics even for failed requests
            self.stats.update_with_response(error_response)
            
            return error_response
    
    def _create_optimized_prompt(self, user_prompt: str, context: Dict[str, Any]) -> str:
        """Create an optimized prompt specifically for OpenAI models"""
        columns_info = self._format_columns_info(context)
        sample_data_info = self._format_sample_data(context)
        
        return f"""Data Analysis Request: {user_prompt}

Dataset Information:
{columns_info}

Dataset Shape: {context.get('shape', 'Unknown')}

Sample Data (first few rows):
{sample_data_info}

Requirements:
1. Generate executable Python code that assumes the dataframe is named 'df'
2. Include all necessary imports at the top
3. Add clear comments explaining each step
4. Handle potential errors gracefully
5. If creating visualizations, use matplotlib or seaborn
6. Return only the Python code without markdown formatting

Task Complexity: {context.get('task_complexity', 'medium')}"""
    
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
        """Validate the OpenAI provider configuration"""
        try:
            # Check if API key is present
            if not self.config.api_key:
                return False
            
            # Test API connectivity (simple way)
            # Note: In production, you might want to make a minimal API call
            return len(self.config.api_key) > 10  # Basic sanity check
            
        except Exception:
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get capabilities of the OpenAI provider"""
        base_capabilities = [
            "code_generation",
            "data_analysis", 
            "statistical_analysis",
            "visualization",
            "data_transformation",
            "natural_language_understanding"
        ]
        
        # Add model-specific capabilities
        if 'gpt-4' in self.config.model_name:
            base_capabilities.extend([
                "complex_reasoning",
                "advanced_mathematics",
                "multi_step_analysis"
            ])
        
        return base_capabilities
    
    def get_system_prompt(self) -> str:
        """Get OpenAI-optimized system prompt"""
        return """You are an expert Python data analyst specializing in generating clean, efficient code for data analysis tasks.

Your expertise includes:
- Pandas for data manipulation and analysis
- NumPy for numerical computing
- SciPy for statistical analysis
- Matplotlib and Seaborn for visualization
- Plotly for interactive visualizations
- Statistical modeling and hypothesis testing

Code Generation Rules:
1. Always assume the dataframe variable is named 'df'
2. Include all necessary imports at the beginning
3. Add clear, explanatory comments for each major step
4. Handle edge cases and potential errors gracefully
5. Use appropriate data types and efficient operations
6. Generate visualizations when they would be helpful
7. Return only executable Python code
8. Follow Python best practices and PEP 8 style guidelines

Response Format:
- Return ONLY the Python code
- NO markdown code blocks or formatting
- NO additional explanations outside of code comments
- Ensure code is immediately executable""" 