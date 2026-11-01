"""
Local provider implementation for the LLM abstraction layer.
Handles communication with local LLM models (e.g., Ollama, local API endpoints).
"""

import time
import json
from typing import Dict, List, Any

import aiohttp

from ..models import LLMProvider, LLMResponse, ModelConfig, ProviderType


class LocalProvider(LLMProvider):
    """Local provider for self-hosted LLM models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if config.provider != ProviderType.LOCAL:
            raise ValueError(f"Config provider must be LOCAL, got {config.provider}")
        
        self.base_url = config.base_url or "http://localhost:11434"
        self._validate_model_name()
    
    def _validate_model_name(self):
        """Validate that the model name is supported by the local provider"""
        # For local models, we're more permissive since users can run various models
        common_models = {
            'llama2': {'cost_per_token': 0.0, 'context_window': 4096},
            'llama2:13b': {'cost_per_token': 0.0, 'context_window': 4096},
            'llama2:70b': {'cost_per_token': 0.0, 'context_window': 4096},
            'codellama': {'cost_per_token': 0.0, 'context_window': 4096},
            'codellama:13b': {'cost_per_token': 0.0, 'context_window': 4096},
            'codellama:34b': {'cost_per_token': 0.0, 'context_window': 16384},
            'mistral': {'cost_per_token': 0.0, 'context_window': 8192},
            'deepseek-coder': {'cost_per_token': 0.0, 'context_window': 4096},
            'deepseek-coder:33b': {'cost_per_token': 0.0, 'context_window': 16384},
            'phi': {'cost_per_token': 0.0, 'context_window': 2048},
            'neural-chat': {'cost_per_token': 0.0, 'context_window': 4096}
        }
        
        # Update config with model-specific defaults if available
        if self.config.model_name in common_models:
            model_info = common_models[self.config.model_name]
            if self.config.cost_per_token == 0.0:
                self.config.cost_per_token = model_info['cost_per_token']
            if self.config.context_window == 4000:
                self.config.context_window = model_info['context_window']
    
    async def generate_code(self, prompt: str, context: Dict[str, Any]) -> LLMResponse:
        """Generate Python code using local LLM models"""
        start_time = time.time()
        
        try:
            # Prepare the optimized prompt for local models
            optimized_prompt = self._create_optimized_prompt(prompt, context)
            
            # Prepare payload for Ollama API
            payload = {
                "model": self.config.model_name,
                "prompt": optimized_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"API returned status {response.status}")
                    
                    result = await response.json()
            
            execution_time = time.time() - start_time
            
            # Extract the generated content
            generated_content = result.get('response', '')
            
            # Local models don't provide token counts, so we estimate
            estimated_tokens = len(generated_content.split()) * 1.3  # Rough estimate
            
            llm_response = LLMResponse(
                content=generated_content,
                model_used=self.config.model_name,
                tokens_used=int(estimated_tokens),
                cost=0.0,  # Local models are free
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
                error=f"Local model error: {str(e)}"
            )
            
            # Update statistics even for failed requests
            self.stats.update_with_response(error_response)
            
            return error_response
    
    def _create_optimized_prompt(self, user_prompt: str, context: Dict[str, Any]) -> str:
        """Create an optimized prompt specifically for local models"""
        columns_info = self._format_columns_info(context)
        sample_data_info = self._format_sample_data(context)
        
        # Local models often need more explicit instructions
        return f"""You are a Python data analysis expert. Generate executable Python code for the following request.

USER REQUEST: {user_prompt}

DATASET INFORMATION:
{columns_info}

Dataset Shape: {context.get('shape', 'Unknown')}

Sample Data:
{sample_data_info}

REQUIREMENTS:
- Generate only Python code, no explanations
- Assume the dataframe is named 'df'
- Include necessary imports at the top
- Add clear comments for each step
- Handle errors gracefully
- Use pandas, numpy, scipy, matplotlib, seaborn as needed
- Create visualizations when appropriate

RESPONSE FORMAT:
Return only the Python code without any markdown formatting or additional text.

Python code:"""
    
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
                values_str = ", ".join(str(v)[:30] for v in values[:3])  # Shorter for local models
                formatted_lines.append(f"  {col}: {values_str}")
            else:
                formatted_lines.append(f"  {col}: {str(values)[:30]}")
        
        return "\n".join(formatted_lines)
    
    async def validate_config(self) -> bool:
        """Validate the local provider configuration by testing connectivity"""
        try:
            # Test connectivity to the local API
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        # Check if our model is available
                        data = await response.json()
                        available_models = [model['name'] for model in data.get('models', [])]
                        return self.config.model_name in available_models
                    return False
                    
        except Exception:
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get capabilities of the local provider"""
        base_capabilities = [
            "code_generation",
            "data_analysis",
            "privacy_focused",
            "offline_operation",
            "cost_effective"
        ]
        
        # Add model-specific capabilities based on model name
        if 'codellama' in self.config.model_name or 'deepseek-coder' in self.config.model_name:
            base_capabilities.extend([
                "code_optimization",
                "advanced_programming",
                "debugging_support"
            ])
        
        if 'llama2' in self.config.model_name:
            base_capabilities.extend([
                "general_purpose",
                "natural_language_understanding"
            ])
        
        if 'mistral' in self.config.model_name:
            base_capabilities.extend([
                "efficient_reasoning",
                "multilingual_support"
            ])
        
        return base_capabilities
    
    def get_system_prompt(self) -> str:
        """Get local model-optimized system prompt"""
        return """You are an expert Python data analyst. Generate clean, executable Python code for data analysis tasks.

Key rules:
1. Always assume the dataframe is named 'df'
2. Include all necessary imports
3. Add clear comments
4. Handle errors gracefully
5. Use pandas, numpy, scipy, matplotlib, seaborn
6. Return only executable Python code
7. No markdown formatting
8. Focus on correctness and efficiency"""
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from the local API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['name'] for model in data.get('models', [])]
                    return []
                    
        except Exception:
            return [] 