"""
LLM Manager for orchestrating multiple LLM providers and handling runtime configuration.
Provides a unified interface for code generation with configurable model backends.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Type
import logging

from .models import (
    LLMProvider, LLMResponse, ModelConfig, ProviderType, 
    ModelUsageStats, TaskComplexity, CodeGenerationContext
)
from .providers import OpenAIProvider, AnthropicProvider, LocalProvider

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Central manager for LLM providers with runtime configuration and graceful error handling.
    Supports multiple models, automatic selection, and usage tracking.
    """
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.current_config: Optional[ModelConfig] = None
        self.provider_classes: Dict[ProviderType, Type[LLMProvider]] = {
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.ANTHROPIC: AnthropicProvider,
            ProviderType.LOCAL: LocalProvider
        }
        self.model_selection_strategy = "manual"  # manual, performance, cost
        
    async def configure_model(self, config: ModelConfig) -> bool:
        """
        Configure the active model with validation and error handling.
        
        Args:
            config: Model configuration to set as active
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        try:
            provider = await self._create_provider(config)
            
            # Validate the configuration
            if not await self._validate_provider(provider):
                logger.error(f"Provider validation failed for {config.model_name}")
                return False
            
            # Store the provider and set as current
            self.providers[config.model_name] = provider
            self.current_config = config
            
            logger.info(f"Successfully configured model: {config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure model {config.model_name}: {str(e)}")
            return False
    
    async def add_model(self, config: ModelConfig) -> bool:
        """
        Add a model configuration without setting it as active.
        
        Args:
            config: Model configuration to add
            
        Returns:
            bool: True if model was added successfully
        """
        try:
            provider = await self._create_provider(config)
            
            if not await self._validate_provider(provider):
                return False
            
            self.providers[config.model_name] = provider
            logger.info(f"Added model: {config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model {config.model_name}: {str(e)}")
            return False
    
    async def generate_code(self, prompt: str, context: CodeGenerationContext) -> LLMResponse:
        """
        Generate Python code using the configured model with graceful error handling.
        
        Args:
            prompt: Natural language description of the analysis task
            context: Context information about the data and task
            
        Returns:
            LLMResponse: Response containing generated code or error information
        """
        if not self.current_config:
            return LLMResponse(
                content="",
                model_used="none",
                tokens_used=0,
                cost=0,
                execution_time=0,
                success=False,
                error="No model configured. Please configure a model first."
            )
        
        try:
            # Get the active provider
            provider = self.providers.get(self.current_config.model_name)
            if not provider:
                raise ValueError(f"Provider not found for model: {self.current_config.model_name}")
            
            # Generate code using the provider
            response = await provider.generate_code(prompt, context.to_dict())
            
            # Log usage
            self._log_usage(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            
            return LLMResponse(
                content="",
                model_used=self.current_config.model_name,
                tokens_used=0,
                cost=0,
                execution_time=0,
                success=False,
                error=f"Code generation failed: {str(e)}"
            )
    
    async def generate_code_with_best_model(self, prompt: str, context: CodeGenerationContext) -> LLMResponse:
        """
        Generate code using the best available model based on task complexity and model performance.
        
        Args:
            prompt: Natural language description of the analysis task
            context: Context information about the data and task
            
        Returns:
            LLMResponse: Response from the best available model
        """
        best_model = await self._select_best_model(context.task_complexity)
        
        if best_model:
            # Temporarily switch to the best model
            original_config = self.current_config
            self.current_config = best_model.config
            
            try:
                response = await self.generate_code(prompt, context)
                return response
            finally:
                # Restore original configuration
                self.current_config = original_config
        else:
            # Fall back to current model
            return await self.generate_code(prompt, context)
    
    async def _create_provider(self, config: ModelConfig) -> LLMProvider:
        """Create a provider instance based on configuration"""
        provider_class = self.provider_classes.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        return provider_class(config)
    
    async def _validate_provider(self, provider: LLMProvider) -> bool:
        """Validate provider configuration and connectivity"""
        try:
            # For local providers, we can test connectivity
            if hasattr(provider, 'validate_config'):
                if asyncio.iscoroutinefunction(provider.validate_config):
                    return await provider.validate_config()
                else:
                    return provider.validate_config()
            return True
        except Exception as e:
            logger.error(f"Provider validation error: {str(e)}")
            return False
    
    async def _select_best_model(self, task_complexity: TaskComplexity) -> Optional[LLMProvider]:
        """Select the best model based on task complexity and performance metrics"""
        if not self.providers:
            return None
        
        # Simple selection logic - can be enhanced with more sophisticated algorithms
        available_providers = list(self.providers.values())
        
        if task_complexity == TaskComplexity.SIMPLE:
            # Prefer fastest/cheapest models for simple tasks
            return min(available_providers, 
                      key=lambda p: p.stats.avg_execution_time + p.config.cost_per_token * 1000)
        
        elif task_complexity == TaskComplexity.COMPLEX:
            # Prefer highest quality models for complex tasks
            return max(available_providers,
                      key=lambda p: p.stats.avg_code_quality_score if p.stats.successful_requests > 0 else 0)
        
        else:  # MEDIUM
            # Balanced approach
            def balanced_score(provider: LLMProvider) -> float:
                stats = provider.stats
                if stats.successful_requests == 0:
                    return 0
                
                # Weighted score combining success rate, quality, and speed
                return (stats.success_rate * 0.4 + 
                       stats.avg_code_quality_score * 10 * 0.4 + 
                       (1 / max(stats.avg_execution_time, 0.1)) * 0.2)
            
            return max(available_providers, key=balanced_score)
    
    def _log_usage(self, response: LLMResponse):
        """Log model usage for monitoring and analytics"""
        logger.info(f"Model {response.model_used}: "
                   f"Success={response.success}, "
                   f"Tokens={response.tokens_used}, "
                   f"Cost=${response.cost:.4f}, "
                   f"Time={response.execution_time:.2f}s")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get information about all configured models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        for model_name, provider in self.providers.items():
            stats = provider.stats
            
            model_info = {
                'name': model_name,
                'provider': provider.config.provider.value,
                'is_active': self.current_config and self.current_config.model_name == model_name,
                'cost_per_token': provider.config.cost_per_token,
                'context_window': provider.config.context_window,
                'capabilities': provider.get_capabilities(),
                'stats': {
                    'total_requests': stats.total_requests,
                    'success_rate': stats.success_rate,
                    'total_cost': stats.total_cost,
                    'avg_execution_time': stats.avg_execution_time,
                    'avg_code_quality': stats.avg_code_quality_score,
                    'last_used': stats.last_used
                }
            }
            
            models.append(model_info)
        
        # Sort by active status, then by success rate
        models.sort(key=lambda m: (m['is_active'], m['stats']['success_rate']), reverse=True)
        
        return models
    
    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently active model"""
        if not self.current_config:
            return None
        
        provider = self.providers.get(self.current_config.model_name)
        if not provider:
            return None
        
        return {
            'name': self.current_config.model_name,
            'provider': self.current_config.provider.value,
            'config': {
                'max_tokens': self.current_config.max_tokens,
                'temperature': self.current_config.temperature,
                'cost_per_token': self.current_config.cost_per_token,
                'context_window': self.current_config.context_window
            },
            'capabilities': provider.get_capabilities(),
            'stats': provider.stats.__dict__
        }
    
    async def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different configured model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            bool: True if switch was successful
        """
        if model_name not in self.providers:
            logger.error(f"Model {model_name} not found in configured providers")
            return False
        
        provider = self.providers[model_name]
        self.current_config = provider.config
        
        logger.info(f"Switched to model: {model_name}")
        return True
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model configuration.
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            bool: True if removal was successful
        """
        if model_name not in self.providers:
            return False
        
        # Don't remove the active model
        if self.current_config and self.current_config.model_name == model_name:
            logger.warning(f"Cannot remove active model: {model_name}")
            return False
        
        del self.providers[model_name]
        logger.info(f"Removed model: {model_name}")
        return True
    
    def get_total_usage_stats(self) -> Dict[str, Any]:
        """Get aggregate usage statistics across all models"""
        total_requests = sum(p.stats.total_requests for p in self.providers.values())
        total_successful = sum(p.stats.successful_requests for p in self.providers.values())
        total_cost = sum(p.stats.total_cost for p in self.providers.values())
        total_tokens = sum(p.stats.total_tokens for p in self.providers.values())
        
        avg_execution_time = 0
        if self.providers:
            avg_execution_time = sum(p.stats.avg_execution_time for p in self.providers.values()) / len(self.providers)
        
        return {
            'total_requests': total_requests,
            'total_successful': total_successful,
            'overall_success_rate': (total_successful / total_requests * 100) if total_requests > 0 else 0,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'avg_execution_time': avg_execution_time,
            'models_configured': len(self.providers),
            'active_model': self.current_config.model_name if self.current_config else None
        } 