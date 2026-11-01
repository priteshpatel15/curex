"""
Main Application for the Natural Language Data Analysis System.
Orchestrates all components: LLM manager, data manager, NLP processor, and code executor.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..llm.manager import LLMManager
from ..llm.models import ModelConfig, ProviderType, CodeGenerationContext
from ..data.manager import DataManager
from ..nlp.processor import NaturalLanguageProcessor, AnalysisRequest
from ..execution.executor import SafeCodeExecutor, ExecutionResult
from ..generation.generator import CodeGenerator
from ..generation.utils import CodeOptimizer, CodeFormatter, PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass
class AnalysisSession:
    """Represents a complete analysis session"""
    id: str
    user_request: str
    parsed_request: AnalysisRequest
    generated_code: str
    execution_result: ExecutionResult
    model_used: str
    timestamp: datetime
    cost: float = 0.0


class DataAnalysisApp:
    """
    Main application for natural language data analysis.
    Coordinates all system components to provide a seamless analysis experience.
    """
    
    def __init__(self):
        """Initialize the application with all core components"""
        
        # Core components
        self.llm_manager = LLMManager()
        self.data_manager = DataManager()
        self.nlp_processor = None  # Will be initialized when data is loaded
        self.code_executor = None  # Will be initialized when data is loaded
        self.code_generator = CodeGenerator()  # Template-based code generation
        
        # Session management
        self.analysis_sessions: List[AnalysisSession] = []
        self.session_counter = 0
        
        # Application state
        self.is_initialized = False
        
        logger.info("DataAnalysisApp initialized")
    
    async def initialize_default_models(self) -> Dict[str, bool]:
        """
        Initialize default LLM models for the application.
        
        Returns:
            Dict with model configuration results
        """
        results = {}
        
        # You would typically load these from configuration
        # For now, we'll set up some example configurations
        
        # Note: In production, these would come from environment variables or config files
        # This is just for demonstration
        default_configs = [
            {
                'name': 'gpt-3.5-turbo',
                'provider': ProviderType.OPENAI,
                'api_key': None,  # Would be loaded from environment
                'capabilities': ['code_generation', 'data_analysis', 'fast_response']
            },
            {
                'name': 'claude-3-sonnet-20240229',
                'provider': ProviderType.ANTHROPIC,
                'api_key': None,  # Would be loaded from environment
                'capabilities': ['code_generation', 'complex_reasoning', 'large_context']
            },
            {
                'name': 'codellama',
                'provider': ProviderType.LOCAL,
                'base_url': 'http://localhost:11434',
                'capabilities': ['code_generation', 'privacy_focused']
            }
        ]
        
        for config_dict in default_configs:
            try:
                config = ModelConfig(
                    provider=config_dict['provider'],
                    model_name=config_dict['name'],
                    api_key=config_dict.get('api_key'),
                    base_url=config_dict.get('base_url'),
                    capabilities=config_dict['capabilities']
                )
                
                success = await self.llm_manager.add_model(config)
                results[config_dict['name']] = success
                
                # Set the first successful model as active
                if success and not self.llm_manager.current_config:
                    await self.llm_manager.configure_model(config)
                
            except Exception as e:
                logger.error(f"Failed to configure model {config_dict['name']}: {str(e)}")
                results[config_dict['name']] = False
        
        logger.info(f"Model initialization results: {results}")
        return results
    
    async def load_data(self, file_path: str, dataset_name: Optional[str] = None) -> bool:
        """
        Load data and initialize data-dependent components.
        
        Args:
            file_path: Path to the data file
            dataset_name: Optional name for the dataset
            
        Returns:
            bool: True if loading was successful
        """
        try:
            # Determine file type and load accordingly
            if file_path.lower().endswith('.csv'):
                success = self.data_manager.load_csv(file_path, dataset_name)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                success = self.data_manager.load_excel(file_path, dataset_name)
            elif file_path.lower().endswith('.json'):
                success = self.data_manager.load_json(file_path, dataset_name)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return False
            
            if not success:
                logger.error(f"Failed to load data from {file_path}")
                return False
            
            # Initialize data-dependent components
            self._initialize_data_components()
            
            logger.info(f"Successfully loaded data from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def create_sample_data(self, dataset_name: str = "sample") -> bool:
        """
        Create sample data for demonstration purposes.
        
        Args:
            dataset_name: Name for the sample dataset
            
        Returns:
            bool: True if creation was successful
        """
        try:
            import numpy as np
            
            # Create sample sales data
            n_rows = 1000
            sample_data = {
                'date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
                'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], n_rows),
                'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_rows),
                'sales': np.random.normal(1000, 300, n_rows).round(2),
                'quantity': np.random.randint(1, 100, n_rows),
                'profit': np.random.normal(200, 100, n_rows).round(2),
                'customer_rating': np.random.uniform(1, 5, n_rows).round(1),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows)
            }
            
            # Ensure profit is related to sales (makes analysis more realistic)
            sample_data['profit'] = (sample_data['sales'] * 0.2 + np.random.normal(0, 50, n_rows)).round(2)
            
            success = self.data_manager.create_from_dict(sample_data, dataset_name)
            
            if success:
                self._initialize_data_components()
                logger.info(f"Created sample dataset: {dataset_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create sample data: {str(e)}")
            return False
    
    async def analyze(self, user_request: str) -> Dict[str, Any]:
        """
        Process a natural language analysis request.
        
        Args:
            user_request: Natural language description of the analysis
            
        Returns:
            Dict containing the complete analysis result
        """
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'Application not properly initialized. Please load data and configure models first.'
            }
        
        try:
            session_id = f"session_{self.session_counter}"
            self.session_counter += 1
            
            logger.info(f"Starting analysis session {session_id}: {user_request}")
            
            # Step 1: Parse the natural language request
            parsed_request = self.nlp_processor.process_request(user_request)
            
            logger.info(f"Parsed request - Intent: {parsed_request.intent.value}, "
                       f"Columns: {parsed_request.resolved_columns}, "
                       f"Confidence: {parsed_request.confidence:.2f}")
            
            # Step 2: Try template-based code generation first
            try:
                # Create entities dict for the code generator
                entities = {
                    'columns': parsed_request.resolved_columns,
                    'operation': 'mean',  # Default operation, could be extracted from NLP
                    'visualization_type': parsed_request.visualization_type.value if parsed_request.visualization_type else None
                }
                
                # Get available columns
                available_columns = self.data_manager.get_column_names()
                
                # Generate code using templates
                generated_code = self.code_generator.generate_code(
                    intent=user_request,
                    entities=entities,
                    available_columns=available_columns
                )
                
                logger.info(f"Generated code using templates - Type: {generated_code.analysis_type.value}, "
                           f"Complexity: {generated_code.estimated_complexity}")
                
                # Optimize and format the code
                optimized_code = CodeOptimizer.optimize_pandas_operations(generated_code.code)
                dataset_info = {
                    'rows': self.data_manager.get_active_dataframe().shape[0],
                    'columns': self.data_manager.get_active_dataframe().shape[1],
                    'memory_mb': self.data_manager.get_active_dataframe().memory_usage(deep=True).sum() / 1024 / 1024
                }
                optimized_code = CodeOptimizer.add_memory_optimization(optimized_code, dataset_info['memory_mb'])
                optimized_code = CodeOptimizer.add_progress_indicators(optimized_code, generated_code.estimated_complexity)
                
                # Add comments and formatting
                final_code = CodeFormatter.add_comments(
                    optimized_code, 
                    user_request, 
                    generated_code.required_columns
                )
                final_code = CodeFormatter.add_error_handling(final_code)
                
                # Step 3: Execute the generated code
                execution_result = self.code_executor.execute_code(final_code)
                
                # If template-based generation succeeded, use it
                if execution_result.success:
                    model_used = "template_engine"
                    cost = 0.0  # No cost for template-based generation
                    generated_code_text = final_code
                    
                    logger.info("Template-based code generation successful")
                else:
                    logger.warning("Template-based code execution failed, falling back to LLM")
                    raise Exception("Template execution failed")
                    
            except Exception as template_error:
                logger.info(f"Template-based generation failed ({template_error}), falling back to LLM")
                
                # Step 2b: Fall back to LLM-based code generation
                context = self._create_code_generation_context(parsed_request)
                
                llm_response = await self.llm_manager.generate_code(user_request, context)
                
                if not llm_response.success:
                    return {
                        'success': False,
                        'error': f'Both template and LLM code generation failed. Template error: {template_error}. LLM error: {llm_response.error}',
                        'session_id': session_id
                    }
                
                logger.info(f"Generated code using LLM model: {llm_response.model_used}")
                
                # Execute LLM-generated code
                execution_result = self.code_executor.execute_code(llm_response.content)
                model_used = llm_response.model_used
                cost = llm_response.cost
                generated_code_text = llm_response.content
            
            # Step 4: Evaluate code quality for feedback
            code_quality = self.code_executor.evaluate_code_quality(
                generated_code_text, execution_result
            )
            
            # Update LLM statistics with code quality (only if LLM was used)
            if model_used != "template_engine" and model_used in self.llm_manager.providers:
                provider = self.llm_manager.providers[model_used]
                # Create a mock response for statistics
                mock_response = type('MockResponse', (), {
                    'success': True,
                    'content': generated_code_text,
                    'model_used': model_used,
                    'tokens_used': len(generated_code_text.split()) * 1.5,  # Rough estimate
                    'cost': cost
                })()
                provider.stats.update_with_response(mock_response, code_quality)
            
            # Step 5: Create analysis session record
            session = AnalysisSession(
                id=session_id,
                user_request=user_request,
                parsed_request=parsed_request,
                generated_code=generated_code_text,
                execution_result=execution_result,
                model_used=model_used,
                timestamp=datetime.now(),
                cost=cost
            )
            
            self.analysis_sessions.append(session)
            
            # Step 6: Get performance suggestions
            performance_suggestions = []
            if model_used == "template_engine":
                performance_suggestions = PerformanceProfiler.suggest_optimizations(
                    generated_code_text, dataset_info
                )
            
            # Step 7: Prepare response
            result = {
                'success': execution_result.success,
                'session_id': session_id,
                'user_request': user_request,
                'parsed_request': {
                    'intent': parsed_request.intent.value,
                    'visualization_type': parsed_request.visualization_type.value if parsed_request.visualization_type else None,
                    'columns_used': parsed_request.resolved_columns,
                    'confidence': parsed_request.confidence,
                    'complexity': parsed_request.task_complexity.value
                },
                'generated_code': generated_code_text,
                'execution': {
                    'success': execution_result.success,
                    'output': execution_result.output,
                    'error': execution_result.error,
                    'execution_time': execution_result.execution_time,
                    'plots_generated': execution_result.plots_generated,
                    'variables_created': execution_result.variables_created
                },
                'model_info': {
                    'model_used': model_used,
                    'generation_method': 'template' if model_used == 'template_engine' else 'llm',
                    'tokens_used': len(generated_code_text.split()) * 1.5 if model_used == 'template_engine' else (llm_response.tokens_used if 'llm_response' in locals() else 0),
                    'cost': cost,
                    'code_quality_score': code_quality,
                    'performance_suggestions': performance_suggestions
                },
                'timestamp': session.timestamp.isoformat()
            }
            
            logger.info(f"Analysis session {session_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}',
                'session_id': session_id if 'session_id' in locals() else None
            }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded data"""
        if not self.data_manager.get_active_dataframe() is not None:
            return {'error': 'No data loaded'}
        
        return {
            'datasets': self.data_manager.get_datasets_info(),
            'active_dataset': self.data_manager.active_dataset,
            'basic_stats': self.data_manager.get_basic_stats(),
            'column_names': self.data_manager.get_column_names(),
            'sample_data': self.data_manager.get_sample_data(n_rows=3)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about configured LLM models"""
        return {
            'available_models': self.llm_manager.get_available_models(),
            'current_model': self.llm_manager.get_current_model_info(),
            'usage_stats': self.llm_manager.get_total_usage_stats()
        }
    
    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis session history"""
        recent_sessions = self.analysis_sessions[-limit:] if self.analysis_sessions else []
        
        return [
            {
                'session_id': session.id,
                'user_request': session.user_request,
                'intent': session.parsed_request.intent.value,
                'success': session.execution_result.success,
                'model_used': session.model_used,
                'cost': session.cost,
                'timestamp': session.timestamp.isoformat()
            }
            for session in reversed(recent_sessions)
        ]
    
    async def configure_model(self, provider: str, model_name: str, **kwargs) -> bool:
        """
        Configure a new LLM model.
        
        Args:
            provider: Provider name ('openai', 'anthropic', 'local')
            model_name: Name of the model
            **kwargs: Additional configuration parameters
            
        Returns:
            bool: True if configuration was successful
        """
        try:
            provider_type = ProviderType(provider.lower())
            
            config = ModelConfig(
                provider=provider_type,
                model_name=model_name,
                **kwargs
            )
            
            return await self.llm_manager.configure_model(config)
            
        except Exception as e:
            logger.error(f"Failed to configure model: {str(e)}")
            return False
    
    def _initialize_data_components(self):
        """Initialize components that depend on loaded data"""
        df = self.data_manager.get_active_dataframe()
        if df is None:
            logger.error("No active dataframe available")
            return
        
        # Initialize NLP processor with column names
        column_names = self.data_manager.get_column_names()
        self.nlp_processor = NaturalLanguageProcessor(column_names)
        
        # Initialize code executor with dataframe
        self.code_executor = SafeCodeExecutor(df)
        
        self.is_initialized = True
        logger.info("Data-dependent components initialized")
    
    def _create_code_generation_context(self, parsed_request: AnalysisRequest) -> CodeGenerationContext:
        """Create context for LLM code generation"""
        
        # Get columns to use (either specified or all available)
        columns_to_use = parsed_request.resolved_columns
        if not columns_to_use:
            columns_to_use = self.data_manager.get_column_names()[:10]  # Limit to first 10 if none specified
        
        return CodeGenerationContext(
            columns=columns_to_use,
            dtypes=self.data_manager.get_data_types(),
            shape=self.data_manager.get_active_dataframe().shape,
            sample_data=self.data_manager.get_sample_data(n_rows=3),
            user_intent=parsed_request.original_text,
            task_complexity=parsed_request.task_complexity
        )
    
    def shutdown(self):
        """Clean up resources and shut down the application"""
        try:
            # Clear any plots
            if self.code_executor:
                self.code_executor.clear_plots()
            
            # Clear data from memory
            self.data_manager.datasets.clear()
            self.data_manager.dataset_info.clear()
            
            logger.info("Application shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


# Import pandas for the sample data creation
import pandas as pd 