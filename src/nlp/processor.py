"""
Natural Language Processor for analyzing user requests and extracting intent and entities.
Handles column name resolution, operation classification, and task complexity assessment.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from fuzzywuzzy import fuzz, process
import spacy
from spacy.lang.en import English

from ..llm.models import TaskComplexity

logger = logging.getLogger(__name__)


class AnalysisIntent(Enum):
    """Types of analysis operations"""
    STATISTICAL = "statistical"
    VISUALIZATION = "visualization"
    TRANSFORMATION = "transformation"
    FILTERING = "filtering"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


class VisualizationType(Enum):
    """Types of visualizations"""
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    VIOLIN = "violin"
    PAIR = "pair"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from user input"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int


@dataclass
class AnalysisRequest:
    """Parsed analysis request with intent and entities"""
    original_text: str
    intent: AnalysisIntent
    visualization_type: Optional[VisualizationType]
    mentioned_columns: List[str]
    resolved_columns: List[str]
    parameters: Dict[str, Any]
    task_complexity: TaskComplexity
    confidence: float


class NaturalLanguageProcessor:
    """
    Natural Language Processor for data analysis requests.
    Handles intent classification, entity extraction, and column resolution.
    """
    
    def __init__(self, column_names: List[str]):
        """
        Initialize the NLP processor.
        
        Args:
            column_names: List of available column names in the dataset
        """
        self.column_names = column_names
        self.column_aliases = self._build_column_aliases()
        
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            logger.warning("spaCy English model not found, using basic English model")
            self.nlp = English()
        
        # Define pattern keywords for intent classification
        self._intent_patterns = self._build_intent_patterns()
        self._visualization_patterns = self._build_visualization_patterns()
        self._statistical_keywords = self._build_statistical_keywords()
        
    def process_request(self, user_input: str) -> AnalysisRequest:
        """
        Process a natural language request and extract intent and entities.
        
        Args:
            user_input: Natural language description of the analysis task
            
        Returns:
            AnalysisRequest: Parsed request with identified intent and entities
        """
        # Clean and normalize the input
        cleaned_input = self._clean_input(user_input)
        
        # Process with spaCy
        doc = self.nlp(cleaned_input)
        
        # Extract intent
        intent = self._classify_intent(cleaned_input, doc)
        
        # Extract visualization type if applicable
        viz_type = self._extract_visualization_type(cleaned_input) if intent == AnalysisIntent.VISUALIZATION else None
        
        # Extract column mentions
        mentioned_columns = self._extract_column_mentions(cleaned_input, doc)
        
        # Resolve column names with fuzzy matching
        resolved_columns = self._resolve_column_names(mentioned_columns)
        
        # Extract parameters
        parameters = self._extract_parameters(cleaned_input, doc, intent)
        
        # Assess task complexity
        complexity = self._assess_task_complexity(intent, len(resolved_columns), parameters)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(intent, resolved_columns, mentioned_columns)
        
        return AnalysisRequest(
            original_text=user_input,
            intent=intent,
            visualization_type=viz_type,
            mentioned_columns=mentioned_columns,
            resolved_columns=resolved_columns,
            parameters=parameters,
            task_complexity=complexity,
            confidence=confidence
        )
    
    def update_columns(self, new_columns: List[str]):
        """Update the available column names"""
        self.column_names = new_columns
        self.column_aliases = self._build_column_aliases()
    
    def _clean_input(self, text: str) -> str:
        """Clean and normalize user input"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase for processing (but keep original case for column matching)
        return text
    
    def _build_column_aliases(self) -> Dict[str, str]:
        """Build aliases and variations for column names"""
        aliases = {}
        
        for col in self.column_names:
            # Add exact match
            aliases[col.lower()] = col
            
            # Add without special characters
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', col.lower())
            aliases[clean_name] = col
            
            # Add with spaces replaced by underscores
            space_to_underscore = col.lower().replace(' ', '_')
            aliases[space_to_underscore] = col
            
            # Add with underscores replaced by spaces
            underscore_to_space = col.lower().replace('_', ' ')
            aliases[underscore_to_space] = col
            
            # Add common synonyms
            synonyms = self._get_column_synonyms(col.lower())
            for synonym in synonyms:
                aliases[synonym] = col
        
        return aliases
    
    def _get_column_synonyms(self, column_name: str) -> List[str]:
        """Get common synonyms for column names"""
        synonym_map = {
            'price': ['cost', 'amount', 'value', 'fee'],
            'sales': ['revenue', 'income', 'earnings'],
            'profit': ['margin', 'gain', 'earnings'],
            'date': ['time', 'timestamp', 'datetime'],
            'customer': ['client', 'user', 'buyer'],
            'product': ['item', 'goods', 'merchandise'],
            'quantity': ['qty', 'amount', 'count', 'number'],
            'category': ['type', 'class', 'group'],
            'id': ['identifier', 'key', 'index'],
            'name': ['title', 'label', 'description']
        }
        
        synonyms = []
        for key, values in synonym_map.items():
            if key in column_name:
                synonyms.extend(values)
            elif any(val in column_name for val in values):
                synonyms.append(key)
        
        return synonyms
    
    def _build_intent_patterns(self) -> Dict[AnalysisIntent, List[str]]:
        """Build keyword patterns for intent classification"""
        return {
            AnalysisIntent.STATISTICAL: [
                'mean', 'average', 'median', 'mode', 'std', 'standard deviation',
                'variance', 'summary', 'describe', 'statistics', 'stats',
                'min', 'max', 'count', 'sum', 'total'
            ],
            AnalysisIntent.CORRELATION: [
                'correlation', 'correlate', 'relationship', 'association',
                'dependency', 'relate', 'connection', 'linked'
            ],
            AnalysisIntent.VISUALIZATION: [
                'plot', 'chart', 'graph', 'visualize', 'show', 'display',
                'draw', 'create chart', 'create plot'
            ],
            AnalysisIntent.TRANSFORMATION: [
                'transform', 'convert', 'change', 'modify', 'create column',
                'new column', 'calculate', 'derive', 'compute'
            ],
            AnalysisIntent.FILTERING: [
                'filter', 'where', 'select', 'subset', 'rows where',
                'condition', 'criteria', 'only', 'exclude'
            ],
            AnalysisIntent.AGGREGATION: [
                'group by', 'aggregate', 'summarize', 'total by',
                'count by', 'average by', 'sum by'
            ],
            AnalysisIntent.REGRESSION: [
                'regression', 'predict', 'model', 'fit', 'linear model',
                'trend line', 'prediction', 'forecast'
            ],
            AnalysisIntent.CLUSTERING: [
                'cluster', 'group', 'segment', 'classify', 'categorize',
                'k-means', 'similar'
            ],
            AnalysisIntent.TIME_SERIES: [
                'time series', 'trend', 'over time', 'temporal',
                'seasonal', 'time-based', 'chronological'
            ],
            AnalysisIntent.COMPARISON: [
                'compare', 'comparison', 'versus', 'vs', 'against',
                'difference', 'contrast', 'between'
            ]
        }
    
    def _build_visualization_patterns(self) -> Dict[VisualizationType, List[str]]:
        """Build patterns for visualization type detection"""
        return {
            VisualizationType.SCATTER: [
                'scatter', 'scatter plot', 'point plot', 'xy plot'
            ],
            VisualizationType.LINE: [
                'line', 'line plot', 'line chart', 'time series plot'
            ],
            VisualizationType.BAR: [
                'bar', 'bar chart', 'bar plot', 'column chart'
            ],
            VisualizationType.HISTOGRAM: [
                'histogram', 'distribution', 'frequency', 'hist'
            ],
            VisualizationType.BOX: [
                'box', 'box plot', 'boxplot', 'whisker'
            ],
            VisualizationType.HEATMAP: [
                'heatmap', 'heat map', 'correlation matrix'
            ],
            VisualizationType.PIE: [
                'pie', 'pie chart', 'donut'
            ],
            VisualizationType.VIOLIN: [
                'violin', 'violin plot'
            ],
            VisualizationType.PAIR: [
                'pair plot', 'pairplot', 'scatter matrix'
            ]
        }
    
    def _build_statistical_keywords(self) -> Set[str]:
        """Build set of statistical operation keywords"""
        return {
            'mean', 'average', 'median', 'mode', 'std', 'variance',
            'min', 'max', 'sum', 'count', 'describe', 'quantile',
            'percentile', 'skew', 'kurtosis', 'correlation'
        }
    
    def _classify_intent(self, text: str, doc) -> AnalysisIntent:
        """Classify the intent of the user request"""
        text_lower = text.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        
        for intent, keywords in self._intent_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Exact match gets higher score
                    score += 1
                    # Bonus for whole word matches
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                        score += 0.5
            
            intent_scores[intent] = score
        
        # Special case: if visualization keywords are present, it's likely visualization
        viz_keywords = ['plot', 'chart', 'graph', 'visualize', 'show']
        if any(keyword in text_lower for keyword in viz_keywords):
            intent_scores[AnalysisIntent.VISUALIZATION] += 2
        
        # Find the intent with the highest score
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return best_intent[0]
        
        # Fallback based on common patterns
        if any(word in text_lower for word in ['calculate', 'compute', 'find']):
            return AnalysisIntent.STATISTICAL
        
        return AnalysisIntent.UNKNOWN
    
    def _extract_visualization_type(self, text: str) -> Optional[VisualizationType]:
        """Extract the type of visualization requested"""
        text_lower = text.lower()
        
        viz_scores = {}
        for viz_type, keywords in self._visualization_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                viz_scores[viz_type] = score
        
        if viz_scores:
            return max(viz_scores.items(), key=lambda x: x[1])[0]
        
        return VisualizationType.UNKNOWN
    
    def _extract_column_mentions(self, text: str, doc) -> List[str]:
        """Extract potential column name mentions from text"""
        mentioned_columns = []
        text_lower = text.lower()
        
        # Direct column name matches (case insensitive)
        for col in self.column_names:
            if col.lower() in text_lower:
                mentioned_columns.append(col.lower())
        
        # Extract quoted strings (might be column names)
        quoted_strings = re.findall(r'["\']([^"\']+)["\']', text)
        mentioned_columns.extend([s.lower() for s in quoted_strings])
        
        # Extract words that might be column names using NLP
        # Look for nouns and proper nouns
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                mentioned_columns.append(token.text.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_mentions = []
        for mention in mentioned_columns:
            if mention not in seen:
                seen.add(mention)
                unique_mentions.append(mention)
        
        return unique_mentions
    
    def _resolve_column_names(self, mentioned_columns: List[str]) -> List[str]:
        """Resolve mentioned column names to actual column names using fuzzy matching"""
        resolved = []
        
        for mention in mentioned_columns:
            # Try exact match first
            if mention in self.column_aliases:
                resolved.append(self.column_aliases[mention])
                continue
            
            # Try fuzzy matching
            best_match = process.extractOne(
                mention, 
                self.column_names,
                scorer=fuzz.ratio,
                score_cutoff=70
            )
            
            if best_match:
                resolved.append(best_match[0])
            else:
                # Try fuzzy matching against aliases
                alias_match = process.extractOne(
                    mention,
                    list(self.column_aliases.keys()),
                    scorer=fuzz.ratio,
                    score_cutoff=60
                )
                
                if alias_match:
                    resolved.append(self.column_aliases[alias_match[0]])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_resolved = []
        for col in resolved:
            if col not in seen:
                seen.add(col)
                unique_resolved.append(col)
        
        return unique_resolved
    
    def _extract_parameters(self, text: str, doc, intent: AnalysisIntent) -> Dict[str, Any]:
        """Extract parameters and options from the request"""
        parameters = {}
        text_lower = text.lower()
        
        # Extract numeric values
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            parameters['numbers'] = [float(n) for n in numbers]
        
        # Extract percentage mentions
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        if percentages:
            parameters['percentages'] = [float(p) for p in percentages]
        
        # Extract comparison operators
        if any(op in text_lower for op in ['greater than', '>', 'above']):
            parameters['comparison'] = 'greater'
        elif any(op in text_lower for op in ['less than', '<', 'below']):
            parameters['comparison'] = 'less'
        elif any(op in text_lower for op in ['equal', '=', 'equals']):
            parameters['comparison'] = 'equal'
        
        # Extract grouping mentions
        if any(phrase in text_lower for phrase in ['group by', 'grouped by', 'by']):
            # Try to find what to group by
            group_by_match = re.search(r'(?:group by|grouped by|by)\s+(\w+)', text_lower)
            if group_by_match:
                parameters['group_by'] = group_by_match.group(1)
        
        # Extract time period mentions
        time_periods = ['day', 'week', 'month', 'year', 'hour', 'minute']
        for period in time_periods:
            if period in text_lower:
                parameters['time_period'] = period
                break
        
        # Extract aggregation functions
        agg_functions = ['sum', 'mean', 'average', 'count', 'max', 'min', 'median']
        for func in agg_functions:
            if func in text_lower:
                parameters['aggregation'] = func
                break
        
        return parameters
    
    def _assess_task_complexity(self, intent: AnalysisIntent, num_columns: int, parameters: Dict[str, Any]) -> TaskComplexity:
        """Assess the complexity of the analysis task"""
        
        # Base complexity scoring
        complexity_score = 0
        
        # Intent-based complexity
        complex_intents = [
            AnalysisIntent.REGRESSION, 
            AnalysisIntent.CLUSTERING,
            AnalysisIntent.CLASSIFICATION,
            AnalysisIntent.TIME_SERIES
        ]
        
        if intent in complex_intents:
            complexity_score += 3
        elif intent in [AnalysisIntent.CORRELATION, AnalysisIntent.AGGREGATION]:
            complexity_score += 2
        elif intent in [AnalysisIntent.STATISTICAL, AnalysisIntent.VISUALIZATION]:
            complexity_score += 1
        
        # Column count complexity
        if num_columns > 5:
            complexity_score += 2
        elif num_columns > 2:
            complexity_score += 1
        
        # Parameter complexity
        if len(parameters) > 3:
            complexity_score += 2
        elif len(parameters) > 1:
            complexity_score += 1
        
        # Determine final complexity
        if complexity_score >= 5:
            return TaskComplexity.COMPLEX
        elif complexity_score >= 2:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.SIMPLE
    
    def _calculate_confidence(self, intent: AnalysisIntent, resolved_columns: List[str], mentioned_columns: List[str]) -> float:
        """Calculate confidence in the parsed request"""
        confidence = 0.5  # Base confidence
        
        # Intent confidence
        if intent != AnalysisIntent.UNKNOWN:
            confidence += 0.3
        
        # Column resolution confidence
        if mentioned_columns:
            resolution_rate = len(resolved_columns) / len(mentioned_columns)
            confidence += 0.2 * resolution_rate
        else:
            # No columns mentioned might be okay for some intents
            if intent in [AnalysisIntent.STATISTICAL]:
                confidence += 0.1
        
        return min(confidence, 1.0) 