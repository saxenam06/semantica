"""
Reasoner Module

This module provides a high-level Reasoner class that unifies various reasoning strategies
supported by the Semantica framework. It serves as a facade for different reasoning engines
<<<<<<< HEAD
like SPARQLReasoner, etc.
=======
like InferenceEngine, SPARQLReasoner, etc.
>>>>>>> main
"""

from typing import Any, Dict, List, Optional, Union
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
<<<<<<< HEAD
=======
from .inference_engine import InferenceEngine, InferenceStrategy
>>>>>>> main
from .rule_manager import Rule

class Reasoner:
    """
    High-level Reasoner class for knowledge graph inference.
    
    This class provides a unified interface for applying reasoning rules to facts
    or knowledge graphs.
    """
    
    def __init__(self, strategy: str = "forward", **kwargs):
        """
        Initialize the Reasoner.
        
        Args:
            strategy: Inference strategy ("forward", "backward", etc.)
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("reasoner")
        self.progress_tracker = get_progress_tracker()
        self.strategy = strategy
        self.config = kwargs
        
<<<<<<< HEAD
        # Inference engine disabled
        self.engine = None
=======
        # Initialize the underlying engine based on strategy
        # Currently defaults to InferenceEngine for forward/backward chaining
        if strategy in ["forward", "backward"]:
            self.engine = InferenceEngine(strategy=strategy, **kwargs)
        else:
            # Default to forward chaining if unknown strategy
            self.engine = InferenceEngine(strategy="forward", **kwargs)
>>>>>>> main
            
    def infer_facts(
        self, 
        facts: Union[List[Any], Dict[str, Any]], 
        rules: Optional[List[Union[str, Rule]]] = None
    ) -> List[Any]:
        """
        Infer new facts from existing facts or a knowledge graph.
        
        Args:
            facts: List of initial facts or a knowledge graph dictionary.
<<<<<<< HEAD
                   If a list is provided, it can contain strings, dicts, or MergeOperation objects.
=======
                  If a list is provided, it can contain strings, dicts, or MergeOperation objects.
>>>>>>> main
            rules: List of rules to apply (strings or Rule objects)
            
        Returns:
            List of inferred facts (conclusions)
        """
<<<<<<< HEAD
        if self.engine is None:
            self.logger.warning("Inference engine is currently disabled.")
            return []

        # Note: This method is a placeholder as the InferenceEngine is currently disabled.
        return []
            
    def add_rule(self, rule: Union[str, Rule]) -> None:
        """Add a rule to the reasoner."""
        if self.engine:
            self.engine.add_rule(rule)
        
    def clear(self) -> None:
        """Clear facts and rules."""
        if self.engine:
            self.engine.reset()
=======
        # Pre-process facts if they come from a knowledge graph or are MergeOperation objects
        processed_facts = []
        
        if isinstance(facts, dict):
            # If it's a knowledge graph, extract facts from entities and relationships
            if "entities" in facts:
                for entity in facts["entities"]:
                    # Create simple fact string for reasoning
                    # e.g., Person(John)
                    name = entity.get("name", entity.get("id"))
                    etype = entity.get("type", "Entity")
                    processed_facts.append(f"{etype}({name})")
            
            if "relationships" in facts:
                for rel in facts["relationships"]:
                    # Create simple relationship fact string
                    # e.g., WorksFor(John, Acme)
                    source = rel.get("source_name", rel.get("source_id"))
                    target = rel.get("target_name", rel.get("target_id"))
                    rtype = rel.get("type", "Relationship")
                    processed_facts.append(f"{rtype}({source}, {target})")
        
        elif isinstance(facts, list):
            for item in facts:
                # Check if it's a MergeOperation (which has a merged_entity)
                if hasattr(item, "merged_entity"):
                    entity = item.merged_entity
                    name = entity.get("name", entity.get("id"))
                    etype = entity.get("type", "Entity")
                    processed_facts.append(f"{etype}({name})")
                elif isinstance(item, dict):
                    # It's an entity dictionary
                    name = item.get("name", item.get("id"))
                    etype = item.get("type", "Entity")
                    processed_facts.append(f"{etype}({name})")
                else:
                    # Assume it's already a fact string or compatible object
                    processed_facts.append(item)
        
        if not processed_facts:
            processed_facts = facts if isinstance(facts, list) else []

        tracking_id = self.progress_tracker.start_tracking(
            module="reasoning",
            submodule="Reasoner",
            message=f"Inferring facts from {len(processed_facts)} processed facts"
        )
        
        try:
            # Add rules if provided
            if rules:
                for rule in rules:
                    self.engine.add_rule(rule)
            
            # Perform inference
            # The engine.infer method (or forward_chain) returns InferenceResult objects
            # We want to return the actual facts (conclusions)
            if self.strategy == "forward":
                results = self.engine.forward_chain(processed_facts)
            else:
                # Fallback to general infer method
                results = self.engine.infer(processed_facts, rules)
                
            # Extract conclusions from results
            inferred_facts = [result.conclusion for result in results]
            
            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Inferred {len(inferred_facts)} new facts"
            )
            
            return inferred_facts
            
        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, 
                status="failed", 
                message=str(e)
            )
            self.logger.error(f"Inference failed: {e}")
            raise
            
    def add_rule(self, rule: Union[str, Rule]) -> None:
        """Add a rule to the reasoner."""
        self.engine.add_rule(rule)
        
    def clear(self) -> None:
        """Clear facts and rules."""
        self.engine.reset()
>>>>>>> main
