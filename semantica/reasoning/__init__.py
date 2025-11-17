"""
Reasoning Module

This module provides reasoning and inference capabilities for knowledge graph
analysis and query answering, supporting multiple reasoning strategies including
rule-based inference, SPARQL reasoning, abductive and deductive reasoning.

Algorithms Used:

Rule-Based Inference:
    - Forward Chaining: Data-driven inference, iterative rule application (while new facts exist, apply all applicable rules, add conclusions to fact set, repeat until no new facts), fact propagation through rule network, termination condition checking (no new facts or max iterations), conflict resolution (priority-based rule selection)
    - Backward Chaining: Goal-driven inference, recursive goal decomposition (decompose goal into subgoals, recursively solve subgoals, combine solutions), proof tree construction (tree of subgoals and their proofs), backtracking mechanism (try alternative paths when subgoal fails), subgoal resolution (match subgoals to facts or rules)
    - Bidirectional Chaining: Combined forward and backward chaining, meet-in-the-middle strategy (forward from facts, backward from goal, meet when paths connect), path intersection detection, combined path construction
    - Rule Matching: Pattern matching algorithms (condition-to-fact matching), unification algorithms (variable binding, term unification), variable binding mechanisms (substitution application, binding consistency checking), conflict resolution strategies (priority-based, recency-based, specificity-based)

Rete Algorithm:
    - Rete Network Construction: Alpha node creation (one per unique condition pattern), beta node join operations (join left and right inputs with variable unification), terminal node activation (rule activation when all conditions matched), network optimization (shared node detection, redundant node elimination)
    - Pattern Matching: Single condition matching (alpha nodes match facts against single condition patterns), multi-condition matching (beta nodes join results from multiple alpha nodes), incremental fact processing (add/remove facts efficiently, propagate changes through network), match propagation (matches flow from alpha to beta to terminal nodes)
    - Join Operations: Beta node joins (left input × right input with unification constraints), variable unification (bind variables consistently across joins), binding management (maintain variable bindings through join chain), join optimization (selectivity-based join ordering)
    - Conflict Resolution: Rule activation ordering (priority-based, recency-based), conflict set management (maintain set of activatable rules), selection strategy (choose rule from conflict set)

SPARQL Reasoning:
    - Query Expansion: Inference rule integration (convert rules to SPARQL patterns), query rewriting (add inferred patterns to WHERE clause), transitive closure (compute transitive relationships), property chain inference (infer relationships through property chains)
    - Query Optimization: Query plan generation (logical plan construction), join ordering (selectivity-based ordering), filter pushdown (apply filters early), projection pushdown (select only needed variables)
    - Inference Rule Integration: Rule-to-SPARQL translation (convert rule conditions/conclusions to SPARQL triple patterns), query augmentation (add inferred patterns), materialization (pre-compute inferred triples)
    - Caching: Query result caching (cache query results by query pattern), cache invalidation (invalidate on data updates), cache hit optimization (fast lookup for repeated queries)

Abductive Reasoning:
    - Hypothesis Generation: Candidate hypothesis creation (generate hypotheses from rules whose conclusions match observation), explanation space exploration (search space of possible explanations), rule-based hypothesis construction (use rules to construct explanatory hypotheses)
    - Hypothesis Ranking: Simplicity scoring (prefer hypotheses with fewer conditions, inverse of condition count), plausibility assessment (confidence-based scoring, evidence-based plausibility), consistency checking (check hypothesis consistency with known facts), coverage calculation (measure how well hypothesis explains observation, fraction of observation explained)
    - Best Explanation Selection: Ranking-based selection (select highest-ranked hypothesis), multi-criteria optimization (combine simplicity, plausibility, consistency, coverage scores), weighted scoring (weighted sum of ranking criteria)
    - Confidence Calculation: Evidence-based confidence (confidence from supporting evidence), uncertainty propagation (propagate uncertainty through reasoning chain), confidence aggregation (combine confidences from multiple sources)

Deductive Reasoning:
    - Logical Inference: Modus ponens (if P then Q, P, therefore Q), modus tollens (if P then Q, not Q, therefore not P), syllogistic reasoning (all A are B, all B are C, therefore all A are C), logical rule application (apply standard logical inference rules)
    - Proof Generation: Proof tree construction (build tree of inference steps), step-by-step derivation (record each inference step), proof validation (verify proof correctness), proof completion checking (check if proof is complete)
    - Theorem Proving: Goal decomposition (decompose theorem into subgoals), subgoal resolution (solve subgoals recursively), proof search (search space of possible proofs), backtracking (backtrack when subgoal fails)
    - Argument Construction: Premise-conclusion chains (build chain from premises to conclusion), logical validity checking (verify argument validity), soundness verification (check if conclusion follows from premises)

Rule Management:
    - Rule Parsing: Natural language rule parsing (parse "IF condition THEN conclusion" format), structured rule definition (parse structured rule formats), pattern matching (regex-based pattern matching for rule syntax), condition/conclusion extraction
    - Rule Validation: Syntax validation (check rule syntax correctness), semantic validation (check rule semantics), consistency checking (check rule consistency with existing rules), completeness checking (check if rule is well-formed)
    - Rule Execution: Condition evaluation (evaluate rule conditions against facts), conclusion generation (generate conclusion when conditions match), execution tracking (record rule execution history), performance monitoring (track execution time, success rate)
    - Rule Prioritization: Priority-based ordering (order rules by priority), conflict resolution (resolve conflicts using priority), rule scheduling (schedule rule execution based on priority)

Explanation Generation:
    - Reasoning Path Construction: Step-by-step path building (build path from initial facts to conclusion), inference chain tracking (track chain of inference steps), path visualization (visualize reasoning path as graph), path optimization (simplify path for readability)
    - Justification Creation: Evidence collection (collect supporting evidence for conclusion), support calculation (calculate support strength), confidence propagation (propagate confidence through reasoning chain), justification validation (verify justification correctness)
    - Natural Language Generation: Human-readable explanation generation (convert reasoning path to natural language), template-based formatting (use templates for explanation structure), step-by-step narration (narrate each reasoning step), conclusion summarization (summarize final conclusion)
    - Visualization: Reasoning path visualization (visualize as directed graph), proof tree rendering (render proof as tree structure), evidence highlighting (highlight supporting evidence), interactive exploration (allow interactive path exploration)

Key Features:
    - Rule-based inference with forward/backward chaining
    - SPARQL-based reasoning and query expansion
    - Rete algorithm implementation for efficient rule matching
    - Abductive reasoning for hypothesis generation
    - Deductive reasoning for logical inference
    - Rule management and execution tracking
    - Explanation generation for reasoning results
    - Proof generation and validation
    - Method registry for extensibility
    - Configuration management with environment variables and config files

Main Classes:
    - InferenceEngine: Rule-based inference with forward/backward chaining
    - SPARQLReasoner: SPARQL-based reasoning and query expansion
    - ReteEngine: Rete algorithm implementation for efficient rule matching
    - AbductiveReasoner: Abductive reasoning for hypothesis generation
    - DeductiveReasoner: Deductive reasoning for logical inference
    - RuleManager: Rule management and execution tracking
    - ExplanationGenerator: Explanation generation for reasoning results

Convenience Functions:
    - infer: General inference wrapper with strategy selection
    - forward_chain: Forward chaining inference wrapper
    - backward_chain: Backward chaining inference wrapper
    - sparql_query: SPARQL query execution wrapper
    - rete_match: Rete algorithm matching wrapper
    - abduce: Abductive reasoning wrapper
    - deduce: Deductive reasoning wrapper
    - generate_explanation: Explanation generation wrapper
    - get_reasoning_method: Get reasoning method by task and name
    - list_available_methods: List registered reasoning methods

Example Usage:
    >>> from semantica.reasoning import InferenceEngine, SPARQLReasoner, forward_chain
    >>> # Using convenience functions
    >>> results = forward_chain(facts, rules, method="default")
    >>> # Using classes directly
    >>> engine = InferenceEngine()
    >>> result = engine.infer(facts, rules)
    >>> reasoner = SPARQLReasoner()
    >>> query_result = reasoner.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")

Author: Semantica Contributors
License: MIT
"""

from .inference_engine import (
    InferenceEngine,
    InferenceResult,
    InferenceStrategy
)
from .sparql_reasoner import (
    SPARQLReasoner,
    SPARQLQueryResult
)
from .rete_engine import (
    ReteEngine,
    Fact,
    Match,
    ReteNode,
    AlphaNode,
    BetaNode,
    TerminalNode
)
from .abductive_reasoner import (
    AbductiveReasoner,
    Observation,
    Hypothesis,
    Explanation as AbductiveExplanation,
    HypothesisRanking
)
from .deductive_reasoner import (
    DeductiveReasoner,
    Premise,
    Conclusion,
    Proof,
    Argument
)
from .rule_manager import (
    RuleManager,
    Rule,
    RuleExecution,
    RuleType
)
from .explanation_generator import (
    ExplanationGenerator,
    Explanation,
    ReasoningStep,
    ReasoningPath,
    Justification
)

__all__ = [
    # Inference
    "InferenceEngine",
    "InferenceResult",
    "InferenceStrategy",
    
    # SPARQL reasoning
    "SPARQLReasoner",
    "SPARQLQueryResult",
    
    # Rete algorithm
    "ReteEngine",
    "Fact",
    "Match",
    "ReteNode",
    "AlphaNode",
    "BetaNode",
    "TerminalNode",
    
    # Abductive reasoning
    "AbductiveReasoner",
    "Observation",
    "Hypothesis",
    "AbductiveExplanation",
    "HypothesisRanking",
    
    # Deductive reasoning
    "DeductiveReasoner",
    "Premise",
    "Conclusion",
    "Proof",
    "Argument",
    
    # Rule management
    "RuleManager",
    "Rule",
    "RuleExecution",
    "RuleType",
    
    # Explanation
    "ExplanationGenerator",
    "Explanation",
    "ReasoningStep",
    "ReasoningPath",
    "Justification",
]
