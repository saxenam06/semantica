
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_deduplication_notebook():
    print("=== Starting Deduplication Notebook Verification ===")
    
    # Import all deduplication classes
    print("\n[1] Importing deduplication classes...")
    from semantica.deduplication import (
        # Main Classes
        DuplicateDetector,
        EntityMerger,
        SimilarityCalculator,
        ClusterBuilder,
        MergeStrategyManager,
        MethodRegistry,
        DeduplicationConfig,
        # Data Classes
        DuplicateCandidate,
        DuplicateGroup,
        MergeOperation,
        SimilarityResult,
        Cluster,
        ClusterResult,
        MergeResult,
        MergeStrategy,
        # Global Instances
        method_registry,
        dedup_config,
    )
    print("Imports successful.")

    # Create sample entities with potential duplicates
    print("\n[2] Creating sample entities...")
    entities = [
        {
            "id": "e1",
            "name": "Apple Inc.",
            "type": "Company",
            "founded": 1976,
            "properties": {"industry": "Technology", "headquarters": "Cupertino"},
            "relationships": [{"subject": "e1", "predicate": "founded_by", "object": "Steve Jobs"}],
        },
        {
            "id": "e2",
            "name": "Apple Inc",
            "type": "Company",
            "founded": 1976,
            "properties": {"industry": "Tech", "headquarters": "Cupertino, CA"},
            "relationships": [{"subject": "e2", "predicate": "founded_by", "object": "Steve Jobs"}],
        },
        {
            "id": "e3",
            "name": "Microsoft Corporation",
            "type": "Company",
            "founded": 1975,
            "properties": {"industry": "Technology", "headquarters": "Redmond"},
        },
        {
            "id": "e4",
            "name": "Microsoft",
            "type": "Company",
            "founded": 1975,
            "properties": {"industry": "Tech", "headquarters": "Redmond, WA"},
        },
        {
            "id": "e5",
            "name": "Google LLC",
            "type": "Company",
            "founded": 1998,
            "properties": {"industry": "Technology"},
        },
    ]

    print(f"Created {len(entities)} sample entities")
    print("Entity names:")
    for e in entities:
        print(f"  - {e['name']} (ID: {e['id']})")

    # Example: Duplicate Detection
    print("\n[3] Testing Duplicate Detection...")
    
    # Initialize DuplicateDetector
    detector = DuplicateDetector(
        similarity_threshold=0.7,
        confidence_threshold=0.6,
        # use_clustering=True, # Note: Constructor signature might verify this
    )

    # Detect duplicate candidates (pairwise)
    print("  Running pairwise detection...")
    candidates = detector.detect_duplicates(entities)
    print(f"  Found {len(candidates)} duplicate candidate(s)")
    for candidate in candidates:
        print(f"    {candidate.entity1['name']} <-> {candidate.entity2['name']}")
        print(f"      Similarity: {candidate.similarity_score:.3f}, Confidence: {candidate.confidence:.3f}")

    # Detect duplicate groups
    print("  Running group detection...")
    duplicate_groups = detector.detect_duplicate_groups(entities)
    print(f"  Found {len(duplicate_groups)} duplicate group(s)")
    for i, group in enumerate(duplicate_groups, 1):
        names = [e['name'] for e in group.entities]
        print(f"    Group {i}: {names} (confidence: {group.confidence:.3f})")

    # Incremental detection
    print("  Running incremental detection...")
    existing_entities = entities[:3]
    new_entities = entities[3:]
    incremental_candidates = detector.incremental_detect(new_entities, existing_entities, threshold=0.7)
    print(f"  Found {len(incremental_candidates)} incremental duplicate(s)")
    for candidate in incremental_candidates:
        print(f"    {candidate.entity1['name']} duplicates {candidate.entity2['name']} (confidence: {candidate.confidence:.3f})")

    # Example: Similarity Calculation
    print("\n[4] Testing Similarity Calculation...")
    
    # Initialize SimilarityCalculator
    calculator = SimilarityCalculator(
        string_weight=0.4,
        property_weight=0.3,
        relationship_weight=0.2,
        embedding_weight=0.1,
    )

    # Calculate overall similarity (multi-factor)
    entity1, entity2 = entities[0], entities[1]
    result = calculator.calculate_similarity(entity1, entity2)
    print(f"  Overall Similarity: {result.score:.3f}")
    print(f"  Components: {result.components}")

    # String similarity methods
    str1, str2 = "Apple Inc.", "Apple Inc"
    for method in ["levenshtein", "jaro_winkler", "cosine"]:
        score = calculator.calculate_string_similarity(str1, str2, method=method)
        print(f"  {method}: {score:.3f}")

    # Property and relationship similarity
    prop_score = calculator.calculate_property_similarity(entity1, entity2)
    rel_score = calculator.calculate_relationship_similarity(entity1, entity2)
    print(f"  Property Similarity: {prop_score:.3f}")
    print(f"  Relationship Similarity: {rel_score:.3f}")

    # Batch similarity calculation
    similarity_pairs = calculator.batch_calculate_similarity(entities, threshold=0.5)
    print(f"  Found {len(similarity_pairs)} similar pairs (threshold >= 0.5)")
    for e1, e2, score in similarity_pairs:
        print(f"    {e1['name']} <-> {e2['name']}: {score:.3f}")

    # Example: Entity Merging
    print("\n[5] Testing Entity Merging...")
    
    # Initialize EntityMerger
    merger = EntityMerger(preserve_provenance=True)

    # Merge duplicates (automatic detection)
    print("  Merging duplicates (automatic)...")
    merge_operations = merger.merge_duplicates(entities)
    print(f"  Original entities: {len(entities)}")
    print(f"  Merge operations: {len(merge_operations)}")
    for i, op in enumerate(merge_operations, 1):
        print(f"    Operation {i}: Merged {len(op.source_entities)} entities -> {op.merged_entity.get('name')}")
        if op.merge_result.conflicts:
            print(f"      Conflicts: {len(op.merge_result.conflicts)}")

    # Merge with specific strategy
    print("  Merging with KEEP_MOST_COMPLETE strategy...")
    operations = merger.merge_duplicates(entities, strategy=MergeStrategy.KEEP_MOST_COMPLETE)
    print(f"  Merged using KEEP_MOST_COMPLETE: {len(operations)} operations")

    # Merge specific group
    print("  Merging specific group...")
    duplicate_entities = [entities[0], entities[1]]
    operation = merger.merge_entity_group(duplicate_entities, strategy=MergeStrategy.KEEP_FIRST)
    print(f"  Merged group: {[e['name'] for e in operation.source_entities]} -> {operation.merged_entity['name']}")

    # Get merge history
    history = merger.get_merge_history()
    print(f"  Total merge operations in history: {len(history)}")

    # Validate merge quality
    if operations:
        validation = merger.validate_merge_quality(operations[0])
        print(f"  Validation: Valid={validation['valid']}, Quality={validation['quality_score']:.3f}")
        
    print("\n=== Deduplication Verification Completed Successfully ===")

if __name__ == "__main__":
    verify_deduplication_notebook()
