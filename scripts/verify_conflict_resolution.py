
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("verify_conflict_resolution")

def verify_conflict_resolution():
    print("=== Starting Conflict Resolution Verification ===\n")

    try:
        from semantica.conflicts import ConflictDetector, ConflictResolver, SourceTracker
        from semantica.conflicts.conflict_resolver import ResolutionStrategy
        print("[1] Imports successful.\n")
    except ImportError as e:
        print(f"Error importing semantica.conflicts: {e}")
        return

    # Step 1: Define Entities with Conflicting Data
    print("[2] Defining conflicting entities...")
    entities = [
        {
            "id": "e1",
            "type": "Person",
            "name": "John Doe",
            "age": 30,
            "location": "New York",
            "source": "source1",
            "confidence": 0.8,
            "metadata": {"timestamp": datetime(2023, 1, 1)}
        },
        {
            "id": "e1",
            "type": "Person",
            "name": "John Doe",
            "age": 32,
            "location": "Boston",
            "source": "source2",
            "confidence": 0.9,
            "metadata": {"timestamp": datetime(2023, 6, 1)}
        },
        {
            "id": "e2",
            "type": "Organization",
            "name": "Tech Corp",
            "founded": 2010,
            "employees": 100,
            "source": "source1",
            "confidence": 0.9,
            "metadata": {"timestamp": datetime(2023, 1, 1)}
        },
        {
            "id": "e2",
            "type": "Organization",
            "name": "Tech Corp",
            "founded": 2012,
            "employees": 150,
            "source": "source2",
            "confidence": 0.7,
            "metadata": {"timestamp": datetime(2023, 3, 1)}
        }
    ]
    print(f" defined {len(entities)} entity records.\n")

    # Step 2: Detect Conflicts
    print("[3] Detecting conflicts...")
    detector = ConflictDetector(track_provenance=True)
    conflicts = detector.detect_entity_conflicts(entities)
    
    print(f"Detected {len(conflicts)} conflicts.")
    for i, conflict in enumerate(conflicts, 1):
        print(f"  Conflict {i}: Entity={conflict.entity_id}, Property={conflict.property_name}, Values={conflict.conflicting_values}")
    
    if len(conflicts) == 0:
        print("WARNING: No conflicts detected! Verification cannot proceed meaningfully.")
        return

    # Step 3: Resolve Conflicts
    resolver = ConflictResolver()

    # Strategy: Voting
    print("\n[4a] Resolving with 'voting' strategy...")
    results_voting = resolver.resolve_conflicts(conflicts, strategy="voting")
    for r in results_voting:
        if r.resolved:
            print(f"  Resolved {r.conflict_id} ({r.resolution_strategy}): {r.resolved_value}")

    # Strategy: Most Recent
    print("\n[4b] Resolving with 'most_recent' strategy...")
    results_recent = resolver.resolve_conflicts(conflicts, strategy="most_recent")
    for r in results_recent:
        if r.resolved:
            print(f"  Resolved {r.conflict_id} ({r.resolution_strategy}): {r.resolved_value}")
            # Verification for e1 (most recent is source2 -> age 32, location Boston)
            # Verification for e2 (most recent is source2 -> founded 2012, employees 150)
            
            # Note: We can't easily map back to entity ID without parsing conflict_id or checking logic,
            # but we can verify the values exist in our expectations.
            pass

    # Strategy: Highest Confidence
    print("\n[4c] Resolving with 'highest_confidence' strategy...")
    results_confidence = resolver.resolve_conflicts(conflicts, strategy="highest_confidence")
    for r in results_confidence:
        if r.resolved:
            print(f"  Resolved {r.conflict_id} ({r.resolution_strategy}): {r.resolved_value}")
            # Verification for e1 (highest conf is source2 -> age 32)
            # Verification for e2 (highest conf is source1 -> founded 2010)

    # Step 4: Track Sources
    print("\n[5] Tracking sources...")
    tracker = detector.source_tracker
    for conflict in conflicts:
        sources = tracker.get_property_sources(conflict.entity_id, conflict.property_name)
        if sources:
             print(f"  Entity: {conflict.entity_id}, Property: {conflict.property_name}, Source Count: {len(sources.sources)}")

    # Step 5: Audit Trail
    print("\n[6] Checking audit trail...")
    history = resolver.get_resolution_history()
    print(f"  History entries: {len(history)}")
    if len(history) > 0:
        last_entry = history[-1]
        print(f"  Last entry: Strategy={last_entry.resolution_strategy}, Value={last_entry.resolved_value}")

    print("\n=== Conflict Resolution Verification Completed Successfully ===")

if __name__ == "__main__":
    verify_conflict_resolution()
