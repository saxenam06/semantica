
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("verify_conflict_complete")

def verify_conflict_complete():
    print("=== Starting Complete Conflict Module Verification ===\n")

    try:
        from semantica.conflicts import (
            ConflictDetector, ConflictResolver, ConflictAnalyzer,
            SourceTracker, InvestigationGuideGenerator, SourceReference,
            method_registry, ResolutionResult, ConflictsConfig, conflicts_config
        )
        from semantica.conflicts.methods import (
            detect_conflicts, resolve_conflicts, analyze_conflicts,
            track_sources, generate_investigation_guide,
            list_available_methods, get_conflict_method
        )
        print("[1] Imports successful.\n")
    except ImportError as e:
        print(f"Error importing semantica.conflicts: {e}")
        return

    # --- Step 1: Conflict Detection (Comprehensive) ---
    print("[2] Testing Comprehensive Conflict Detection...")
    detector = ConflictDetector(
        confidence_threshold=0.7,
        track_provenance=True,
        conflict_fields={"Company": ["name", "founded", "revenue"]}
    )

    entities = [
        {"id": "e1", "name": "Apple Inc.", "founded": 1976, "type": "Company", 
         "source": "wikipedia", "confidence": 0.9, "metadata": {"timestamp": datetime(2023, 1, 15)}},
        {"id": "e1", "name": "Apple Incorporated", "founded": 1976, "type": "Company",
         "source": "official_site", "confidence": 0.95, "metadata": {"timestamp": datetime(2023, 3, 20)}},
        {"id": "e1", "name": "Apple Inc.", "founded": 1977, "type": "Company",
         "source": "news", "confidence": 0.7, "metadata": {"timestamp": datetime(2023, 2, 10)}},
        {"id": "e2", "name": "Microsoft", "type": "Company", "founded": 1975, "source": "source1"},
        {"id": "e2", "name": "Microsoft Corporation", "type": "Organization", 
         "founded": 1975, "source": "source2"},
    ]

    # 1.1 Value Conflicts
    value_conflicts = detector.detect_value_conflicts(entities, "name")
    print(f"  Value Conflicts Detected: {len(value_conflicts)}")
    
    # 1.2 Type Conflicts
    type_conflicts = detector.detect_type_conflicts(entities)
    print(f"  Type Conflicts Detected: {len(type_conflicts)}")
    
    # 1.3 Temporal Conflicts (founded date mismatch)
    # Note: detect_temporal_conflicts usually checks for logical temporal issues or changing values over time
    temporal_conflicts = detector.detect_temporal_conflicts(entities)
    print(f"  Temporal Conflicts Detected: {len(temporal_conflicts)}")

    # 1.6 General Detection
    all_conflicts = detector.detect_conflicts(entities)
    print(f"  Total Conflicts Detected (General): {len(all_conflicts)}")
    
    # --- Step 2: Source Tracking ---
    print("\n[3] Testing Source Tracking...")
    tracker = SourceTracker()
    source1 = SourceReference(document="wikipedia", timestamp=datetime(2023, 1, 15), confidence=0.9)
    source2 = SourceReference(document="official_site", timestamp=datetime(2023, 3, 20), confidence=0.95)
    
    tracker.track_property_source("e1", "name", "Apple Inc.", source1)
    tracker.track_property_source("e1", "name", "Apple Incorporated", source2)
    tracker.set_source_credibility("wikipedia", 0.85)
    tracker.set_source_credibility("official_site", 0.95)
    
    sources = tracker.get_property_sources("e1", "name")
    print(f"  Sources for e1.name: {len(sources.sources) if sources else 0}")
    
    # --- Step 3: Conflict Resolution (Advanced) ---
    print("\n[4] Testing Advanced Conflict Resolution...")
    resolver = ConflictResolver(default_strategy="voting", source_tracker=tracker)
    
    if value_conflicts:
        # Credibility Weighted
        results = resolver.resolve_conflicts(value_conflicts, strategy="credibility_weighted")
        for r in results:
            if r.resolved:
                print(f"  Resolved (Credibility): {r.resolved_value} (Conf: {r.confidence:.2f})")
    
    # --- Step 4: Conflict Analysis ---
    print("\n[5] Testing Conflict Analyzer...")
    analyzer = ConflictAnalyzer()
    analysis = analyzer.analyze_conflicts(all_conflicts)
    print(f"  Analysis Keys: {list(analysis.keys())}")
    print(f"  Total Conflicts in Analysis: {analysis.get('total_conflicts', 0)}")
    
    # Check insights report
    try:
        insights = analyzer.generate_insights_report(all_conflicts)
        print(f"  Insights Report Generated (Length: {len(insights)})")
    except Exception as e:
        print(f"  Error generating insights report: {e}")

    # --- Step 5: Investigation Guides ---
    print("\n[6] Testing Investigation Guide Generator...")
    guide_generator = InvestigationGuideGenerator(source_tracker=tracker)
    
    if value_conflicts:
        guide = guide_generator.generate_guide(value_conflicts[0])
        print(f"  Guide Generated for: {guide.conflict_id}")
        print(f"  Recommended Actions: {guide.recommended_actions}")
        
        checklist = guide_generator.export_investigation_checklist(guide, format="markdown")
        print(f"  Checklist Exported (Markdown length: {len(checklist)})")

    # --- Step 6: Methods Module (Functional API) ---
    print("\n[7] Testing Functional API...")
    try:
        # Detection
        func_conflicts = detect_conflicts(entities, method="value", property_name="name")
        print(f"  Functional Detect (Value): {len(func_conflicts)}")
        
        # Resolution
        if func_conflicts:
            func_results = resolve_conflicts(func_conflicts, method="voting")
            print(f"  Functional Resolve (Voting): {len(func_results)}")
            
        # Analysis
        func_analysis = analyze_conflicts(all_conflicts, method="pattern")
        print(f"  Functional Analyze (Pattern): {len(func_analysis) if func_analysis else 0} patterns found")
        
    except Exception as e:
        print(f"  Error in Functional API: {e}")

    # --- Step 7: Method Registry ---
    print("\n[8] Testing Method Registry...")
    def custom_resolution_func(conflicts, **kwargs):
        results = []
        for conflict in conflicts:
            res = ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=True,
                resolved_value="CUSTOM_VALUE",
                resolution_strategy="custom_test"
            )
            results.append(res)
        return results

    method_registry.register("resolution", "custom_test", custom_resolution_func)
    print("  Registered 'custom_test' method.")
    
    reg_methods = method_registry.list_all("resolution")
    if "custom_test" in reg_methods.get("resolution", []):
        print("  Verified 'custom_test' in registry.")
        
    # Test usage
    if value_conflicts:
        custom_res = resolve_conflicts(value_conflicts, method="custom_test")
        print(f"  Custom Resolution Result: {custom_res[0].resolved_value}")
        
    method_registry.unregister("resolution", "custom_test")
    print("  Unregistered 'custom_test'.")

    # --- Step 8: Configuration ---
    print("\n[9] Testing Configuration...")
    conflicts_config.set("confidence_threshold", 0.85)
    val = conflicts_config.get("confidence_threshold")
    print(f"  Config Value Set/Get: {val}")
    
    print("\n=== Complete Conflict Module Verification Finished ===")

if __name__ == "__main__":
    verify_conflict_complete()
