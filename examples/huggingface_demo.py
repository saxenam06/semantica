"""
HuggingFace Local Model Usage Demo (Bring Your Own Model)

This script demonstrates how to use the 'semantica' library with local HuggingFace models
for Named Entity Recognition (NER), Relation Extraction (RE), and Triplet Extraction.

Prerequisites:
    pip install transformers torch

Usage:
    python examples/huggingface_demo.py
"""

import sys
import os

# Add project root to path (for running from this dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantica.semantic_extract import NERExtractor, RelationExtractor, TripletExtractor, Entity

def demo_ner():
    print("\n" + "="*50)
    print("NER Demo: Bring Your Own Model (BYOM)")
    print("="*50)
    
    # 1. Initialize NERExtractor with HuggingFace method and a specific model
    # Common models: "dslim/bert-base-NER", "dbmdz/bert-large-cased-finetuned-conll03-english"
    model_name = "dslim/bert-base-NER" 
    print(f"Initializing NERExtractor with model: {model_name}...")
    
    extractor = NERExtractor(
        method="huggingface", 
        model=model_name,
        device="cpu"  # Use "cuda" for GPU
    )
    
    text = "Steve Jobs founded Apple Inc. in Cupertino, California on April 1, 1976."
    print(f"\nInput text: {text}")
    
    try:
        # Note: This will download the model if not cached (approx 400MB)
        print("Extracting entities (this may take a moment on first run)...")
        entities = extractor.extract_entities(text)
        
        print(f"\nExtracted {len(entities)} entities:")
        for ent in entities:
            print(f"  - {ent.text:20} | Type: {ent.label:10} | Conf: {ent.confidence:.2f}")
            
    except Exception as e:
        print(f"Extraction failed (missing dependencies?): {e}")


def demo_relation():
    print("\n" + "="*50)
    print("Relation Extraction Demo: Local Model")
    print("="*50)
    
    # 1. Initialize RelationExtractor
    # Note: Relation extraction usually requires a SequenceClassification model 
    # trained on relation datasets (e.g., TACRED, SemEval).
    # For demo purposes, we'll use a generic placeholder or a widely used one.
    model_name = "semantica/relation-model-v1" # This is hypothetical; replace with real model
    print(f"Initializing RelationExtractor with method='huggingface'...")
    
    extractor = RelationExtractor(
        method="huggingface",
        model=model_name,
        device="cpu"
    )
    
    text = "Steve Jobs founded Apple Inc."
    # Pre-defined entities are usually required for relation extraction
    entities = [
        Entity(text="Steve Jobs", label="PERSON", start_char=0, end_char=10), 
        Entity(text="Apple Inc.", label="ORG", start_char=19, end_char=29)
    ]
    
    print(f"\nInput text: {text}")
    print(f"Entities: {[e.text for e in entities]}")
    
    try:
        print("Extracting relations...")
        # Note: This will fail if the model doesn't exist on HF Hub.
        # In a real scenario, use a valid model ID like "some-user/bert-relation-extraction"
        # For this demo, we just show the call structure.
        relations = extractor.extract_relations(text, entities)
        
        print(f"\nExtracted {len(relations)} relations:")
        for rel in relations:
            print(f"  - {rel.subject.text} --[{rel.predicate}]--> {rel.object.text} (Conf: {rel.confidence:.2f})")
            
    except Exception as e:
        print(f"Note: Relation extraction mock run (model download might fail or be skipped): {e}")


def demo_triplet():
    print("\n" + "="*50)
    print("Triplet Extraction Demo: REBEL (Seq2Seq)")
    print("="*50)
    
    # 1. Initialize TripletExtractor with REBEL model
    # REBEL is a popular model for end-to-end triplet extraction
    model_name = "Babelscape/rebel-large"
    print(f"Initializing TripletExtractor with model: {model_name}...")
    
    extractor = TripletExtractor(
        method="huggingface",
        model=model_name,
        device="cpu"
    )
    
    text = "Apple was founded by Steve Jobs in 1976."
    print(f"\nInput text: {text}")
    
    try:
        print("Extracting triplets (this may take a moment)...")
        triplets = extractor.extract_triplets(text)
        
        print(f"\nExtracted {len(triplets)} triplets:")
        for triplet in triplets:
            print(f"  - ({triplet.subject}, {triplet.predicate}, {triplet.object})")
            
    except Exception as e:
        print(f"Extraction failed (missing dependencies?): {e}")

if __name__ == "__main__":
    print("Starting Semantica HuggingFace Usage Demo...")
    print("Note: This script attempts to download models from Hugging Face Hub.")
    print("Ensure you have an internet connection and 'transformers' installed.")
    
    # Run demos
    # We wrap in try-except to ensure the script doesn't crash the whole session if one fails
    try:
        demo_ner()
    except Exception as e:
        print(f"NER Demo Error: {e}")
        
    try:
        demo_relation()
    except Exception as e:
        print(f"Relation Demo Error: {e}")
        
    try:
        demo_triplet()
    except Exception as e:
        print(f"Triplet Demo Error: {e}")
