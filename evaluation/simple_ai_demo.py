#!/usr/bin/env python3
"""
Simple AI Detection Demo (No external dependencies)

This simplified version demonstrates AI detection concepts without requiring
OpenCV or PIL, using only Python standard library and numpy.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict
import argparse


def simple_ai_detection_demo(image_path: str) -> Dict:
    """
    Simplified AI detection demonstration.
    
    This is a conceptual implementation showing how AI detection would work.
    For full functionality, install opencv-python and pillow.
    """
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Get file size and basic info
    file_size = image_path.stat().st_size
    file_ext = image_path.suffix.lower()
    
    # Simple heuristics based on file characteristics
    ai_score = 0.0
    reasons = []
    
    # Check file extension
    if file_ext in ['.png']:
        ai_score += 0.2
        reasons.append("PNG format often used by AI generators")
    
    # Check file size patterns
    if file_size < 100000:  # Very small file
        ai_score += 0.1
        reasons.append("Unusually small file size")
    elif file_size > 5000000:  # Very large file
        ai_score += 0.1
        reasons.append("Unusually large file size")
    
    # Check filename patterns
    filename = image_path.stem.lower()
    ai_keywords = ['generated', 'ai', 'stable', 'diffusion', 'midjourney', 'dalle', 'synthetic']
    for keyword in ai_keywords:
        if keyword in filename:
            ai_score += 0.4
            reasons.append(f"Filename contains AI-related keyword: {keyword}")
            break
    
    # Determine classification
    if ai_score >= 0.5:
        category = 'AI_GENERATED'
    elif ai_score >= 0.2:
        category = 'POSSIBLY_EDITED'
    else:
        category = 'LIKELY_ORIGINAL'
    
    return {
        'image_path': str(image_path),
        'category': category,
        'confidence': min(ai_score, 1.0),
        'file_size': file_size,
        'file_extension': file_ext,
        'reasoning': '; '.join(reasons) if reasons else 'No strong indicators found',
        'note': 'This is a simplified demo. Install opencv-python and pillow for full AI detection capabilities.'
    }


def main():
    parser = argparse.ArgumentParser(description="Simple AI detection demo")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    try:
        result = simple_ai_detection_demo(args.image_path)
        
        print(f"🔍 AI Detection Demo Results:")
        print(f"   Image: {result['image_path']}")
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Reasoning: {result['reasoning']}")
        print(f"   Note: {result['note']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Results saved to: {args.output}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())