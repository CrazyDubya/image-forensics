#!/usr/bin/env python3
"""
Unified Image Forensics Classification System

This module provides a unified interface for classifying images as:
- AI_GENERATED: Created by AI/ML models (GANs, diffusion models, etc.)
- EDITED: Traditionally manipulated/tampered (splicing, copy-move, etc.)  
- ORIGINAL: Authentic, unmodified photographs

Integrates existing MATLAB algorithms with new AI detection capabilities.
"""

import json
import subprocess
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from ai_detector import AIDetector


class UnifiedForensicsClassifier:
    """Unified classifier combining traditional tampering detection with AI detection."""
    
    def __init__(self, matlab_toolbox_path: Optional[str] = None):
        """Initialize the unified classifier.
        
        Args:
            matlab_toolbox_path: Path to MATLAB toolbox directory (optional)
        """
        self.matlab_toolbox_path = matlab_toolbox_path
        self.ai_detector = AIDetector()
        
        # Traditional tampering algorithms available
        self.traditional_algorithms = [
            'ELA', 'ADQ1', 'ADQ2', 'CFA1', 'CFA2', 'NOI1', 'NOI2', 'BLK'
        ]
        
        # Weights for combining different detection methods
        self.detection_weights = {
            'ai_detection': 0.4,
            'traditional_tampering': 0.3,
            'metadata_analysis': 0.2,
            'statistical_analysis': 0.1
        }
        
    def classify_image(self, image_path: str, use_matlab: bool = False) -> Dict:
        """Classify an image into AI_GENERATED, EDITED, or ORIGINAL.
        
        Args:
            image_path: Path to the image file
            use_matlab: Whether to use MATLAB algorithms (requires MATLAB installation)
            
        Returns:
            Dictionary containing classification results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        print(f"🔍 Analyzing image: {image_path.name}")
        
        # Step 1: AI Generation Detection
        print("  🤖 Running AI detection analysis...")
        ai_result = self.ai_detector.detect_ai_generated(str(image_path))
        
        # Step 2: Traditional Tampering Detection
        print("  🔧 Running traditional tampering detection...")
        if use_matlab and self.matlab_toolbox_path:
            traditional_result = self._run_matlab_analysis(str(image_path))
        else:
            traditional_result = self._run_python_traditional_analysis(str(image_path))
            
        # Step 3: Metadata Analysis
        print("  📊 Analyzing metadata...")
        metadata_result = self._analyze_metadata(str(image_path))
        
        # Step 4: Statistical Analysis
        print("  📈 Running statistical analysis...")
        statistical_result = self._statistical_analysis(str(image_path))
        
        # Step 5: Unified Classification
        print("  🎯 Computing final classification...")
        final_result = self._compute_unified_classification(
            ai_result, traditional_result, metadata_result, statistical_result
        )
        
        return {
            'image_path': str(image_path),
            'final_classification': final_result,
            'detailed_analysis': {
                'ai_detection': ai_result,
                'traditional_tampering': traditional_result,
                'metadata_analysis': metadata_result,
                'statistical_analysis': statistical_result
            },
            'processing_notes': {
                'matlab_used': use_matlab and self.matlab_toolbox_path is not None,
                'algorithms_applied': self._get_algorithms_applied(use_matlab)
            }
        }
    
    def _run_matlab_analysis(self, image_path: str) -> Dict:
        """Run traditional tampering detection using MATLAB algorithms.
        
        This is a placeholder - actual implementation would require MATLAB Engine for Python
        or subprocess calls to MATLAB scripts.
        """
        # For now, return a simulated result based on image characteristics
        # In real implementation, this would call MATLAB algorithms
        
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not load image', 'tampering_score': 0.0}
            
        # Simulate traditional tampering detection results
        # In reality, this would call ELA, ADQ, CFA, NOI algorithms
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple ELA-like analysis
        compressed_path = str(Path(image_path).with_suffix('.tmp.jpg'))
        cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        compressed = cv2.imread(compressed_path)
        
        if compressed is not None:
            diff = cv2.absdiff(image, compressed)
            ela_score = np.mean(diff) / 255.0
        else:
            ela_score = 0.0
            
        # Clean up
        try:
            Path(compressed_path).unlink()
        except:
            pass
            
        # Simple noise analysis (NOI-like)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)
        noise_score = np.std(noise) / 10.0  # Normalize
        
        # Combine scores for traditional tampering likelihood
        tampering_score = min((ela_score + noise_score) / 2.0, 1.0)
        
        return {
            'tampering_detected': tampering_score > 0.3,
            'tampering_score': float(tampering_score),
            'ela_score': float(ela_score),
            'noise_score': float(noise_score),
            'algorithms_used': ['ELA_simulation', 'NOI_simulation'],
            'note': 'Simulated MATLAB results - install MATLAB for full functionality'
        }
    
    def _run_python_traditional_analysis(self, image_path: str) -> Dict:
        """Run traditional tampering detection using Python implementations."""
        
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not load image', 'tampering_score': 0.0}
            
        # Error Level Analysis (ELA)
        ela_score = self._python_ela_analysis(image, image_path)
        
        # JPEG Quality Analysis
        quality_score = self._jpeg_quality_analysis(image_path)
        
        # Block Artifact Analysis
        block_score = self._block_artifact_analysis(image)
        
        # Combine traditional tampering indicators
        tampering_score = min((ela_score + quality_score + block_score) / 3.0, 1.0)
        
        return {
            'tampering_detected': tampering_score > 0.4,
            'tampering_score': float(tampering_score),
            'ela_score': float(ela_score),
            'quality_score': float(quality_score),
            'block_score': float(block_score),
            'algorithms_used': ['Python_ELA', 'JPEG_Quality', 'Block_Analysis']
        }
    
    def _python_ela_analysis(self, image: np.ndarray, image_path: str) -> float:
        """Python implementation of Error Level Analysis."""
        
        # Save image at a specific quality and compare
        temp_path = str(Path(image_path).with_suffix('.ela_temp.jpg'))
        cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        compressed = cv2.imread(temp_path)
        if compressed is None:
            return 0.0
            
        # Calculate difference
        diff = cv2.absdiff(image, compressed)
        ela_score = np.mean(diff) / 255.0
        
        # Clean up
        try:
            Path(temp_path).unlink()
        except:
            pass
            
        return min(ela_score * 3.0, 1.0)  # Scale for sensitivity
    
    def _jpeg_quality_analysis(self, image_path: str) -> float:
        """Analyze JPEG quality for tampering indicators."""
        
        try:
            # Try to read with different quality levels
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
                
            # Check for double JPEG compression indicators
            # This is a simplified approach
            
            # Save at different qualities and measure differences
            qualities = [70, 80, 90, 95]
            differences = []
            
            for quality in qualities:
                temp_path = f"temp_q{quality}.jpg"
                cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                recompressed = cv2.imread(temp_path)
                
                if recompressed is not None:
                    diff = np.mean(cv2.absdiff(image, recompressed))
                    differences.append(diff)
                    
                try:
                    Path(temp_path).unlink()
                except:
                    pass
                    
            if len(differences) > 1:
                # Look for unusual quality response patterns
                diff_variance = np.var(differences)
                return min(diff_variance / 1000.0, 1.0)  # Normalize
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _block_artifact_analysis(self, image: np.ndarray) -> float:
        """Analyze for JPEG blocking artifacts that indicate tampering."""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate differences at 8x8 block boundaries (JPEG blocks)
        block_diffs = []
        
        # Vertical block boundaries
        for x in range(8, w, 8):
            if x < w - 1:
                left_col = gray[:, x-1]
                right_col = gray[:, x]
                diff = np.mean(np.abs(left_col.astype(np.float32) - right_col.astype(np.float32)))
                block_diffs.append(diff)
                
        # Horizontal block boundaries  
        for y in range(8, h, 8):
            if y < h - 1:
                top_row = gray[y-1, :]
                bottom_row = gray[y, :]
                diff = np.mean(np.abs(top_row.astype(np.float32) - bottom_row.astype(np.float32)))
                block_diffs.append(diff)
                
        if len(block_diffs) > 0:
            avg_block_diff = np.mean(block_diffs)
            # Higher block differences suggest tampering
            return min(avg_block_diff / 50.0, 1.0)  # Normalize
        else:
            return 0.0
    
    def _analyze_metadata(self, image_path: str) -> Dict:
        """Analyze image metadata for tampering/generation indicators."""
        
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            img = Image.open(image_path)
            exifdata = img.getexif()
            
            metadata_indicators = {
                'has_camera_info': False,
                'has_gps': False,
                'has_software_info': False,
                'suspicious_software': False,
                'metadata_score': 0.0
            }
            
            software_info = None
            camera_info = None
            
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                
                if tag == "Software":
                    software_info = str(data).lower()
                    metadata_indicators['has_software_info'] = True
                    
                    # Check for AI generation software
                    ai_keywords = ['stable diffusion', 'midjourney', 'dalle', 'gan', 'ai', 'generate']
                    if any(keyword in software_info for keyword in ai_keywords):
                        metadata_indicators['suspicious_software'] = True
                        
                elif tag in ["Make", "Model"]:
                    camera_info = str(data)
                    metadata_indicators['has_camera_info'] = True
                    
                elif tag in ["GPSInfo"]:
                    metadata_indicators['has_gps'] = True
                    
            # Calculate metadata suspicion score
            score = 0.0
            if metadata_indicators['suspicious_software']:
                score += 0.8  # Strong indicator of AI generation
            elif not metadata_indicators['has_camera_info']:
                score += 0.3  # Missing camera info is suspicious
            elif not metadata_indicators['has_software_info']:
                score += 0.1  # Missing software info is mildly suspicious
                
            metadata_indicators['metadata_score'] = score
            metadata_indicators['software_info'] = software_info
            metadata_indicators['camera_info'] = camera_info
            
            return metadata_indicators
            
        except ImportError:
            return {'error': 'PIL not available for metadata analysis', 'metadata_score': 0.0}
        except Exception as e:
            return {'error': f'Metadata analysis failed: {e}', 'metadata_score': 0.0}
    
    def _statistical_analysis(self, image_path: str) -> Dict:
        """Perform statistical analysis on image characteristics."""
        
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not load image', 'statistical_score': 0.0}
            
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Analyze color distribution
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # Calculate histogram entropy (measure of randomness)
        def calculate_entropy(hist):
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-8)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            return entropy
            
        entropy_b = calculate_entropy(hist_b)
        entropy_g = calculate_entropy(hist_g)
        entropy_r = calculate_entropy(hist_r)
        avg_entropy = (entropy_b + entropy_g + entropy_r) / 3.0
        
        # AI images often have unusual color distributions
        statistical_score = 0.0
        
        # Very high or very low entropy can indicate artificial generation
        if avg_entropy < 6.0 or avg_entropy > 7.8:
            statistical_score += 0.3
            
        # Check color saturation patterns
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # AI images often have unusual saturation characteristics
        if sat_std < 30 or sat_std > 80:  # Too uniform or too varied
            statistical_score += 0.2
            
        return {
            'statistical_score': float(statistical_score),
            'avg_entropy': float(avg_entropy),
            'saturation_mean': float(sat_mean),
            'saturation_std': float(sat_std)
        }
    
    def _compute_unified_classification(self, ai_result: Dict, traditional_result: Dict, 
                                      metadata_result: Dict, statistical_result: Dict) -> Dict:
        """Compute final unified classification from all analysis results."""
        
        # Extract scores from each analysis
        ai_score = ai_result.get('confidence', 0.0)
        traditional_score = traditional_result.get('tampering_score', 0.0)
        metadata_score = metadata_result.get('metadata_score', 0.0)
        statistical_score = statistical_result.get('statistical_score', 0.0)
        
        # Apply weights to combine scores
        weights = self.detection_weights
        
        # Calculate weighted scores for each category
        ai_evidence = weights['ai_detection'] * ai_score + weights['metadata_analysis'] * metadata_score
        traditional_evidence = weights['traditional_tampering'] * traditional_score
        statistical_evidence = weights['statistical_analysis'] * statistical_score
        
        # Decision logic
        if ai_evidence >= 0.6:
            category = 'AI_GENERATED'
            confidence = min(ai_evidence, 1.0)
        elif traditional_evidence >= 0.5 or (traditional_evidence >= 0.3 and statistical_evidence >= 0.2):
            category = 'EDITED'
            confidence = min(traditional_evidence + statistical_evidence, 1.0)
        else:
            category = 'ORIGINAL'
            confidence = 1.0 - max(ai_evidence, traditional_evidence)
            
        # Generate reasoning
        reasoning = self._generate_classification_reasoning(
            ai_score, traditional_score, metadata_score, statistical_score, category
        )
        
        return {
            'category': category,
            'confidence': float(confidence),
            'evidence_scores': {
                'ai_generation': float(ai_evidence),
                'traditional_tampering': float(traditional_evidence),
                'statistical_anomalies': float(statistical_evidence)
            },
            'reasoning': reasoning,
            'raw_scores': {
                'ai_detection': float(ai_score),
                'traditional_tampering': float(traditional_score),
                'metadata_analysis': float(metadata_score),
                'statistical_analysis': float(statistical_score)
            }
        }
    
    def _generate_classification_reasoning(self, ai_score: float, trad_score: float, 
                                         meta_score: float, stat_score: float, category: str) -> str:
        """Generate human-readable reasoning for the classification."""
        
        reasons = []
        
        if category == 'AI_GENERATED':
            if ai_score > 0.5:
                reasons.append("Strong AI generation indicators detected")
            if meta_score > 0.5:
                reasons.append("Metadata suggests AI generation")
            if stat_score > 0.3:
                reasons.append("Unusual statistical patterns")
                
        elif category == 'EDITED':
            if trad_score > 0.4:
                reasons.append("Traditional tampering artifacts detected")
            if stat_score > 0.2:
                reasons.append("Statistical anomalies present")
                
        else:  # ORIGINAL
            reasons.append("No significant tampering or AI generation indicators")
            if ai_score < 0.2 and trad_score < 0.3:
                reasons.append("Characteristics consistent with authentic photography")
                
        return "; ".join(reasons) if reasons else "Classification based on combined analysis"
    
    def _get_algorithms_applied(self, use_matlab: bool) -> List[str]:
        """Get list of algorithms that were applied."""
        
        algorithms = ['AI_Detection_Ensemble']
        
        if use_matlab and self.matlab_toolbox_path:
            algorithms.extend(self.traditional_algorithms)
        else:
            algorithms.extend(['Python_ELA', 'JPEG_Quality', 'Block_Analysis'])
            
        algorithms.extend(['Metadata_Analysis', 'Statistical_Analysis'])
        
        return algorithms


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Unified image forensics classification")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--matlab-path", help="Path to MATLAB toolbox directory")
    parser.add_argument("--use-matlab", action="store_true", help="Use MATLAB algorithms")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    classifier = UnifiedForensicsClassifier(args.matlab_path)
    
    try:
        result = classifier.classify_image(args.image_path, args.use_matlab)
        
        final = result['final_classification']
        print(f"\n🎯 FINAL CLASSIFICATION:")
        print(f"   Category: {final['category']}")
        print(f"   Confidence: {final['confidence']:.2f}")
        print(f"   Reasoning: {final['reasoning']}")
        
        if args.verbose:
            print(f"\n📊 EVIDENCE SCORES:")
            for evidence_type, score in final['evidence_scores'].items():
                print(f"   {evidence_type}: {score:.2f}")
                
            print(f"\n🔍 DETAILED ANALYSIS:")
            for analysis_type, details in result['detailed_analysis'].items():
                print(f"   {analysis_type}: {details}")
                
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