#!/usr/bin/env python3
"""
AI-Generated Image Detection Module

This module implements algorithms specifically designed to detect AI-generated images,
distinguishing them from traditionally edited or original photographs.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import argparse


class AIDetector:
    """AI-generated image detection using multiple feature-based approaches."""
    
    def __init__(self):
        """Initialize AI detector with default parameters."""
        self.algorithms = {
            'noise_analysis': self._noise_inconsistency_analysis,
            'frequency_analysis': self._frequency_domain_analysis, 
            'texture_analysis': self._texture_pattern_analysis,
            'ensemble': self._ensemble_classification
        }
        
    def detect_ai_generated(self, image_path: str, algorithm: str = 'ensemble') -> Dict:
        """Detect if an image is AI-generated.
        
        Args:
            image_path: Path to the image file
            algorithm: Detection algorithm to use ('noise_analysis', 'frequency_analysis', 
                      'texture_analysis', or 'ensemble')
                      
        Returns:
            Dictionary containing detection results
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Run detection algorithm
        result = self.algorithms[algorithm](image, image_path)
        
        # Add metadata
        result['image_path'] = image_path
        result['algorithm_used'] = algorithm
        result['image_shape'] = image.shape
        
        return result
    
    def _noise_inconsistency_analysis(self, image: np.ndarray, image_path: str) -> Dict:
        """Analyze noise patterns to detect AI generation.
        
        AI-generated images often have unusual noise characteristics compared to
        natural camera noise or traditional editing artifacts.
        """
        # Convert to grayscale for noise analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and calculate noise residual
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise_residual = gray.astype(np.float32) - blurred.astype(np.float32)
        
        # Calculate noise statistics
        noise_std = np.std(noise_residual)
        noise_mean = np.mean(np.abs(noise_residual))
        
        # Analyze noise distribution in different frequency bands
        # High-quality AI generators often produce very low noise or unusual noise patterns
        h, w = gray.shape
        
        # Divide into blocks and analyze local noise variance
        block_size = 32
        block_variances = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise_residual[y:y+block_size, x:x+block_size]
                block_variances.append(np.var(block))
                
        # AI-generated images often have very uniform noise characteristics
        noise_uniformity = 1.0 - (np.std(block_variances) / (np.mean(block_variances) + 1e-8))
        
        # Calculate detection score
        # Very low noise or highly uniform noise suggests AI generation
        ai_score = 0.0
        
        if noise_std < 2.0:  # Very low noise
            ai_score += 0.4
        if noise_uniformity > 0.8:  # Very uniform noise
            ai_score += 0.3
        if noise_mean < 1.5:  # Unusually clean
            ai_score += 0.3
            
        return {
            'category': 'AI_GENERATED' if ai_score > 0.5 else 'NATURAL_OR_EDITED',
            'confidence': min(ai_score, 1.0),
            'details': {
                'noise_std': float(noise_std),
                'noise_mean': float(noise_mean),
                'noise_uniformity': float(noise_uniformity),
                'reasoning': self._get_noise_reasoning(noise_std, noise_uniformity, noise_mean)
            }
        }
    
    def _frequency_domain_analysis(self, image: np.ndarray, image_path: str) -> Dict:
        """Analyze frequency domain characteristics for AI detection.
        
        AI-generated images often have distinct frequency patterns compared to
        natural images or traditionally edited images.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Analyze different frequency bands
        # High frequencies (edges and details)
        high_freq_mask = np.zeros((h, w))
        radius_high = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 >= radius_high**2
        high_freq_mask[mask] = 1
        high_freq_energy = np.mean(magnitude_spectrum * high_freq_mask)
        
        # Low frequencies (smooth regions)
        low_freq_mask = np.zeros((h, w))
        radius_low = min(h, w) // 8
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius_low**2
        low_freq_mask[mask] = 1
        low_freq_energy = np.mean(magnitude_spectrum * low_freq_mask)
        
        # Mid frequencies
        mid_freq_mask = np.zeros((h, w))
        radius_mid_inner = min(h, w) // 8
        radius_mid_outer = min(h, w) // 4
        mask = ((x - center_x)**2 + (y - center_y)**2 >= radius_mid_inner**2) & \
               ((x - center_x)**2 + (y - center_y)**2 <= radius_mid_outer**2)
        mid_freq_mask[mask] = 1
        mid_freq_energy = np.mean(magnitude_spectrum * mid_freq_mask)
        
        # Calculate frequency ratios
        total_energy = np.mean(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / total_energy
        low_freq_ratio = low_freq_energy / total_energy
        mid_freq_ratio = mid_freq_energy / total_energy
        
        # AI-generated images often have unusual frequency distributions
        ai_score = 0.0
        
        # Very high low-frequency content (over-smooth)
        if low_freq_ratio > 0.7:
            ai_score += 0.3
            
        # Unusual high-frequency patterns
        if high_freq_ratio < 0.1:  # Too little detail
            ai_score += 0.4
        elif high_freq_ratio > 0.4:  # Artificial high-frequency content
            ai_score += 0.2
            
        # Check for periodic patterns (common in AI generation)
        # Simple check for regular patterns in frequency domain
        freq_variance = np.var(magnitude_spectrum)
        if freq_variance < np.mean(magnitude_spectrum) * 0.5:
            ai_score += 0.3
            
        return {
            'category': 'AI_GENERATED' if ai_score > 0.5 else 'NATURAL_OR_EDITED',
            'confidence': min(ai_score, 1.0),
            'details': {
                'high_freq_ratio': float(high_freq_ratio),
                'low_freq_ratio': float(low_freq_ratio),
                'mid_freq_ratio': float(mid_freq_ratio),
                'freq_variance': float(freq_variance),
                'reasoning': self._get_frequency_reasoning(high_freq_ratio, low_freq_ratio, freq_variance)
            }
        }
    
    def _texture_pattern_analysis(self, image: np.ndarray, image_path: str) -> Dict:
        """Analyze texture patterns for AI generation detection.
        
        AI-generated images often have distinct texture characteristics.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features using Local Binary Patterns concept
        # Simplified implementation for basic texture analysis
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze gradient distribution
        grad_mean = np.mean(gradient_magnitude)
        grad_std = np.std(gradient_magnitude)
        
        # Calculate texture regularity
        # AI images often have either very regular or very irregular textures
        h, w = gray.shape
        block_size = 16
        texture_variances = []
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gradient_magnitude[y:y+block_size, x:x+block_size]
                texture_variances.append(np.var(block))
                
        texture_uniformity = 1.0 - (np.std(texture_variances) / (np.mean(texture_variances) + 1e-8))
        
        # Analyze edge consistency
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Calculate AI likelihood
        ai_score = 0.0
        
        # Very uniform textures (common in AI)
        if texture_uniformity > 0.85:
            ai_score += 0.4
            
        # Unusual edge characteristics
        if edge_density < 0.02:  # Too few edges
            ai_score += 0.3
        elif edge_density > 0.15:  # Too many edges
            ai_score += 0.2
            
        # Very low gradient variation (over-smooth)
        if grad_std < grad_mean * 0.3:
            ai_score += 0.3
            
        return {
            'category': 'AI_GENERATED' if ai_score > 0.5 else 'NATURAL_OR_EDITED',
            'confidence': min(ai_score, 1.0),
            'details': {
                'texture_uniformity': float(texture_uniformity),
                'gradient_mean': float(grad_mean),
                'gradient_std': float(grad_std),
                'edge_density': float(edge_density),
                'reasoning': self._get_texture_reasoning(texture_uniformity, edge_density, grad_std, grad_mean)
            }
        }
    
    def _ensemble_classification(self, image: np.ndarray, image_path: str) -> Dict:
        """Combine multiple detection methods for more robust classification."""
        
        # Run all individual algorithms
        noise_result = self._noise_inconsistency_analysis(image, image_path)
        freq_result = self._frequency_domain_analysis(image, image_path)
        texture_result = self._texture_pattern_analysis(image, image_path)
        
        # Combine scores with weights
        weights = {
            'noise': 0.4,
            'frequency': 0.3,
            'texture': 0.3
        }
        
        combined_score = (
            weights['noise'] * noise_result['confidence'] +
            weights['frequency'] * freq_result['confidence'] +
            weights['texture'] * texture_result['confidence']
        )
        
        # Classification thresholds
        if combined_score >= 0.7:
            category = 'AI_GENERATED'
            confidence = combined_score
        elif combined_score >= 0.3:
            category = 'LIKELY_EDITED'
            confidence = 0.6  # Medium confidence for edited
        else:
            category = 'LIKELY_ORIGINAL'
            confidence = 1.0 - combined_score
            
        return {
            'category': category,
            'confidence': min(confidence, 1.0),
            'ensemble_score': float(combined_score),
            'individual_results': {
                'noise_analysis': noise_result,
                'frequency_analysis': freq_result,
                'texture_analysis': texture_result
            },
            'reasoning': self._get_ensemble_reasoning(noise_result, freq_result, texture_result, combined_score)
        }
    
    def _get_noise_reasoning(self, noise_std: float, noise_uniformity: float, noise_mean: float) -> str:
        """Generate human-readable reasoning for noise analysis."""
        reasons = []
        
        if noise_std < 2.0:
            reasons.append("Very low noise levels (typical of AI generation)")
        if noise_uniformity > 0.8:
            reasons.append("Highly uniform noise distribution (unusual for natural images)")
        if noise_mean < 1.5:
            reasons.append("Unusually clean image (possible AI generation)")
            
        return "; ".join(reasons) if reasons else "Normal noise characteristics"
    
    def _get_frequency_reasoning(self, high_freq: float, low_freq: float, freq_var: float) -> str:
        """Generate human-readable reasoning for frequency analysis."""
        reasons = []
        
        if low_freq > 0.7:
            reasons.append("High low-frequency content (over-smooth)")
        if high_freq < 0.1:
            reasons.append("Insufficient high-frequency detail")
        if high_freq > 0.4:
            reasons.append("Artificial high-frequency patterns")
        if freq_var < 50:  # Arbitrary threshold
            reasons.append("Regular frequency patterns")
            
        return "; ".join(reasons) if reasons else "Normal frequency characteristics"
    
    def _get_texture_reasoning(self, texture_unif: float, edge_density: float, grad_std: float, grad_mean: float) -> str:
        """Generate human-readable reasoning for texture analysis."""
        reasons = []
        
        if texture_unif > 0.85:
            reasons.append("Very uniform texture patterns")
        if edge_density < 0.02:
            reasons.append("Insufficient edge detail")
        if edge_density > 0.15:
            reasons.append("Excessive edge artifacts")
        if grad_std < grad_mean * 0.3:
            reasons.append("Low gradient variation (over-smooth)")
            
        return "; ".join(reasons) if reasons else "Normal texture characteristics"
    
    def _get_ensemble_reasoning(self, noise_result: Dict, freq_result: Dict, texture_result: Dict, score: float) -> str:
        """Generate human-readable reasoning for ensemble classification."""
        reasons = []
        
        if noise_result['confidence'] > 0.5:
            reasons.append("Suspicious noise patterns")
        if freq_result['confidence'] > 0.5:
            reasons.append("Unusual frequency characteristics")
        if texture_result['confidence'] > 0.5:
            reasons.append("Artificial texture patterns")
            
        if score >= 0.7:
            return f"Strong indicators of AI generation: {'; '.join(reasons)}"
        elif score >= 0.3:
            return f"Possible editing detected: {'; '.join(reasons) if reasons else 'Mixed indicators'}"
        else:
            return "Characteristics consistent with original photography"


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Detect AI-generated images")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument(
        "--algorithm", 
        choices=['noise_analysis', 'frequency_analysis', 'texture_analysis', 'ensemble'],
        default='ensemble',
        help="Detection algorithm to use"
    )
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    detector = AIDetector()
    
    try:
        print(f"🔍 Analyzing image: {args.image_path}")
        print(f"📊 Using algorithm: {args.algorithm}")
        
        result = detector.detect_ai_generated(args.image_path, args.algorithm)
        
        print(f"\n🎯 DETECTION RESULT:")
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        
        if 'reasoning' in result:
            print(f"   Reasoning: {result['reasoning']}")
            
        if args.verbose:
            print(f"\n📋 DETAILED ANALYSIS:")
            if 'details' in result:
                for key, value in result['details'].items():
                    print(f"   {key}: {value}")
                    
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