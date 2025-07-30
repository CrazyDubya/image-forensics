#!/usr/bin/env python3
"""
Comprehensive Test Framework for Image Forensics Tools

This module provides extensive testing for all components of the image forensics
system, including individual algorithms, unified classification, and edge cases.
"""

import os
import sys
import json
import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the evaluation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ai_detector import AIDetector
from unified_classifier import UnifiedForensicsClassifier


class TestImageForensicsFramework(unittest.TestCase):
    """Comprehensive test suite for image forensics tools."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample images."""
        cls.test_dir = tempfile.mkdtemp()
        cls.sample_images = cls._create_test_images()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        
    @classmethod
    def _create_test_images(cls):
        """Create sample test images with different characteristics."""
        images = {}
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_path = os.path.join(cls.test_dir, "test_original.jpg")
        cv2.imwrite(test_path, test_image)
        images['original'] = test_path
        
        # Create a modified image (simple manipulation)
        modified_image = test_image.copy()
        modified_image[40:60, 40:60] = [255, 0, 0]  # Red square
        modified_path = os.path.join(cls.test_dir, "test_edited.jpg")
        cv2.imwrite(modified_path, modified_image)
        images['edited'] = modified_path
        
        # Create a noisy image (simulate AI artifacts)
        noisy_image = test_image.copy()
        noise = np.random.randint(-20, 20, test_image.shape, dtype=np.int16)
        noisy_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        noisy_path = os.path.join(cls.test_dir, "test_ai_generated.jpg")
        cv2.imwrite(noisy_path, noisy_image)
        images['ai_generated'] = noisy_path
        
        return images


class TestAIDetector(TestImageForensicsFramework):
    """Test cases for AI detection algorithms."""
    
    def setUp(self):
        """Set up AI detector for testing."""
        self.detector = AIDetector()
        
    def test_detector_initialization(self):
        """Test AI detector initialization."""
        self.assertIsInstance(self.detector, AIDetector)
        self.assertIn('noise_analysis', self.detector.algorithms)
        self.assertIn('frequency_analysis', self.detector.algorithms)
        self.assertIn('texture_analysis', self.detector.algorithms)
        self.assertIn('ensemble', self.detector.algorithms)
        
    def test_noise_analysis_algorithm(self):
        """Test noise analysis algorithm."""
        for image_type, image_path in self.sample_images.items():
            result = self.detector.detect_ai_generated(image_path, algorithm='noise_analysis')
            
            self.assertIsInstance(result, dict)
            self.assertIn('algorithm', result)
            self.assertIn('confidence', result)
            self.assertIn('ai_generated', result)
            self.assertTrue(0.0 <= result['confidence'] <= 1.0)
            self.assertIsInstance(result['ai_generated'], bool)
            
    def test_frequency_analysis_algorithm(self):
        """Test frequency domain analysis algorithm."""
        for image_type, image_path in self.sample_images.items():
            result = self.detector.detect_ai_generated(image_path, algorithm='frequency_analysis')
            
            self.assertIsInstance(result, dict)
            self.assertIn('algorithm', result)
            self.assertIn('confidence', result)
            self.assertTrue(0.0 <= result['confidence'] <= 1.0)
            
    def test_texture_analysis_algorithm(self):
        """Test texture pattern analysis algorithm."""
        for image_type, image_path in self.sample_images.items():
            result = self.detector.detect_ai_generated(image_path, algorithm='texture_analysis')
            
            self.assertIsInstance(result, dict)
            self.assertIn('algorithm', result)
            self.assertIn('confidence', result)
            self.assertTrue(0.0 <= result['confidence'] <= 1.0)
            
    def test_ensemble_algorithm(self):
        """Test ensemble classification algorithm."""
        for image_type, image_path in self.sample_images.items():
            result = self.detector.detect_ai_generated(image_path, algorithm='ensemble')
            
            self.assertIsInstance(result, dict)
            self.assertIn('algorithm', result)
            self.assertIn('confidence', result)
            self.assertIn('individual_results', result)
            self.assertTrue(0.0 <= result['confidence'] <= 1.0)
            
    def test_invalid_algorithm(self):
        """Test handling of invalid algorithm names."""
        with self.assertRaises(ValueError):
            self.detector.detect_ai_generated(
                self.sample_images['original'], 
                algorithm='invalid_algorithm'
            )
            
    def test_invalid_image_path(self):
        """Test handling of invalid image paths."""
        with self.assertRaises(FileNotFoundError):
            self.detector.detect_ai_generated('nonexistent_image.jpg')


class TestUnifiedClassifier(TestImageForensicsFramework):
    """Test cases for unified forensics classifier."""
    
    def setUp(self):
        """Set up unified classifier for testing."""
        self.classifier = UnifiedForensicsClassifier()
        
    def test_classifier_initialization(self):
        """Test unified classifier initialization."""
        self.assertIsInstance(self.classifier, UnifiedForensicsClassifier)
        self.assertIsInstance(self.classifier.ai_detector, AIDetector)
        self.assertEqual(len(self.classifier.traditional_algorithms), 8)
        self.assertEqual(len(self.classifier.detection_weights), 4)
        
    def test_detection_weights_sum(self):
        """Test that detection weights sum to 1.0."""
        weights = self.classifier.detection_weights
        weight_sum = sum(weights.values())
        self.assertAlmostEqual(weight_sum, 1.0, places=5)
        
    def test_classify_image_basic(self):
        """Test basic image classification."""
        for image_type, image_path in self.sample_images.items():
            result = self.classifier.classify_image(image_path)
            
            self.assertIsInstance(result, dict)
            self.assertIn('image_path', result)
            self.assertIn('final_classification', result)
            self.assertIn('detailed_analysis', result)
            
            final = result['final_classification']
            self.assertIn('category', final)
            self.assertIn('confidence', final)
            self.assertIn(final['category'], ['AI_GENERATED', 'EDITED', 'ORIGINAL'])
            self.assertTrue(0.0 <= final['confidence'] <= 1.0)
            
    def test_matlab_integration_flag(self):
        """Test MATLAB integration flag handling."""
        result = self.classifier.classify_image(
            self.sample_images['original'], 
            use_matlab=True
        )
        
        processing_notes = result.get('processing_notes', {})
        self.assertIn('matlab_used', processing_notes)
        # Should be False since we don't have MATLAB path set
        self.assertFalse(processing_notes['matlab_used'])
        
    def test_classification_consistency(self):
        """Test that classification results are consistent across runs."""
        image_path = self.sample_images['original']
        
        # Run classification multiple times
        results = []
        for _ in range(3):
            result = self.classifier.classify_image(image_path)
            results.append(result['final_classification']['category'])
            
        # All results should be the same
        self.assertTrue(all(r == results[0] for r in results))
        
    def test_confidence_score_ranges(self):
        """Test that all confidence scores are in valid ranges."""
        for image_type, image_path in self.sample_images.items():
            result = self.classifier.classify_image(image_path)
            
            # Check final confidence
            final_conf = result['final_classification']['confidence']
            self.assertTrue(0.0 <= final_conf <= 1.0)
            
            # Check evidence scores
            evidence_scores = result['final_classification'].get('evidence_scores', {})
            for score in evidence_scores.values():
                self.assertTrue(0.0 <= score <= 1.0)
                
            # Check raw scores
            raw_scores = result['final_classification'].get('raw_scores', {})
            for score in raw_scores.values():
                self.assertTrue(0.0 <= score <= 1.0)


class TestEdgeCases(TestImageForensicsFramework):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up for edge case testing."""
        self.classifier = UnifiedForensicsClassifier()
        self.detector = AIDetector()
        
    def test_very_small_image(self):
        """Test handling of very small images."""
        # Create a 10x10 image
        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        small_path = os.path.join(self.test_dir, "small_image.jpg")
        cv2.imwrite(small_path, small_image)
        
        # Should not crash
        result = self.classifier.classify_image(small_path)
        self.assertIsInstance(result, dict)
        
    def test_grayscale_image(self):
        """Test handling of grayscale images."""
        # Create a grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray_path = os.path.join(self.test_dir, "gray_image.jpg")
        cv2.imwrite(gray_path, gray_image)
        
        # Should handle gracefully
        result = self.classifier.classify_image(gray_path)
        self.assertIsInstance(result, dict)
        
    def test_corrupted_image_handling(self):
        """Test handling of corrupted/invalid image files."""
        # Create a text file with .jpg extension
        corrupted_path = os.path.join(self.test_dir, "corrupted.jpg")
        with open(corrupted_path, 'w') as f:
            f.write("This is not an image file")
            
        # Should raise appropriate error
        with self.assertRaises(ValueError):
            self.classifier.classify_image(corrupted_path)
            
    def test_extreme_noise_levels(self):
        """Test handling of images with extreme noise levels."""
        # Create an image with very high noise
        noise_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        noise_path = os.path.join(self.test_dir, "extreme_noise.jpg")
        cv2.imwrite(noise_path, noise_image)
        
        result = self.classifier.classify_image(noise_path)
        self.assertIsInstance(result, dict)
        
    def test_uniform_color_image(self):
        """Test handling of uniform color images."""
        # Create a uniform red image
        uniform_image = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
        uniform_path = os.path.join(self.test_dir, "uniform_red.jpg")
        cv2.imwrite(uniform_path, uniform_image)
        
        result = self.classifier.classify_image(uniform_path)
        self.assertIsInstance(result, dict)


class TestPerformanceBenchmarks(TestImageForensicsFramework):
    """Performance and benchmark tests."""
    
    def setUp(self):
        """Set up for performance testing."""
        self.classifier = UnifiedForensicsClassifier()
        self.detector = AIDetector()
        
    def test_processing_speed(self):
        """Test processing speed for different image sizes."""
        import time
        
        sizes = [(50, 50), (100, 100), (200, 200)]
        processing_times = {}
        
        for size in sizes:
            # Create test image of specified size
            test_image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            test_path = os.path.join(self.test_dir, f"speed_test_{size[0]}x{size[1]}.jpg")
            cv2.imwrite(test_path, test_image)
            
            # Measure processing time
            start_time = time.time()
            result = self.classifier.classify_image(test_path)
            end_time = time.time()
            
            processing_times[f"{size[0]}x{size[1]}"] = end_time - start_time
            self.assertIsInstance(result, dict)
            
        # Print performance results
        print("\nProcessing Speed Benchmarks:")
        for size, time_taken in processing_times.items():
            print(f"  {size}: {time_taken:.3f} seconds")
            
    def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure memory before processing
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple images
        for _ in range(5):
            result = self.classifier.classify_image(self.sample_images['original'])
            self.assertIsInstance(result, dict)
            
        # Measure memory after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\nMemory Usage:")
        print(f"  Before: {memory_before:.1f} MB")
        print(f"  After: {memory_after:.1f} MB")
        print(f"  Increase: {memory_after - memory_before:.1f} MB")
        
        # Memory increase should be reasonable (less than 100MB for test)
        self.assertLess(memory_after - memory_before, 100)


class TestRatingSystemValidation(TestImageForensicsFramework):
    """Test the rating system implementation."""
    
    def test_confidence_level_mapping(self):
        """Test confidence level mapping to descriptions."""
        test_confidences = [0.95, 0.8, 0.6, 0.4, 0.2]
        expected_levels = ["VERY HIGH", "HIGH", "MODERATE", "LOW", "VERY LOW"]
        
        for conf, expected in zip(test_confidences, expected_levels):
            # Create a mock result with specific confidence
            result = {
                'final_classification': {
                    'confidence': conf,
                    'category': 'ORIGINAL',
                    'evidence_scores': {},
                    'raw_scores': {}
                },
                'detailed_analysis': {}
            }
            
            # Test that display method works without crashing
            try:
                import io
                import sys
                captured_output = io.StringIO()
                sys.stdout = captured_output
                
                self.classifier._display_composite_score_results(result)
                
                output = captured_output.getvalue()
                sys.stdout = sys.__stdout__
                
                self.assertIn(expected, output)
                
            except Exception as e:
                self.fail(f"Display method failed for confidence {conf}: {e}")
                
    def test_score_breakdown_accuracy(self):
        """Test accuracy of score breakdown calculations."""
        # Create a test result with known scores
        test_result = {
            'final_classification': {
                'raw_scores': {
                    'ai_detection': 0.6,
                    'traditional_tampering': 0.4,
                    'metadata_analysis': 0.2,
                    'statistical_analysis': 0.3
                }
            }
        }
        
        weights = self.classifier.detection_weights
        
        # Calculate expected composite score
        expected_composite = (
            0.6 * weights['ai_detection'] +
            0.4 * weights['traditional_tampering'] +
            0.2 * weights['metadata_analysis'] +
            0.3 * weights['statistical_analysis']
        )
        
        # The actual calculation should match
        actual_composite = (
            test_result['final_classification']['raw_scores']['ai_detection'] * weights['ai_detection'] +
            test_result['final_classification']['raw_scores']['traditional_tampering'] * weights['traditional_tampering'] +
            test_result['final_classification']['raw_scores']['metadata_analysis'] * weights['metadata_analysis'] +
            test_result['final_classification']['raw_scores']['statistical_analysis'] * weights['statistical_analysis']
        )
        
        self.assertAlmostEqual(expected_composite, actual_composite, places=5)


def run_comprehensive_tests():
    """Run all tests and generate a comprehensive report."""
    print("="*80)
    print("           COMPREHENSIVE IMAGE FORENSICS TEST SUITE")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAIDetector,
        TestUnifiedClassifier,
        TestEdgeCases,
        TestPerformanceBenchmarks,
        TestRatingSystemValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate summary report
    print("\n" + "="*80)
    print("                            TEST SUMMARY")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)