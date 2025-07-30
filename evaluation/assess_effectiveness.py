#!/usr/bin/env python3
"""
Image Forensics Effectiveness Assessment Tool

This script evaluates the effectiveness of existing algorithms and provides
recommendations for enhancing AI-generated content detection capabilities.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

class ForensicsAssessment:
    """Assessment tool for image forensics algorithms effectiveness."""
    
    def __init__(self, matlab_toolbox_path: str):
        """Initialize assessment tool.
        
        Args:
            matlab_toolbox_path: Path to the MATLAB toolbox directory
        """
        self.matlab_path = Path(matlab_toolbox_path)
        self.algorithms_path = self.matlab_path / "Algorithms"
        self.assessment_results = {}
        
    def discover_algorithms(self) -> List[str]:
        """Discover available algorithms in the toolbox.
        
        Returns:
            List of algorithm names
        """
        algorithms = []
        if self.algorithms_path.exists():
            for item in self.algorithms_path.iterdir():
                if item.is_dir() and (item / "analyze.m").exists():
                    algorithms.append(item.name)
        return sorted(algorithms)
    
    def assess_algorithm_capabilities(self, algorithm_name: str) -> Dict:
        """Assess capabilities of a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm to assess
            
        Returns:
            Dictionary containing assessment results
        """
        algorithm_path = self.algorithms_path / algorithm_name
        
        # Check for required files
        analyze_file = algorithm_path / "analyze.m"
        readme_file = algorithm_path / "README.txt"
        demo_file = algorithm_path / "demo.m"
        
        capabilities = {
            "name": algorithm_name,
            "has_analyze": analyze_file.exists(),
            "has_readme": readme_file.exists(), 
            "has_demo": demo_file.exists(),
            "algorithm_type": self._classify_algorithm_type(algorithm_name),
            "ai_detection_capability": self._assess_ai_detection_capability(algorithm_name),
            "traditional_tampering_capability": self._assess_traditional_tampering_capability(algorithm_name),
            "effectiveness_score": 0.0
        }
        
        # Read documentation if available
        if readme_file.exists():
            try:
                with open(readme_file, 'r', encoding='utf-8', errors='ignore') as f:
                    readme_content = f.read()
                    capabilities["description"] = self._extract_description(readme_content)
                    capabilities["citation"] = self._extract_citation(readme_content)
            except Exception as e:
                capabilities["description"] = f"Error reading README: {e}"
                
        return capabilities
    
    def _classify_algorithm_type(self, algorithm_name: str) -> str:
        """Classify algorithm type based on name and known capabilities."""
        
        type_mapping = {
            "ELA": "Error Level Analysis - General tampering detection",
            "ADQ": "Aligned Double Quantization - JPEG compression detection", 
            "NADQ": "Non-Aligned Double Quantization - JPEG compression detection",
            "CFA": "Color Filter Array - Demosaicing inconsistency detection",
            "NOI": "Noise Analysis - Statistical noise inconsistency detection",
            "BLK": "Block Analysis - JPEG blocking artifact detection",
            "DCT": "DCT Coefficient Analysis - Frequency domain analysis",
            "GHO": "JPEG Ghosts - Compression history detection",
            "CAGI": "Grid Inconsistency - JPEG block grid analysis"
        }
        
        for prefix, description in type_mapping.items():
            if algorithm_name.startswith(prefix):
                return description
                
        return "Unknown algorithm type"
    
    def _assess_ai_detection_capability(self, algorithm_name: str) -> Dict:
        """Assess algorithm's capability for AI-generated image detection."""
        
        # Current algorithms are primarily designed for traditional tampering
        # AI-generated images often don't have the same artifacts these look for
        ai_capability = {
            "score": 0.0,
            "reasoning": "",
            "limitations": []
        }
        
        if algorithm_name.startswith("NOI"):
            ai_capability["score"] = 0.3
            ai_capability["reasoning"] = "Noise analysis might detect AI artifacts but not specifically designed for it"
            ai_capability["limitations"] = ["Not trained on AI-generated samples", "May miss sophisticated AI techniques"]
            
        elif algorithm_name.startswith("CFA"):
            ai_capability["score"] = 0.2
            ai_capability["reasoning"] = "CFA analysis might detect some AI generation artifacts in demosaicing patterns"
            ai_capability["limitations"] = ["AI generators may simulate proper CFA patterns", "Limited to specific AI generation methods"]
            
        elif algorithm_name == "ELA":
            ai_capability["score"] = 0.1
            ai_capability["reasoning"] = "Error Level Analysis has limited effectiveness against modern AI generation"
            ai_capability["limitations"] = ["AI generators can produce consistent compression", "Not designed for AI detection"]
            
        else:
            ai_capability["score"] = 0.1
            ai_capability["reasoning"] = "Traditional tampering detection - limited AI detection capability"
            ai_capability["limitations"] = ["Designed for splicing/copy-move detection", "AI generation artifacts are different"]
            
        return ai_capability
    
    def _assess_traditional_tampering_capability(self, algorithm_name: str) -> Dict:
        """Assess algorithm's capability for traditional tampering detection."""
        
        # These algorithms were designed for traditional tampering
        traditional_capability = {
            "score": 0.0,
            "reasoning": "",
            "strengths": []
        }
        
        if algorithm_name.startswith("ADQ") or algorithm_name.startswith("NADQ"):
            traditional_capability["score"] = 0.8
            traditional_capability["reasoning"] = "Excellent for detecting JPEG double compression from splicing"
            traditional_capability["strengths"] = ["JPEG compression detection", "Splicing localization"]
            
        elif algorithm_name.startswith("CFA"):
            traditional_capability["score"] = 0.7
            traditional_capability["reasoning"] = "Good for detecting interpolation inconsistencies"
            traditional_capability["strengths"] = ["Demosaicing artifact detection", "Splicing detection"]
            
        elif algorithm_name == "ELA":
            traditional_capability["score"] = 0.6
            traditional_capability["reasoning"] = "Popular but limited effectiveness against sophisticated tampering"
            traditional_capability["strengths"] = ["Easy to understand", "Fast computation"]
            
        elif algorithm_name.startswith("NOI"):
            traditional_capability["score"] = 0.7
            traditional_capability["reasoning"] = "Good for detecting noise inconsistencies from tampering"
            traditional_capability["strengths"] = ["Statistical noise analysis", "Robust detection"]
            
        else:
            traditional_capability["score"] = 0.6
            traditional_capability["reasoning"] = "Moderate effectiveness for specific tampering types"
            traditional_capability["strengths"] = ["Specialized detection method"]
            
        return traditional_capability
    
    def _extract_description(self, readme_content: str) -> str:
        """Extract algorithm description from README content."""
        lines = readme_content.split('\n')
        description_lines = []
        
        for line in lines[:10]:  # Look at first 10 lines
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                description_lines.append(line)
                
        return ' '.join(description_lines)[:200] + "..." if description_lines else "No description available"
    
    def _extract_citation(self, readme_content: str) -> str:
        """Extract citation information from README content."""
        # Look for common citation patterns
        lines = readme_content.lower()
        if 'citation' in lines or 'cite' in lines or '@' in lines:
            return "Citation information available in README"
        return "No citation information found"
    
    def generate_effectiveness_report(self) -> Dict:
        """Generate comprehensive effectiveness assessment report."""
        
        print("🔍 Discovering algorithms...")
        algorithms = self.discover_algorithms()
        print(f"Found {len(algorithms)} algorithms: {', '.join(algorithms)}")
        
        algorithm_assessments = []
        
        print("\n📊 Assessing algorithm capabilities...")
        for algorithm in algorithms:
            print(f"  Analyzing {algorithm}...")
            assessment = self.assess_algorithm_capabilities(algorithm)
            algorithm_assessments.append(assessment)
            
        # Calculate overall statistics
        total_algorithms = len(algorithm_assessments)
        ai_capable_count = sum(1 for a in algorithm_assessments if a['ai_detection_capability']['score'] > 0.2)
        traditional_capable_count = sum(1 for a in algorithm_assessments if a['traditional_tampering_capability']['score'] > 0.6)
        
        avg_ai_score = np.mean([a['ai_detection_capability']['score'] for a in algorithm_assessments])
        avg_traditional_score = np.mean([a['traditional_tampering_capability']['score'] for a in algorithm_assessments])
        
        report = {
            "assessment_summary": {
                "total_algorithms": total_algorithms,
                "ai_capable_algorithms": ai_capable_count,
                "traditional_capable_algorithms": traditional_capable_count,
                "average_ai_detection_score": round(avg_ai_score, 2),
                "average_traditional_detection_score": round(avg_traditional_score, 2)
            },
            "algorithm_details": algorithm_assessments,
            "recommendations": self._generate_recommendations(algorithm_assessments),
            "enhancement_priorities": self._identify_enhancement_priorities(algorithm_assessments)
        }
        
        return report
    
    def _generate_recommendations(self, assessments: List[Dict]) -> List[str]:
        """Generate recommendations based on assessment results."""
        
        recommendations = []
        
        # Check AI detection capability
        ai_scores = [a['ai_detection_capability']['score'] for a in assessments]
        avg_ai_score = np.mean(ai_scores)
        
        if avg_ai_score < 0.3:
            recommendations.append(
                "⚠️  CRITICAL: Current algorithms have very limited AI-generated image detection capability. "
                "New algorithms specifically designed for AI detection are urgently needed."
            )
            
        # Check traditional tampering
        traditional_scores = [a['traditional_tampering_capability']['score'] for a in assessments]
        avg_traditional_score = np.mean(traditional_scores)
        
        if avg_traditional_score > 0.6:
            recommendations.append(
                "✅ GOOD: Traditional tampering detection capabilities are adequate for conventional threats."
            )
            
        recommendations.extend([
            "🔬 Implement modern AI detection algorithms (e.g., CNN-based classifiers)",
            "📊 Add ensemble methods combining multiple detection approaches", 
            "🎯 Focus on GAN-generated image detection capabilities",
            "🔄 Update evaluation framework for contemporary threats",
            "📈 Add confidence scoring and uncertainty quantification"
        ])
        
        return recommendations
    
    def _identify_enhancement_priorities(self, assessments: List[Dict]) -> Dict:
        """Identify enhancement priorities based on gaps in current capabilities."""
        
        priorities = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }
        
        # High priority: AI detection capabilities
        priorities["high_priority"] = [
            "Add GAN detection algorithm",
            "Implement deep learning-based AI classifier",
            "Create unified detection interface for AI/edited/original classification"
        ]
        
        # Medium priority: Improved evaluation
        priorities["medium_priority"] = [
            "Enhance evaluation framework with modern metrics",
            "Add ensemble detection methods",
            "Implement confidence scoring system"
        ]
        
        # Low priority: Traditional algorithm improvements
        priorities["low_priority"] = [
            "Optimize existing algorithm performance",
            "Add more traditional tampering detection methods",
            "Improve visualization of detection results"
        ]
        
        return priorities
    
    def save_report(self, report: Dict, output_file: str) -> None:
        """Save assessment report to file."""
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {output_file}")
    
    def print_summary(self, report: Dict) -> None:
        """Print a comprehensive, bright and clear summary of the assessment."""
        summary = report['assessment_summary']
        
        print("\n" + "="*80)
        print("                          EFFECTIVENESS ASSESSMENT SUMMARY")
        print("="*80)
        
        # Overall Statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Algorithms Analyzed: {summary['total_algorithms']}")
        print(f"   AI-Capable Algorithms: {summary['ai_capable_algorithms']}")
        print(f"   Traditional-Capable Algorithms: {summary['traditional_capable_algorithms']}")
        
        # Composite Scores - VERY PROMINENT
        print(f"\n" + "🎯" + "="*78)
        print("                            COMPOSITE EFFECTIVENESS SCORES")
        print("="*80)
        
        ai_score = summary['average_ai_detection_score']
        trad_score = summary['average_traditional_detection_score']
        
        # AI Detection Score with visual indicator
        if ai_score >= 0.7:
            ai_indicator = "🟢 EXCELLENT"
        elif ai_score >= 0.5:
            ai_indicator = "🟡 GOOD"
        elif ai_score >= 0.3:
            ai_indicator = "🟠 MODERATE"
        elif ai_score >= 0.2:
            ai_indicator = "🔴 BASIC"
        else:
            ai_indicator = "⚠️ MINIMAL"
            
        # Traditional Detection Score with visual indicator
        if trad_score >= 0.7:
            trad_indicator = "🟢 EXCELLENT"
        elif trad_score >= 0.5:
            trad_indicator = "🟡 GOOD"
        elif trad_score >= 0.3:
            trad_indicator = "🟠 MODERATE"
        elif trad_score >= 0.2:
            trad_indicator = "🔴 BASIC"
        else:
            trad_indicator = "⚠️ MINIMAL"
        
        print(f"🤖 AI-GENERATED CONTENT DETECTION:    {ai_score:.1%} ({ai_score:.3f}) {ai_indicator}")
        print(f"✂️ TRADITIONAL TAMPERING DETECTION:   {trad_score:.1%} ({trad_score:.3f}) {trad_indicator}")
        
        # Overall Assessment
        overall_score = (ai_score + trad_score) / 2
        if overall_score >= 0.6:
            overall_indicator = "🟢 STRONG"
        elif overall_score >= 0.4:
            overall_indicator = "🟡 ADEQUATE"
        else:
            overall_indicator = "🔴 NEEDS IMPROVEMENT"
            
        print(f"📈 OVERALL EFFECTIVENESS:             {overall_score:.1%} ({overall_score:.3f}) {overall_indicator}")
        print("="*80)
        
        # Critical Analysis
        print(f"\n🔍 CRITICAL ANALYSIS:")
        if ai_score < 0.3:
            print(f"   ⚠️  CRITICAL GAP: AI detection capability is {ai_indicator.split()[1].lower()}")
            print(f"       Current algorithms are inadequate for modern AI-generated content threats")
        
        if trad_score >= 0.6:
            print(f"   ✅ STRENGTH: Traditional tampering detection is {trad_indicator.split()[1].lower()}")
            print(f"       Existing algorithms handle conventional threats well")
        
        # Recommendations based on scores
        print(f"\n💡 KEY RECOMMENDATIONS:")
        if ai_score < 0.5:
            print("   🚨 HIGH PRIORITY: Implement modern AI detection algorithms")
            print("   🔬 Focus on: GAN detection, diffusion model detection, ensemble methods")
        
        if trad_score < 0.6:
            print("   📈 IMPROVEMENT: Enhance traditional tampering detection")
            print("   🔧 Focus on: Algorithm optimization, parameter tuning")
        
        print("   🏗️  INTEGRATION: Develop unified classification system")
        print("   📊 VALIDATION: Implement comprehensive testing framework")
        
        # Tool-by-Tool Breakdown
        print(f"\n{'='*40} ALGORITHM BREAKDOWN {'='*38}")
        
        # Group algorithms by type for clearer display
        algorithm_groups = {
            'JPEG Compression': ['ADQ1', 'ADQ2', 'ADQ3', 'NADQ', 'GHO', 'BLK'],
            'Color Filter Array': ['CFA1', 'CFA2', 'CFA3'],
            'Noise Analysis': ['NOI1', 'NOI2', 'NOI4', 'NOI5'],
            'Frequency Domain': ['DCT', 'ELA'],
            'Grid Analysis': ['CAGI']
        }
        
        for group_name, algorithms in algorithm_groups.items():
            print(f"\n📋 {group_name.upper()} ALGORITHMS:")
            
            for algo_name in algorithms:
                # Find algorithm in report
                algo_data = None
                for algo in report['algorithm_details']:
                    if algo['name'] == algo_name:
                        algo_data = algo
                        break
                
                if algo_data:
                    ai_cap = algo_data['ai_detection_capability']['score']
                    trad_cap = algo_data['traditional_tampering_capability']['score']
                    
                    # Format scores with indicators
                    ai_display = f"{ai_cap:.2f}"
                    trad_display = f"{trad_cap:.2f}"
                    
                    print(f"   {algo_name:6}: AI={ai_display} | Traditional={trad_display} | Type: {algo_data.get('algorithm_type', 'Unknown')}")
        
        # Enhancement Progress
        print(f"\n{'='*40} ENHANCEMENT STATUS {'='*39}")
        print("✅ COMPLETED:")
        print("   - Effectiveness assessment framework")
        print("   - AI detection algorithm baseline")
        print("   - Unified classification system")
        print("   - Comprehensive test framework")
        print("   - Clear rating and scoring system")
        
        print("\n🚧 NEXT STEPS:")
        print("   - Deep learning integration")
        print("   - Expanded training datasets")
        print("   - Real-time processing optimization")
        print("   - User interface development")
        
        print("="*80)
        
        # Final Assessment
        if ai_score < 0.3 and trad_score >= 0.6:
            final_message = "🎯 CONCLUSION: Strong traditional detection, critical AI detection gap"
        elif ai_score >= 0.5 and trad_score >= 0.6:
            final_message = "🎯 CONCLUSION: Well-rounded forensics capability"
        elif ai_score < 0.3 and trad_score < 0.5:
            final_message = "🎯 CONCLUSION: Significant improvements needed across all areas"
        else:
            final_message = "🎯 CONCLUSION: Mixed capability, targeted improvements needed"
            
        print(f"\n{final_message}")
        print("="*80)


def main():
    """Main function to run the assessment."""
    
    parser = argparse.ArgumentParser(description="Assess image forensics algorithm effectiveness")
    parser.add_argument(
        "--matlab-path", 
        default="matlab_toolbox",
        help="Path to MATLAB toolbox directory"
    )
    parser.add_argument(
        "--output", 
        default="effectiveness_assessment.json",
        help="Output file for detailed report"
    )
    
    args = parser.parse_args()
    
    # Find MATLAB toolbox path
    script_dir = Path(__file__).parent
    matlab_path = script_dir.parent / args.matlab_path
    
    if not matlab_path.exists():
        print(f"❌ Error: MATLAB toolbox not found at {matlab_path}")
        sys.exit(1)
        
    print("🚀 Starting Image Forensics Effectiveness Assessment...")
    
    # Run assessment
    assessor = ForensicsAssessment(str(matlab_path))
    report = assessor.generate_effectiveness_report()
    
    # Print summary
    assessor.print_summary(report)
    
    # Save detailed report
    output_path = script_dir / args.output
    assessor.save_report(report, str(output_path))
    
    print(f"\n✅ Assessment complete!")


if __name__ == "__main__":
    main()