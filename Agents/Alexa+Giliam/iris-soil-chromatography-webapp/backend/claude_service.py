import os
import json
from typing import Dict, Any, Optional
import anthropic
from anthropic import Anthropic

# Try to load from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env file
    pass


class ClaudeAnalysisService:
    """
    Service for analyzing chromatography results using Claude AI.
    Provides intelligent insights and interpretations of chromatographic data from analysis results only.
    """
    
    def __init__(self):
        """Initialize the Claude API client."""
        self.client = None
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if self.api_key:
            try:
                self.client = Anthropic(api_key=self.api_key)
                print("✅ Claude API service initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Claude API: {e}")
                self.client = None
        else:
            print("⚠️  ANTHROPIC_API_KEY not found. Claude analysis will be disabled.")
    
    def is_available(self) -> bool:
        """Check if Claude API service is available."""
        return self.client is not None
    
    def analyze_chromatography_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze chromatography results using Claude AI (text-based analysis only).
        
        Args:
            analysis_results: Analysis results from the chromatography analyzer
            
        Returns:
            Dictionary containing Claude's analysis and insights
        """
        if not self.is_available():
            return {
                "error": "Claude API service is not available",
                "available": False,
                "message": "Please set ANTHROPIC_API_KEY environment variable to enable Claude analysis"
            }
        
        if not analysis_results:
            return {
                "error": "No analysis results provided",
                "available": True,
                "message": "Please provide chromatography analysis results for Claude to analyze"
            }
        
        try:
            # Prepare the prompt based on analysis results
            prompt = self._build_results_analysis_prompt(analysis_results)
            
            # Make API call to Claude (text-only)
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse Claude's response
            claude_response = message.content[0].text
            
            # Structure the response with enhanced text report
            text_report = self._generate_text_report(claude_response, analysis_results)
            
            return {
                "available": True,
                "analysis": claude_response,
                "text_report": text_report,
                "model": "claude-3-5-sonnet-20241022",
                "insights": self._extract_insights(claude_response),
                "recommendations": self._extract_recommendations(claude_response)
            }
            
        except Exception as e:
            return {
                "error": f"Claude API error: {str(e)}",
                "available": True,
                "message": "Failed to analyze results with Claude"
            }
    
    def _load_prompt_template(self) -> str:
        """Load the Claude prompt template from file."""
        try:
            prompt_file = os.path.join(os.path.dirname(__file__), 'claude_prompt.txt')
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback to a basic prompt if file not found
            return """
You are an expert chromatographer analyzing computational results from a chromatography image analysis system.

Please analyze the following data and provide insights:

Analysis Status: {status}
Method: {method}
Center: {center}
Zones: {zones_count}

{zones_data}

{summary_data}

Please provide a comprehensive analysis of these chromatography results.
"""

    def _load_chromatogram_reference_table(self) -> str:
        """Load the chromatogram reference table with interpretation rules."""
        try:
            table_file = os.path.join(os.path.dirname(__file__), 'chromatogram_table.md')
            with open(table_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback if reference table not found
            return """
## Reference Information
No detailed reference table available. Using standard chromatographic interpretation principles.
"""
    
    def _build_results_analysis_prompt(self, analysis_results: Dict[str, Any]) -> str:
        """Build the analysis prompt for Claude based on computational results."""
        
        # Load prompt template and reference table
        prompt_template = self._load_prompt_template()
        reference_table = self._load_chromatogram_reference_table()
        
        # Extract key information from results
        zones_count = len(analysis_results.get('zones', {}))
        method = analysis_results.get('analysis_parameters', {}).get('segmentation_method', 'Unknown')
        center = analysis_results.get('center', 'N/A')
        success = analysis_results.get('success', False)
        
        # Build zones data section
        zones_data = ""
        if 'zones' in analysis_results:
            for zone_key, zone_data in analysis_results['zones'].items():
                zones_data += f"""
**Zone {zone_key} ({zone_data.get('zone_full_name', 'Unknown Zone')})**:
- Area: {zone_data.get('area_cm2', 'N/A')} cm²
- Mean Radius: {zone_data.get('mean_radius_cm', 'N/A')} cm
- Thickness: {zone_data.get('thickness_cm', 'N/A')} cm
- Angular Coverage: {zone_data.get('angular_coverage', 'N/A')}°
- Mean Intensity: {zone_data.get('mean_intensity', 'N/A')}
- Contrast: {zone_data.get('contrast', 'N/A')}
- Color Values: {zone_data.get('color', 'N/A')}
"""
        
        # Build summary data section
        summary_data = ""
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            summary_data = f"""
- Total Area: {summary.get('total_area_cm2', 'N/A')} cm²
- Average Zone Area: {summary.get('average_zone_area', 'N/A')} cm²
- Zone Count: {summary.get('total_zones', 'N/A')}
"""
        
        # Add radial analysis data if available
        radial_data = ""
        if 'radial_analysis' in analysis_results:
            radial = analysis_results['radial_analysis']
            radial_data = f"""

### Radial Analysis (Channels & Spikes):
"""
            if 'channel_development' in radial:
                ch = radial['channel_development']
                radial_data += f"""
**Channel Development**:
- Total Channels: {ch.get('total_channels', 'N/A')}
- Average Length: {ch.get('avg_channel_length_cm', 'N/A')} cm
- Density: {ch.get('channel_density', 'N/A')}
- Continuity: {ch.get('avg_continuity', 'N/A')}
"""
            
            if 'spike_development' in radial:
                sp = radial['spike_development']
                radial_data += f"""
**Spike Development**:
- Total Spikes: {sp.get('total_spikes', 'N/A')}
- Density: {sp.get('spike_density', 'N/A')}/10k pixels
- Average Intensity: {sp.get('avg_spike_intensity', 'N/A')}
"""
        
        # Combine reference table with prompt and data
        full_prompt = f"""
{reference_table}

---

{prompt_template.format(
    status='Successful' if success else 'Failed/Incomplete',
    method=method,
    center=center,
    zones_count=zones_count,
    zones_data=zones_data + radial_data,
    summary_data=summary_data
)}

## IMPORTANT INTERPRETATION GUIDELINES

Please use the reference table above to interpret the chromatogram features. Pay special attention to:

1. **Zone Interpretation**: Use the Feature descriptions to understand what each zone (CZ, MZ, Outer) represents
2. **Channel Analysis**: Apply the channels interpretation rules (1=absent, 5=fully developed)
3. **Spike Analysis**: Use spike development guidelines for organic matter assessment
4. **Color Assessment**: Consider the color intensity implications for soil health
5. **Structural Features**: Analyze rings and other patterns according to the reference guidelines

Base your analysis on both the computational data provided and the scientific interpretation framework from the reference table.
"""
        
        return full_prompt
    
    def _extract_insights(self, response: str) -> list:
        """Extract key insights from Claude's response."""
        insights = []
        
        # Simple keyword-based extraction
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['insight:', 'observation:', 'notable:', 'important:']):
                insights.append(line)
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                insights.append(line)
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_recommendations(self, response: str) -> list:
        """Extract recommendations from Claude's response."""
        recommendations = []
        
        # Look for recommendation sections
        lines = response.split('\n')
        in_recommendation_section = False
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'improve', 'optimize']):
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    recommendations.append(line)
                elif ':' in line:
                    recommendations.append(line)
                    in_recommendation_section = True
            elif in_recommendation_section and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                recommendations.append(line)
            elif not line and in_recommendation_section:
                in_recommendation_section = False
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _generate_text_report(self, claude_response: str, analysis_results: Dict[str, Any] = None) -> str:
        """Generate a structured text report combining Claude analysis with computational data."""
        
        # Header
        report = "=" * 80 + "\n"
        report += "CHROMATOGRAPHY ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Add timestamp
        from datetime import datetime
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Analysis Engine: Claude AI + Computational Analysis\n\n"
        
        # Computational Summary (if available)
        if analysis_results:
            report += "COMPUTATIONAL ANALYSIS SUMMARY\n"
            report += "-" * 40 + "\n"
            
            if 'zones' in analysis_results:
                report += f"Total Zones Detected: {len(analysis_results['zones'])}\n"
                
                for zone_key, zone_data in analysis_results['zones'].items():
                    report += f"\nZone {zone_key} ({zone_data.get('zone_full_name', 'Unknown')}):\n"
                    report += f"  • Area: {zone_data.get('area_cm2', 'N/A')} cm²\n"
                    report += f"  • Mean Radius: {zone_data.get('mean_radius_cm', 'N/A')} cm\n"
                    report += f"  • Thickness: {zone_data.get('thickness_cm', 'N/A')} cm\n"
                    report += f"  • Mean Intensity: {zone_data.get('mean_intensity', 'N/A')}\n"
                    report += f"  • Color: {zone_data.get('color', 'N/A')}\n"
            
            if 'summary' in analysis_results:
                summary = analysis_results['summary']
                report += f"\nOverall Statistics:\n"
                report += f"  • Total Area: {summary.get('total_area_cm2', 'N/A')} cm²\n"
                report += f"  • Average Zone Size: {summary.get('average_zone_area', 'N/A')} cm²\n"
                report += f"  • Method Used: {analysis_results.get('method', 'N/A')}\n"
            
            report += "\n"
        
        # Claude AI Analysis
        report += "EXPERT AI ANALYSIS\n"
        report += "-" * 40 + "\n"
        report += claude_response + "\n\n"
        
        # Key Insights Section
        insights = self._extract_insights(claude_response)
        if insights:
            report += "KEY INSIGHTS\n"
            report += "-" * 40 + "\n"
            for i, insight in enumerate(insights, 1):
                report += f"{i}. {insight.strip()}\n"
            report += "\n"
        
        # Recommendations Section
        recommendations = self._extract_recommendations(claude_response)
        if recommendations:
            report += "RECOMMENDATIONS\n"
            report += "-" * 40 + "\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec.strip()}\n"
            report += "\n"
        
        # Footer
        report += "=" * 80 + "\n"
        report += "End of Report\n"
        report += "=" * 80 + "\n"
        
        return report
    
    def analyze_comparison(self, analysis_results_list: list) -> Dict[str, Any]:
        """
        Compare multiple chromatography analysis results using Claude AI.
        
        Args:
            analysis_results_list: List of analysis results from the chromatography analyzer
            
        Returns:
            Dictionary containing comparative analysis
        """
        if not self.is_available():
            return {
                "error": "Claude API service is not available",
                "available": False
            }
        
        if len(analysis_results_list) > 6:  # Limit to prevent token overflow
            return {"error": "Maximum 6 analyses supported for comparison"}
        
        if not analysis_results_list:
            return {"error": "No analysis results provided for comparison"}
        
        try:
            # Build comparison prompt with all analysis data
            comparison_prompt = f"""
You are an expert chromatographer comparing {len(analysis_results_list)} chromatography analysis results. Please provide a comprehensive comparative analysis.

## ANALYSIS RESULTS DATA

"""
            
            # Add each analysis result
            for i, analysis_results in enumerate(analysis_results_list, 1):
                comparison_prompt += f"### Analysis {i}:\n"
                comparison_prompt += f"- Method: {analysis_results.get('method', 'N/A')}\n"
                comparison_prompt += f"- Success: {analysis_results.get('success', False)}\n"
                comparison_prompt += f"- Zones Detected: {len(analysis_results.get('zones', {}))}\n"
                
                if 'zones' in analysis_results:
                    for zone_key, zone_data in analysis_results['zones'].items():
                        comparison_prompt += f"  - Zone {zone_key}: Area={zone_data.get('area_cm2', 'N/A')} cm², "
                        comparison_prompt += f"Radius={zone_data.get('mean_radius_cm', 'N/A')} cm, "
                        comparison_prompt += f"Intensity={zone_data.get('mean_intensity', 'N/A')}\n"
                
                if 'summary' in analysis_results:
                    summary = analysis_results['summary']
                    comparison_prompt += f"  - Total Area: {summary.get('total_area_cm2', 'N/A')} cm²\n"
                
                comparison_prompt += "\n"
            
            comparison_prompt += """
## COMPARATIVE ANALYSIS REQUEST

Please provide:

### 1. CONSISTENCY ASSESSMENT
- How consistent are the results across analyses?
- Which parameters show good reproducibility?
- Which parameters show high variability?

### 2. QUALITY RANKING
- Rank the analyses by apparent quality/reliability
- Explain the ranking criteria used
- Identify the best and worst performing analyses

### 3. METHOD PERFORMANCE
- Overall assessment of method performance across samples
- Identification of systematic issues or biases
- Assessment of method robustness

### 4. PATTERN IDENTIFICATION
- Common trends across all analyses
- Outliers or anomalous results
- Consistent problem areas

### 5. OPTIMIZATION INSIGHTS
- What can be learned from this comparison?
- Recommendations for method improvement
- Suggestions for quality control measures

### 6. STATISTICAL INSIGHTS
- Assessment of precision based on replicate measurements
- Identification of the most reliable parameters
- Recommendations for data reporting and interpretation

Provide specific, actionable insights based on the comparative analysis of these computational results.
"""
            
            # Make API call
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2500,
                messages=[{"role": "user", "content": comparison_prompt}]
            )
            
            return {
                "available": True,
                "comparison_analysis": message.content[0].text,
                "model": "claude-3-5-sonnet-20241022",
                "analyses_compared": len(analysis_results_list)
            }
            
        except Exception as e:
            return {
                "error": f"Claude API error: {str(e)}",
                "available": True,
                "message": "Failed to perform comparison analysis"
            }
    
    def analyze_chromatography_feral(self, claude_analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate poetic interpretation using the feral prompt based on Claude analysis results.
        
        Args:
            claude_analysis_data: Analysis results from the Claude AI analysis endpoint
            
        Returns:
            Dictionary containing poetic utterances and health score
        """
        if not self.is_available():
            return {
                "error": "Claude API service is not available",
                "available": False,
                "message": "Please set ANTHROPIC_API_KEY environment variable to enable Claude analysis"
            }
        
        if not claude_analysis_data:
            return {
                "error": "No Claude analysis data provided",
                "available": True,
                "message": "Please provide Claude analysis results for feral analysis"
            }
        
        try:
            # Calculate health score from Claude analysis insights
            health_score = self._calculate_health_score_from_claude(claude_analysis_data)
            
            # Load the feral prompt template
            feral_prompt = self._load_feral_prompt_template()
            
            # Build the complete prompt with health score and Claude insights
            claude_insights = claude_analysis_data.get('analysis', 'No analysis available')
            complete_prompt = f"{feral_prompt}\n\nSoil health score: {health_score:.2f}\n\nClaude analysis insights:\n{claude_insights[:500]}..."
            
            # Make API call to Claude
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": complete_prompt
                    }
                ]
            )
            
            # Parse Claude's response - it should be JSON with utterances
            claude_response = message.content[0].text
            
            # Try to parse as JSON directly since the prompt returns JSON
            try:
                import json
                parsed_response = json.loads(claude_response)
                # Return the JSON directly - it should have utterances field
                if isinstance(parsed_response, dict) and 'utterances' in parsed_response:
                    return parsed_response
                else:
                    # Fallback if format is unexpected
                    return {"utterances": [claude_response]}
            except json.JSONDecodeError:
                # If not valid JSON, wrap in utterances array
                utterances = [line.strip() for line in claude_response.split('\n') if line.strip()]
                return {"utterances": utterances[:8]}  # Take first 8 lines if more than 8
            
        except Exception as e:
            return {
                "error": f"Claude API error: {str(e)}",
                "available": True,
                "message": "Failed to generate feral analysis"
            }
    
    def _load_feral_prompt_template(self) -> str:
        """Load the feral prompt template from file."""
        try:
            prompt_file = os.path.join(os.path.dirname(__file__), 'claude_prompt_feral.txt')
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback prompt if file not found
            return """You are a poetic interpreter of soil chromatogram data. Based on the health score of a soil sample (ranging from 0.0 to 1.0), generate 8 separate one-line utterances reflecting the soil's voice. Return as JSON array."""
    
    def _calculate_health_score_from_claude(self, claude_analysis_data: Dict[str, Any]) -> float:
        """Calculate soil health score from Claude analysis insights."""
        try:
            score = 0.5  # Base score
            
            # Extract Claude's analysis text
            analysis_text = claude_analysis_data.get('analysis', '').lower()
            
            # Positive indicators
            positive_keywords = [
                'healthy', 'good', 'excellent', 'strong', 'well-developed',
                'balanced', 'optimal', 'rich', 'diverse', 'robust'
            ]
            
            # Negative indicators
            negative_keywords = [
                'poor', 'weak', 'degraded', 'limited', 'stressed',
                'deficient', 'problematic', 'concerning', 'inadequate', 'depleted'
            ]
            
            # Count positive and negative indicators
            positive_count = sum(1 for keyword in positive_keywords if keyword in analysis_text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in analysis_text)
            
            # Adjust score based on keyword analysis
            score += (positive_count * 0.05)  # Each positive keyword adds 0.05
            score -= (negative_count * 0.05)   # Each negative keyword subtracts 0.05
            
            # Look for specific soil health indicators in Claude's analysis
            if 'channel development' in analysis_text:
                if any(word in analysis_text for word in ['good', 'strong', 'well']):
                    score += 0.1
                elif any(word in analysis_text for word in ['poor', 'weak', 'limited']):
                    score -= 0.1
            
            if 'organic matter' in analysis_text:
                if any(word in analysis_text for word in ['high', 'rich', 'abundant']):
                    score += 0.1
                elif any(word in analysis_text for word in ['low', 'poor', 'limited']):
                    score -= 0.1
            
            # Check for zone development assessment
            if 'zone' in analysis_text:
                if any(word in analysis_text for word in ['clear', 'distinct', 'well-defined']):
                    score += 0.05
                elif any(word in analysis_text for word in ['unclear', 'poor', 'indistinct']):
                    score -= 0.05
            
            # Ensure score stays within bounds
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default middle score if calculation fails

    def _calculate_soil_health_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate a simple soil health score from chromatography analysis results."""
        try:
            score = 0.5  # Base score
            
            # Factor in zone distribution
            zones = analysis_results.get('zones', {})
            if len(zones) >= 3:
                score += 0.1  # Good zone separation
            
            # Factor in total area coverage
            summary = analysis_results.get('summary', {})
            total_area = summary.get('total_area_cm2', 0)
            if total_area > 20:  # Assuming good development area
                score += 0.1
            elif total_area < 5:  # Poor development
                score -= 0.2
            
            # Factor in radial features if available
            radial = analysis_results.get('radial_analysis', {})
            if radial:
                # Channel development indicates good biological activity
                if 'channel_development' in radial:
                    ch = radial['channel_development']
                    channel_density = ch.get('channel_density', 0)
                    if channel_density > 0.5:
                        score += 0.1
                
                # Uniformity indicates balanced soil
                if 'radial_uniformity' in radial:
                    ru = radial['radial_uniformity']
                    consistency = ru.get('radial_consistency', 0)
                    if consistency > 0.7:
                        score += 0.1
                    elif consistency < 0.3:
                        score -= 0.1
            
            # Ensure score stays within bounds
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default middle score if calculation fails


# Global instance
claude_service = ClaudeAnalysisService()
