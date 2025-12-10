"""
Topic Modeling Module
=====================
Extract recurring themes and topics from interview transcripts using BERTopic.

Features:
- Unsupervised topic discovery
- Topic labeling with key phrases
- Visualization export
- Category mapping to institutional areas
"""

import warnings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Institutional topic categories
INSTITUTIONAL_CATEGORIES = {
    "curriculum": ["course", "subject", "class", "lesson", "module", "curriculum", "syllabus", 
                   "teaching", "learning", "lecture", "assignment", "project", "exam"],
    "support_services": ["counseling", "guidance", "help", "support", "assistance", "mentor",
                         "advisor", "tutorial", "scholarship", "financial", "health"],
    "facilities": ["library", "laboratory", "computer", "building", "room", "equipment",
                   "facility", "campus", "wifi", "internet", "resource"],
    "faculty": ["professor", "teacher", "instructor", "faculty", "staff", "dean", "sir", "ma'am"],
    "career": ["job", "career", "work", "internship", "employment", "industry", "company",
               "professional", "skills", "opportunity", "future"],
    "student_life": ["friends", "classmates", "organization", "club", "event", "activity",
                     "social", "community", "experience", "life"],
    "academic_stress": ["stress", "pressure", "difficult", "hard", "struggle", "challenge",
                        "overwhelmed", "tired", "deadline", "workload", "anxiety"],
    "admissions": ["admission", "enrollment", "apply", "application", "requirement",
                   "orientation", "entrance", "enroll", "register"]
}

# Topic label templates
TOPIC_LABELS = {
    "curriculum": "📚 Curriculum & Learning",
    "support_services": "🤝 Support Services",
    "facilities": "🏫 Facilities & Resources",
    "faculty": "👩‍🏫 Faculty & Staff",
    "career": "💼 Career & Employment",
    "student_life": "🎓 Student Life",
    "academic_stress": "😰 Academic Challenges",
    "admissions": "📝 Admissions Process",
    "general": "💬 General Feedback"
}


# ============================================================================
# TOPIC MODELER CLASS
# ============================================================================

class TopicModeler:
    """
    Extract and analyze topics from interview transcripts.
    
    Uses BERTopic for unsupervised topic discovery with optional
    mapping to institutional categories.
    """
    
    def __init__(self, use_bertopic: bool = True):
        """
        Initialize the topic modeler.
        
        Args:
            use_bertopic: Whether to use BERTopic (requires GPU for best performance).
                         Falls back to keyword-based extraction if False.
        """
        self.bertopic_model = None
        self.use_bertopic = use_bertopic
        
        if use_bertopic:
            self._load_bertopic()
    
    def _load_bertopic(self):
        """Load BERTopic model."""
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            # Use multilingual model for Taglish support
            embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            
            self.bertopic_model = BERTopic(
                embedding_model=embedding_model,
                language="multilingual",
                calculate_probabilities=True,
                verbose=False
            )
            print("✅ BERTopic loaded")
        except ImportError:
            print("⚠️ BERTopic not available. Using keyword-based extraction.")
            self.use_bertopic = False
        except Exception as e:
            print(f"⚠️ BERTopic failed to load: {e}")
            self.use_bertopic = False
    
    # ------------------------------------------------------------------------
    # MAIN METHODS
    # ------------------------------------------------------------------------
    
    def extract_topics(self, texts: List[str], num_topics: int = 5) -> Dict[str, Any]:
        """
        Extract topics from a list of texts.
        
        Args:
            texts: List of transcript texts
            num_topics: Maximum number of topics to extract
            
        Returns:
            Dict containing topics, frequencies, and visualizations
        """
        if not texts:
            return {"topics": [], "error": "No texts provided"}
        
        if self.use_bertopic and self.bertopic_model:
            return self._extract_with_bertopic(texts, num_topics)
        else:
            return self._extract_with_keywords(texts, num_topics)
    
    def _extract_with_bertopic(self, texts: List[str], num_topics: int) -> Dict[str, Any]:
        """Extract topics using BERTopic."""
        try:
            topics, probs = self.bertopic_model.fit_transform(texts)
            
            # Get topic info
            topic_info = self.bertopic_model.get_topic_info()
            
            # Build results
            results = {
                "method": "bertopic",
                "num_documents": len(texts),
                "topics": [],
                "document_topics": topics
            }
            
            for _, row in topic_info.iterrows():
                if row['Topic'] == -1:  # Skip outlier topic
                    continue
                
                topic_words = self.bertopic_model.get_topic(row['Topic'])
                keywords = [word for word, _ in topic_words[:5]] if topic_words else []
                
                # Map to institutional category
                category = self._map_to_category(keywords)
                
                results["topics"].append({
                    "id": int(row['Topic']),
                    "label": TOPIC_LABELS.get(category, TOPIC_LABELS["general"]),
                    "category": category,
                    "keywords": keywords,
                    "count": int(row['Count']),
                    "percentage": round(row['Count'] / len(texts) * 100, 1)
                })
            
            # Sort by count
            results["topics"] = sorted(results["topics"], key=lambda x: x["count"], reverse=True)[:num_topics]
            
            return results
            
        except Exception as e:
            print(f"BERTopic extraction failed: {e}")
            return self._extract_with_keywords(texts, num_topics)
    
    def _extract_with_keywords(self, texts: List[str], num_topics: int) -> Dict[str, Any]:
        """Extract topics using keyword matching."""
        # Combine all texts
        all_text = " ".join(texts).lower()
        words = all_text.split()
        
        # Count category matches
        category_scores = {}
        for category, keywords in INSTITUTIONAL_CATEGORIES.items():
            score = sum(1 for word in words if any(kw in word for kw in keywords))
            if score > 0:
                category_scores[category] = score
        
        # Build results
        total_matches = sum(category_scores.values()) or 1
        results = {
            "method": "keyword",
            "num_documents": len(texts),
            "topics": []
        }
        
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories[:num_topics]:
            # Get top keywords found
            found_keywords = [
                kw for kw in INSTITUTIONAL_CATEGORIES[category]
                if kw in all_text
            ][:5]
            
            results["topics"].append({
                "id": len(results["topics"]),
                "label": TOPIC_LABELS.get(category, TOPIC_LABELS["general"]),
                "category": category,
                "keywords": found_keywords,
                "count": count,
                "percentage": round(count / total_matches * 100, 1)
            })
        
        return results
    
    def _map_to_category(self, keywords: List[str]) -> str:
        """Map extracted keywords to institutional category."""
        keyword_text = " ".join(keywords).lower()
        
        best_category = "general"
        best_score = 0
        
        for category, category_keywords in INSTITUTIONAL_CATEGORIES.items():
            score = sum(1 for kw in category_keywords if kw in keyword_text)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    # ------------------------------------------------------------------------
    # ANALYSIS METHODS
    # ------------------------------------------------------------------------
    
    def get_topic_distribution(self, topics_result: Dict) -> Dict[str, float]:
        """Get percentage distribution of topics."""
        distribution = {}
        for topic in topics_result.get("topics", []):
            distribution[topic["label"]] = topic["percentage"]
        return distribution
    
    def get_category_summary(self, topics_result: Dict) -> List[Dict]:
        """Get summary by institutional category."""
        summaries = []
        for topic in topics_result.get("topics", []):
            summaries.append({
                "category": topic["category"],
                "label": topic["label"],
                "keywords": topic["keywords"],
                "mentions": topic["count"]
            })
        return summaries
    
    def generate_insights(self, topics_result: Dict) -> List[str]:
        """Generate actionable insights from topic analysis."""
        insights = []
        topics = topics_result.get("topics", [])
        
        if not topics:
            return ["No significant topics detected in the transcripts."]
        
        # Top topic insight
        top_topic = topics[0]
        insights.append(
            f"📊 **Primary Focus**: {top_topic['label']} was the most discussed area "
            f"({top_topic['percentage']}% of mentions)."
        )
        
        # Check for stress indicators
        stress_topics = [t for t in topics if t["category"] == "academic_stress"]
        if stress_topics:
            insights.append(
                f"⚠️ **Attention Required**: Academic stress indicators detected. "
                f"Consider targeted support interventions."
            )
        
        # Career readiness
        career_topics = [t for t in topics if t["category"] == "career"]
        if career_topics:
            insights.append(
                f"💼 **Career Focus**: Students show career orientation. "
                f"Recommend strengthening industry connections."
            )
        
        # Support services
        support_topics = [t for t in topics if t["category"] == "support_services"]
        if support_topics:
            insights.append(
                f"🤝 **Support Awareness**: Students are engaging with support services. "
                f"Monitor satisfaction and accessibility."
            )
        
        return insights


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def extract_topics_simple(texts: List[str]) -> Dict[str, Any]:
    """Quick topic extraction without class initialization."""
    modeler = TopicModeler(use_bertopic=False)  # Fast mode
    return modeler.extract_topics(texts)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("   TOPIC MODELING DEMO")
    print("=" * 60)
    
    # Sample interview excerpts
    sample_texts = [
        "The curriculum is really challenging but I learned a lot about programming.",
        "I struggled with the workload especially during finals week.",
        "The professors are very supportive and always available for consultation.",
        "I'm worried about finding a job after graduation.",
        "The library resources helped me a lot with my thesis research.",
        "Joining student organizations improved my leadership skills.",
        "The internship program prepared me well for the industry.",
        "I experienced a lot of stress balancing academics and personal life.",
    ]
    
    print("\n📝 Sample texts:", len(sample_texts))
    
    modeler = TopicModeler(use_bertopic=False)  # Use keyword mode for demo
    results = modeler.extract_topics(sample_texts)
    
    print(f"\n📊 Extraction method: {results['method']}")
    print(f"📄 Documents analyzed: {results['num_documents']}")
    
    print("\n🏷️ EXTRACTED TOPICS:")
    print("-" * 60)
    
    for topic in results["topics"]:
        print(f"\n{topic['label']}")
        print(f"   Keywords: {', '.join(topic['keywords'])}")
        print(f"   Frequency: {topic['count']} ({topic['percentage']}%)")
    
    print("\n💡 INSIGHTS:")
    print("-" * 60)
    insights = modeler.generate_insights(results)
    for insight in insights:
        print(f"  {insight}")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
