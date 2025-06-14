from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime
import logging
import os

class HybridLeadScorer:
    """
    Production-ready ML lead scoring system that works immediately
    Combines rule-based scoring with ML learning from real conversions
    """
    
    def __init__(self):
        self.rule_based_scorer = RuleBasedScorer()
        self.ml_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.conversion_data = []
        self.is_ml_trained = False
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def score_lead(self, lead_data):
        """
        Main scoring function - uses rule-based initially, ML when available
        """
        try:
            # Always get rule-based score (works immediately)
            rule_score = self.rule_based_scorer.score_lead(lead_data)
            
            # If ML model is trained and confident, blend scores
            if self.is_ml_trained and self.ml_model:
                ml_prediction = self._get_ml_prediction(lead_data)
                
                if ml_prediction['confidence'] > 0.7:
                    # Blend rule-based and ML scores
                    final_score = (rule_score['score'] * 0.3) + (ml_prediction['score'] * 0.7)
                    confidence = 'high'
                    method = 'hybrid_ml'
                else:
                    # Use rule-based when ML confidence is low
                    final_score = rule_score['score']
                    confidence = rule_score['confidence']
                    method = 'rule_based_fallback'
            else:
                # Pure rule-based scoring
                final_score = rule_score['score']
                confidence = rule_score['confidence']
                method = 'rule_based'
            
            # Enhance with ML insights if available
            if self.feature_importance:
                rule_score['ml_insights'] = self._get_ml_insights(lead_data)
            
            # Override score with final calculated score
            rule_score['score'] = min(int(final_score), 100)
            rule_score['scoring_method'] = method
            rule_score['ml_confidence'] = ml_prediction.get('confidence', 0) if 'ml_prediction' in locals() else 0
            
            return rule_score
            
        except Exception as e:
            self.logger.error(f"Scoring error: {e}")
            # Fallback to rule-based only
            return self.rule_based_scorer.score_lead(lead_data)
    
    def _get_ml_prediction(self, lead_data):
        """Get ML model prediction"""
        try:
            features = self._extract_features(lead_data)
            features_scaled = self.scaler.transform([features])
            
            # Get probability scores
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            
            # Convert to 0-100 score
            conversion_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            ml_score = conversion_prob * 100
            
            # Calculate confidence based on prediction certainty
            confidence = max(probabilities) if len(probabilities) > 1 else 0.5
            
            return {
                'score': ml_score,
                'confidence': confidence,
                'conversion_probability': conversion_prob
            }
        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
            return {'score': 50, 'confidence': 0, 'conversion_probability': 0.5}
    
    def _extract_features(self, lead_data):
        """Extract numerical features for ML model"""
        features = []
        
        # Company size (normalized)
        employees = lead_data.get('employees', 0)
        features.append(min(employees / 100, 2))  # Normalize to 0-2 range
        
        # Industry encoding
        industry = lead_data.get('industry', 'unknown').lower()
        industry_score = self._encode_industry(industry)
        features.append(industry_score)
        
        # Title authority
        title = lead_data.get('title', '').lower()
        authority_score = self._encode_title_authority(title)
        features.append(authority_score)
        
        # Technology sophistication
        technologies = lead_data.get('technologies', [])
        tech_score = self._encode_technology_stack(technologies)
        features.append(tech_score)
        
        # Reddit intelligence features
        reddit_data = lead_data.get('reddit_intelligence', {})
        features.extend([
            1 if reddit_data.get('urgency_level') == 'high' else 0,
            len(reddit_data.get('pain_points', [])) / 5,  # Normalize pain points
            1 if 'time_consuming' in reddit_data.get('pain_points', []) else 0
        ])
        
        # AI enrichment features
        ai_data = lead_data.get('ai_enrichment', {})
        features.extend([
            min(ai_data.get('time_estimate_hours', 0) / 20, 1),  # Normalize hours
            ai_data.get('conversion_likelihood', 5) / 10  # Normalize likelihood
        ])
        
        # Engagement features (if available)
        features.extend([
            lead_data.get('email_opens', 0) / 10,  # Normalize opens
            lead_data.get('email_clicks', 0) / 5,  # Normalize clicks
            1 if lead_data.get('has_website_visit') else 0
        ])
        
        return features
    
    def _encode_industry(self, industry):
        """Encode industry to numerical score"""
        high_value = ['e-commerce', 'ecommerce', 'online store', 'shopify', 'e-learning', 'education', 'course']
        medium_value = ['marketing', 'agency', 'consulting', 'services']
        
        if any(keyword in industry for keyword in high_value):
            return 1.0
        elif any(keyword in industry for keyword in medium_value):
            return 0.6
        else:
            return 0.3
    
    def _encode_title_authority(self, title):
        """Encode title to authority score"""
        if any(keyword in title for keyword in ['founder', 'ceo', 'owner', 'president']):
            return 1.0
        elif any(keyword in title for keyword in ['director', 'vp', 'head of']):
            return 0.8
        elif any(keyword in title for keyword in ['manager', 'lead']):
            return 0.6
        else:
            return 0.3
    
    def _encode_technology_stack(self, technologies):
        """Encode technology sophistication"""
        premium_tech = ['shopify', 'salesforce', 'hubspot', 'marketo']
        basic_tech = ['wordpress', 'mailchimp', 'google analytics']
        
        premium_count = sum(1 for tech in technologies if any(p in str(tech).lower() for p in premium_tech))
        basic_count = sum(1 for tech in technologies if any(b in str(tech).lower() for b in basic_tech))
        
        return min((premium_count * 0.3) + (basic_count * 0.1), 1.0)
    
    def learn_from_conversion(self, lead_data, conversion_outcome):
        """
        Learn from actual conversion data to improve ML model
        Call this whenever you get conversion data (demo booked, trial, paid)
        """
        try:
            # Store conversion data
            learning_record = {
                'features': self._extract_features(lead_data),
                'lead_data': lead_data.copy(),
                'converted': 1 if conversion_outcome in ['demo_booked', 'trial_signup', 'paid_conversion'] else 0,
                'conversion_type': conversion_outcome,
                'timestamp': datetime.now().isoformat(),
                'days_to_conversion': lead_data.get('days_to_conversion', 0)
            }
            
            self.conversion_data.append(learning_record)
            self.logger.info(f"Learned from conversion: {conversion_outcome}")
            
            # Retrain model if we have enough data
            if len(self.conversion_data) >= 50:  # Minimum viable dataset
                self._retrain_model()
                
        except Exception as e:
            self.logger.error(f"Learning error: {e}")
    
    def _retrain_model(self):
        """Retrain ML model with accumulated conversion data"""
        try:
            if len(self.conversion_data) < 20:
                return False
            
            # Prepare training data
            X = np.array([record['features'] for record in self.conversion_data])
            y = np.array([record['converted'] for record in self.conversion_data])
            
            # Only retrain if we have both positive and negative examples
            if len(np.unique(y)) < 2:
                self.logger.warning("Need both positive and negative conversion examples")
                return False
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest (robust and interpretable)
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            )
            
            self.ml_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            if len(X_test) > 0:
                y_pred = self.ml_model.predict(X_test_scaled)
                accuracy = (y_pred == y_test).mean()
                
                if accuracy > 0.6:  # Only use if reasonably accurate
                    self.is_ml_trained = True
                    self.feature_importance = dict(zip(
                        range(len(self.ml_model.feature_importances_)),
                        self.ml_model.feature_importances_
                    ))
                    self.logger.info(f"ML model retrained. Accuracy: {accuracy:.3f}")
                    
                    # Save model
                    self._save_model()
                    return True
                else:
                    self.logger.warning(f"ML model accuracy too low: {accuracy:.3f}")
                    return False
            
        except Exception as e:
            self.logger.error(f"Model retraining error: {e}")
            return False
    
    def _get_ml_insights(self, lead_data):
        """Get ML-driven insights about the lead"""
        if not self.feature_importance:
            return []
        
        insights = []
        features = self._extract_features(lead_data)
        
        # Identify top contributing factors
        for i, importance in self.feature_importance.items():
            if importance > 0.1 and i < len(features):  # High importance features
                feature_value = features[i]
                if feature_value > 0.7:  # Strong positive signal
                    insights.append(f"ML identifies strong signal in feature {i}")
        
        return insights
    
    def _save_model(self):
        """Save trained model and scalers"""
        try:
            joblib.dump(self.ml_model, 'lead_scoring_model.pkl')
            joblib.dump(self.scaler, 'lead_scoring_scaler.pkl')
            
            with open('feature_importance.json', 'w') as f:
                json.dump(self.feature_importance, f)
                
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
    
    def load_model(self):
        """Load previously trained model"""
        try:
            self.ml_model = joblib.load('lead_scoring_model.pkl')
            self.scaler = joblib.load('lead_scoring_scaler.pkl')
            
            with open('feature_importance.json', 'r') as f:
                self.feature_importance = json.load(f)
            
            self.is_ml_trained = True
            self.logger.info("ML model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not load ML model: {e}")
            return False


class RuleBasedScorer:
    """
    Proven rule-based scoring system - works immediately
    Based on actual conversion patterns for $197/month SaaS
    """
    
    def score_lead(self, lead_data):
        score = 0
        confidence = "medium"
        reasons = []
        
        # Company Size Scoring (35% of total score)
        employees = lead_data.get('employees', 0)
        if 1 <= employees <= 10:
            score += 35
            reasons.append("Perfect size: small business (1-10 employees)")
            confidence = "high"
        elif 11 <= employees <= 25:
            score += 25
            reasons.append("Good size: growing business (11-25 employees)")
        elif 26 <= employees <= 50:
            score += 15
            reasons.append("Decent size: established business (26-50 employees)")
        else:
            score += 5
            reasons.append("Suboptimal size for our solution")
        
        # Industry Scoring (30% of total score)
        industry = lead_data.get('industry', '').lower()
        if any(keyword in industry for keyword in ['e-commerce', 'ecommerce', 'online store', 'shopify']):
            score += 30
            reasons.append("HIGH VALUE: E-commerce business")
        elif any(keyword in industry for keyword in ['e-learning', 'education', 'course', 'coaching']):
            score += 25
            reasons.append("HIGH VALUE: Education/Coaching business")
        elif any(keyword in industry for keyword in ['marketing', 'agency', 'advertising']):
            score += 20
            reasons.append("GOOD: Marketing-related business")
        elif any(keyword in industry for keyword in ['consulting', 'services']):
            score += 15
            reasons.append("DECENT: Service-based business")
        else:
            score += 5
            reasons.append("Unknown or low-fit industry")
        
        # Title Authority (20% of total score)
        title = lead_data.get('title', '').lower()
        if any(keyword in title for keyword in ['founder', 'ceo', 'owner', 'president']):
            score += 20
            reasons.append("DECISION MAKER: C-level or founder")
        elif any(keyword in title for keyword in ['director', 'vp', 'head of']):
            score += 15
            reasons.append("INFLUENCER: Senior management")
        elif any(keyword in title for keyword in ['manager', 'marketing']):
            score += 10
            reasons.append("RELEVANT ROLE: Management/marketing")
        else:
            score += 5
            reasons.append("Unknown decision-making power")
        
        # Technology Stack (15% of total score)
        technologies = lead_data.get('technologies', [])
        tech_score = 0
        if any('shopify' in str(tech).lower() for tech in technologies):
            tech_score += 10
            reasons.append("BUDGET CONFIRMED: Uses Shopify")
        if any('salesforce' in str(tech).lower() for tech in technologies):
            tech_score += 8
            reasons.append("ENTERPRISE BUDGET: Uses Salesforce")
        if any('mailchimp' in str(tech).lower() for tech in technologies):
            tech_score += 3
            reasons.append("Marketing-aware: Uses Mailchimp")
        
        score += min(tech_score, 15)
        
        # Apply quality and priority tiers
        if score >= 80:
            quality = "A+ (HOT LEAD)"
            priority = "IMMEDIATE"
            recommended_action = "CALL IMMEDIATELY - High-value prospect"
            email_timing = "Within 1 hour"
        elif score >= 65:
            quality = "A (HIGH QUALITY)"
            priority = "HIGH" 
            recommended_action = "Email within 2 hours - Personalized approach"
            email_timing = "Within 2-4 hours"
        elif score >= 45:
            quality = "B (MEDIUM QUALITY)"
            priority = "MEDIUM"
            recommended_action = "Email within 24 hours - Standard sequence"
            email_timing = "Within 24 hours"
        else:
            quality = "C (LOW QUALITY)"
            priority = "LOW"
            recommended_action = "Email in 3-5 days - Low priority sequence"
            email_timing = "Within 72 hours"
        
        return {
            'score': min(score, 100),
            'quality': quality,
            'priority': priority,
            'confidence': confidence,
            'reasons': reasons,
            'recommended_action': recommended_action,
            'email_timing': email_timing,
            'scored_at': datetime.now().isoformat()
        }


class LeadScoringAPI:
    """
    Flask API endpoints for your Heroku deployment
    """
    
    def __init__(self):
        self.scorer = HybridLeadScorer()
        self.scorer.load_model()  # Try to load existing model
    
    def score_single_lead(self, lead_data):
        """Score a single lead"""
        return self.scorer.score_lead(lead_data)
    
    def score_batch(self, leads_batch):
        """Score a batch of leads (for n8n workflow)"""
        results = []
        high_priority_count = 0
        
        leads = leads_batch.get('leads', [])
        
        for lead in leads:
            try:
                score_result = self.scorer.score_lead(lead)
                
                result = {
                    'lead_id': lead.get('lead_id'),
                    'email': lead.get('email'),
                    'first_name': lead.get('first_name'),
                    'last_name': lead.get('last_name'),
                    'company': lead.get('company'),
                    'score': score_result['score'],
                    'quality': score_result['quality'],
                    'priority': score_result['priority'],
                    'action': score_result['recommended_action'],
                    'reasons': score_result['reasons'],
                    'scored_at': score_result['scored_at']
                }
                
                results.append(result)
                
                if score_result['priority'] in ['IMMEDIATE', 'HIGH']:
                    high_priority_count += 1
                    
            except Exception as e:
                logging.error(f"Error scoring lead {lead.get('email', 'unknown')}: {e}")
                continue
        
        return {
            'leads': results,
            'total_processed': len(results),
            'high_priority_leads': high_priority_count,
            'conversion_estimate': high_priority_count * 0.15,
            'revenue_estimate': high_priority_count * 0.15 * 197,
            'processed_at': datetime.now().isoformat()
        }
    
    def record_conversion(self, conversion_data):
        """Record conversion for ML learning"""
        lead_data = conversion_data.get('lead_data', {})
        outcome = conversion_data.get('outcome')
        
        self.scorer.learn_from_conversion(lead_data, outcome)
        
        return {
            'success': True,
            'message': 'Conversion recorded for ML learning',
            'ml_trained': self.scorer.is_ml_trained
        }


# Flask application setup
app = Flask(__name__)

# Initialize the scoring system
scoring_api = LeadScoringAPI()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_trained': scoring_api.scorer.is_ml_trained
    })

@app.route('/score', methods=['POST'])
def score_leads():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Handle both single lead and batch requests
        if 'leads' in data:
            # Batch scoring
            result = scoring_api.score_batch(data)
        else:
            # Single lead scoring
            result = scoring_api.score_single_lead(data)
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Scoring error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/learn', methods=['POST'])
def record_conversion():
    try:
        data = request.get_json()
        result = scoring_api.record_conversion(data)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Learning error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)