"""
Advanced Training Script for Go Rank Prediction
Implements feature engineering and ensemble methods for higher accuracy

Run: python training.py --train_dir train --test_dir test
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse


class AdvancedGoRankPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def parse_game_file(self, filepath):
        """Parse game file with error handling."""
        features_list = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('B[') or line.startswith('W['):
                move_color = line[0]
                
                try:
                    policy_line = lines[i + 1].strip().split()
                    value_line = lines[i + 2].strip().split()
                    rank_line = lines[i + 3].strip().split()
                    strength_line = lines[i + 4].strip()
                    katago_line = lines[i + 5].strip().split()
                    
                    policy_probs = [float(x) for x in policy_line]
                    value_preds = [float(x.rstrip('%')) / 100.0 for x in value_line]
                    rank_probs = [float(x) for x in rank_line]
                    strength_score = float(strength_line)
                    
                    winrate = float(katago_line[0].rstrip('%')) / 100.0
                    lead = float(katago_line[1])
                    uncertainty = float(katago_line[2])
                    
                    if move_color == 'W':
                        winrate = 1.0 - winrate
                        lead = -lead
                    
                    move_features = policy_probs + value_preds + rank_probs + \
                                   [strength_score, winrate, lead, uncertainty]
                    
                    features_list.append(move_features)
                    i += 6
                except (IndexError, ValueError):
                    i += 1
                    continue
            else:
                i += 1
        
        return np.array(features_list)
    
    def engineer_features(self, move_features):
        """
        Advanced feature engineering with domain knowledge.
        Creates comprehensive statistical features and rank-specific patterns.
        """
        if len(move_features) == 0:
            return np.zeros(self._get_feature_dim())
        
        aggregated = []
        
        # Split into feature groups
        policy_feats = move_features[:, 0:9]      # Policy from 9 models
        value_feats = move_features[:, 9:18]      # Value from 9 models
        rank_feats = move_features[:, 18:27]      # Rank model outputs
        strength_feats = move_features[:, 27:28]  # Strength score
        katago_feats = move_features[:, 28:31]    # KataGo: winrate, lead, uncertainty
        
        # 1. Basic statistics for each feature
        for feat_idx in range(move_features.shape[1]):
            feat_values = move_features[:, feat_idx]
            aggregated.extend([
                np.mean(feat_values),
                np.std(feat_values),
                np.median(feat_values),
                np.percentile(feat_values, 25),
                np.percentile(feat_values, 75),
                np.max(feat_values) - np.min(feat_values),
                np.max(feat_values),
                np.min(feat_values)
            ])
        
        # 2. Rank-specific features: Which rank models give highest probability?
        policy_argmax = np.argmax(policy_feats, axis=1)
        rank_argmax = np.argmax(rank_feats, axis=1)
        
        # Distribution of which rank is predicted most often
        for rank in range(9):
            aggregated.append(np.mean(policy_argmax == rank))
            aggregated.append(np.mean(rank_argmax == rank))
        
        # 3. Consistency features
        aggregated.append(np.std(policy_argmax))  # How consistent are policy predictions
        aggregated.append(np.std(rank_argmax))    # How consistent are rank predictions
        
        # 4. Rank model confidence
        rank_max_probs = np.max(rank_feats, axis=1)
        aggregated.extend([
            np.mean(rank_max_probs),
            np.std(rank_max_probs),
            np.median(rank_max_probs)
        ])
        
        # 5. Policy-value agreement
        policy_entropy = -np.sum(policy_feats * np.log(policy_feats + 1e-10), axis=1)
        aggregated.extend([
            np.mean(policy_entropy),
            np.std(policy_entropy)
        ])
        
        # 6. KataGo-specific features
        # Strength variation (weaker players have more variable play)
        aggregated.append(np.std(katago_feats[:, 2]))  # Uncertainty variation
        
        # Lead stability (stronger players maintain leads better)
        lead_changes = np.abs(np.diff(katago_feats[:, 1]))
        aggregated.append(np.mean(lead_changes) if len(lead_changes) > 0 else 0)
        
        # 7. Game phase features (early vs late game behavior)
        n_moves = len(move_features)
        early_moves = move_features[:n_moves//3]
        late_moves = move_features[2*n_moves//3:]
        
        if len(early_moves) > 0:
            aggregated.append(np.mean(early_moves[:, 27]))  # Early strength
        else:
            aggregated.append(0)
            
        if len(late_moves) > 0:
            aggregated.append(np.mean(late_moves[:, 27]))   # Late strength
        else:
            aggregated.append(0)
        
        # 8. Temporal features
        aggregated.append(n_moves)  # Total number of moves
        
        return np.array(aggregated)
    
    def _get_feature_dim(self):
        """Calculate expected feature dimension."""
        # 30 features * 8 stats + 18 rank dist + 2 consistency + 3 confidence +
        # 2 entropy + 2 katago + 2 phase + 1 moves = 30*8 + 28 = 268
        return 268
    
    def extract_features_from_file(self, filepath):
        """Extract and engineer features from a file."""
        move_features = self.parse_game_file(filepath)
        return self.engineer_features(move_features)
    
    def load_training_data(self, train_dir='train'):
        """Load training data."""
        X_train = []
        y_train = []
        
        for rank in range(1, 10):
            filename = f'log_{rank}D_policy_train.txt'
            filepath = os.path.join(train_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Loading {filename}...")
                features = self.extract_features_from_file(filepath)
                X_train.append(features)
                y_train.append(rank)
        
        return np.array(X_train), np.array(y_train)
    
    def train(self, train_dir='train'):
        """Train with ensemble of models."""
        print("Loading training data...")
        X_train, y_train = self.load_training_data(train_dir)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Rank distribution: {np.bincount(y_train)}")
        
        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build ensemble
        print("\nBuilding ensemble model...")
        
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        lr = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        )
        
        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft',
            n_jobs=-1
        )
        
        # Since we only have 9 training samples (one per rank), 
        # we cannot do cross-validation. Train directly on all data.
        print("\nTraining ensemble on all data...")
        print("Note: With only 9 samples, cross-validation is not possible.")
        
        # Train individual models for diagnostic purposes
        print("\nTraining individual models...")
        rf.fit(X_train_scaled, y_train)
        print(f"Random Forest training accuracy: {rf.score(X_train_scaled, y_train):.4f}")
        
        gb.fit(X_train_scaled, y_train)
        print(f"Gradient Boosting training accuracy: {gb.score(X_train_scaled, y_train):.4f}")
        
        lr.fit(X_train_scaled, y_train)
        print(f"Logistic Regression training accuracy: {lr.score(X_train_scaled, y_train):.4f}")
        
        # Train ensemble
        ensemble.fit(X_train_scaled, y_train)
        train_acc = ensemble.score(X_train_scaled, y_train)
        print(f"\nEnsemble training accuracy: {train_acc:.4f}")
        
        self.model = ensemble
        self.save_model()
        print("\nTraining complete!")
        
        return train_acc
    
    def predict(self, test_dir='test'):
        """Generate predictions."""
        if self.model is None:
            self.load_model()
        
        predictions = []
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.txt')],
                           key=lambda x: int(x.split('.')[0]))
        
        print(f"Processing {len(test_files)} test files...")
        
        for filename in test_files:
            filepath = os.path.join(test_dir, filename)
            file_id = filename.split('.')[0]
            
            features = self.extract_features_from_file(filepath)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            pred_rank = self.model.predict(features_scaled)[0]
            predictions.append({'id': int(file_id), 'rank': pred_rank})
        
        return predictions
    
    def save_model(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """Save model."""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """Load model."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Model loaded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='train')
    parser.add_argument('--test_dir', type=str, default='test')
    args = parser.parse_args()
    
    predictor = AdvancedGoRankPredictor()
    
    print("=" * 60)
    print("ADVANCED TRAINING")
    print("=" * 60)
    predictor.train(args.train_dir)


if __name__ == '__main__':
    main()