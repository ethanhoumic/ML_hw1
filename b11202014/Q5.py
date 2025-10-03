"""
Go Rank Prediction ML Model - Improved Version
Assignment 1 Q5 - Machine Learning Class Fall 2025

Key improvements:
- Better feature engineering focusing on rank model outputs
- Weighted aggregation giving more importance to rank predictions
- Additional domain-specific features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import os
import argparse


class ImprovedGoRankPredictor:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()  # Match training.py
        
    def parse_game_file(self, filepath):
        """Parse a single game file and extract features from all moves."""
        features_list = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this is a move line (starts with B[ or W[)
            if line.startswith('B[') or line.startswith('W['):
                move_color = line[0]  # 'B' or 'W'
                
                try:
                    # Parse the 6 parts of each move
                    policy_line = lines[i + 1].strip().split()
                    value_line = lines[i + 2].strip().split()
                    rank_line = lines[i + 3].strip().split()
                    strength_line = lines[i + 4].strip()
                    katago_line = lines[i + 5].strip().split()
                    
                    # Extract numerical features
                    policy_probs = [float(x) for x in policy_line]  # 9 values
                    value_preds = [float(x.rstrip('%')) / 100.0 for x in value_line]  # 9 values
                    rank_probs = [float(x) for x in rank_line]  # 9 values
                    strength_score = float(strength_line)  # 1 value
                    
                    # KataGo features (3 values)
                    winrate = float(katago_line[0].rstrip('%')) / 100.0
                    lead = float(katago_line[1])
                    uncertainty = float(katago_line[2])
                    
                    # Invert winrate and lead for white moves (black's perspective)
                    if move_color == 'W':
                        winrate = 1.0 - winrate
                        lead = -lead
                    
                    # Combine all features for this move (total: 30 features)
                    move_features = policy_probs + value_preds + rank_probs + \
                                   [strength_score, winrate, lead, uncertainty]
                    
                    features_list.append(move_features)
                    
                    i += 6  # Move to next move block
                except (IndexError, ValueError) as e:
                    i += 1
                    continue
            else:
                i += 1
        
        return np.array(features_list) if features_list else np.zeros((0, 30))
    
    def engineer_advanced_features(self, move_features):
        """
        Advanced feature engineering with focus on rank prediction.
        The rank model outputs (indices 18-26) are the most important!
        MUST match training.py feature engineering (250 features)
        """
        if len(move_features) == 0:
            return np.zeros(250)  # Return fixed-size zero vector
        
        aggregated = []
        
        # Split features by type
        policy_probs = move_features[:, 0:9]      # Policy probabilities
        value_preds = move_features[:, 9:18]      # Value predictions
        rank_probs = move_features[:, 18:27]      # MOST IMPORTANT: Rank model outputs
        strength = move_features[:, 27]           # Strength score
        winrate = move_features[:, 28]            # KataGo winrate
        lead = move_features[:, 29]               # KataGo lead
        
        rank_indices = np.arange(1, 10)
        
        # ===== RANK MODEL FEATURES (HIGHEST PRIORITY) =====
        
        # 1. Direct rank probability features (9)
        rank_means = np.mean(rank_probs, axis=0)
        aggregated.extend(rank_means)
        
        # 2. Median rank probabilities (9)
        rank_medians = np.median(rank_probs, axis=0)
        aggregated.extend(rank_medians)
        
        # 3. 75th percentile (9)
        rank_p75 = np.percentile(rank_probs, 75, axis=0)
        aggregated.extend(rank_p75)
        
        # 4. 25th percentile (9)
        rank_p25 = np.percentile(rank_probs, 25, axis=0)
        aggregated.extend(rank_p25)
        
        # 5. Max probabilities (9)
        rank_maxs = np.max(rank_probs, axis=0)
        aggregated.extend(rank_maxs)
        
        # 6. Std of probabilities (9)
        rank_stds = np.std(rank_probs, axis=0)
        aggregated.extend(rank_stds)
        
        # 7. Which rank is predicted most often (9)
        rank_argmax = np.argmax(rank_probs, axis=1)
        rank_mode_dist = np.bincount(rank_argmax, minlength=9) / len(rank_argmax)
        aggregated.extend(rank_mode_dist)
        
        # 8. Weighted rank prediction statistics (10)
        weighted_ranks = np.sum(rank_probs * rank_indices, axis=1)
        aggregated.extend([
            np.mean(weighted_ranks),
            np.median(weighted_ranks),
            np.std(weighted_ranks),
            np.percentile(weighted_ranks, 10),
            np.percentile(weighted_ranks, 25),
            np.percentile(weighted_ranks, 50),
            np.percentile(weighted_ranks, 75),
            np.percentile(weighted_ranks, 90),
            np.min(weighted_ranks),
            np.max(weighted_ranks)
        ])
        
        # 9. Rank prediction confidence (5)
        rank_max_probs = np.max(rank_probs, axis=1)
        aggregated.extend([
            np.mean(rank_max_probs),
            np.median(rank_max_probs),
            np.std(rank_max_probs),
            np.min(rank_max_probs),
            np.max(rank_max_probs)
        ])
        
        # 10. Rank entropy (lower entropy = more certain) (1)
        rank_entropy = -np.sum(rank_probs * np.log(rank_probs + 1e-10), axis=1)
        aggregated.append(np.mean(rank_entropy))
        
        # ===== POLICY FEATURES =====
        
        # 11. Policy-based rank inference (9)
        policy_means = np.mean(policy_probs, axis=0)
        aggregated.extend(policy_means)
        
        # 12. Which policy model matches best (9)
        policy_argmax = np.argmax(policy_probs, axis=1)
        policy_mode_dist = np.bincount(policy_argmax, minlength=9) / len(policy_argmax)
        aggregated.extend(policy_mode_dist)
        
        # 13. Policy confidence (3)
        policy_max_probs = np.max(policy_probs, axis=1)
        aggregated.extend([
            np.mean(policy_max_probs),
            np.std(policy_max_probs),
            np.median(policy_max_probs)
        ])
        
        # ===== VALUE FEATURES =====
        
        # 14. Value predictions by rank (9)
        value_means = np.mean(value_preds, axis=0)
        aggregated.extend(value_means)
        
        # ===== STRENGTH FEATURES =====
        
        # 15. Strength statistics (8)
        aggregated.extend([
            np.mean(strength),
            np.median(strength),
            np.std(strength),
            np.percentile(strength, 25),
            np.percentile(strength, 75),
            np.min(strength),
            np.max(strength),
            np.max(strength) - np.min(strength)
        ])
        
        # ===== KATAGO FEATURES =====
        
        # 16. Winrate features (5)
        aggregated.extend([
            np.mean(winrate),
            np.median(winrate),
            np.std(winrate),
            np.mean(np.abs(winrate - 0.5)),
            np.percentile(winrate, 75) - np.percentile(winrate, 25)
        ])
        
        # 17. Lead features (5)
        aggregated.extend([
            np.mean(lead),
            np.median(lead),
            np.std(lead),
            np.mean(np.abs(lead)),
            np.percentile(np.abs(lead), 75)
        ])
        
        # ===== CONSISTENCY FEATURES =====
        
        # 18. How consistent are predictions across moves (4)
        aggregated.extend([
            np.std(rank_argmax),
            np.std(policy_argmax),
            len(np.unique(rank_argmax)) / 9.0,  # Diversity of predictions
            len(np.unique(policy_argmax)) / 9.0
        ])
        
        # ===== TEMPORAL FEATURES =====
        
        # 19. Early vs middle vs late game (27)
        n_moves = len(move_features)
        third = max(1, n_moves // 3)
        
        early_moves = move_features[:third]
        middle_moves = move_features[third:2*third]
        late_moves = move_features[2*third:]
        
        for phase_moves in [early_moves, middle_moves, late_moves]:
            if len(phase_moves) > 0:
                phase_rank_probs = phase_moves[:, 18:27]
                phase_rank_mean = np.mean(phase_rank_probs, axis=0)
                aggregated.extend(phase_rank_mean)
            else:
                aggregated.extend([0.0] * 9)
        
        # 20. Number of moves (1)
        aggregated.append(n_moves)
        
        # 21. Game length category (3) - one-hot encoding
        if n_moves < 100:
            aggregated.extend([1, 0, 0])
        elif n_moves < 200:
            aggregated.extend([0, 1, 0])
        else:
            aggregated.extend([0, 0, 1])
        
        # ===== AGREEMENT FEATURES =====
        
        # 22. Policy-Rank agreement (3)
        policy_weighted = np.sum(policy_probs * rank_indices, axis=1)
        if len(policy_weighted) > 1:
            corr = np.corrcoef(weighted_ranks, policy_weighted)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        aggregated.extend([
            corr,
            np.mean(np.abs(weighted_ranks - policy_weighted)),
            np.std(weighted_ranks - policy_weighted)
        ])
        
        # Pad or truncate to fixed size (250 features)
        result = np.array(aggregated)
        if len(result) < 250:
            result = np.pad(result, (0, 250 - len(result)), constant_values=0)
        else:
            result = result[:250]
        
        return result
    
    def extract_features_from_file(self, filepath):
        """Extract and engineer features from a single file."""
        move_features = self.parse_game_file(filepath)
        return self.engineer_advanced_features(move_features)
    
    def load_training_data(self, train_dir='train'):
        """Load all training data from rank-labeled files."""
        X_train = []
        y_train = []
        
        # Process each rank file (1D to 9D)
        for rank in range(1, 10):
            filename = f'log_{rank}D_policy_train.txt'
            filepath = os.path.join(train_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Loading {filename}...")
                features = self.extract_features_from_file(filepath)
                X_train.append(features)
                y_train.append(rank)
                print(f"  Features shape: {features.shape}, Rank: {rank}")
            else:
                print(f"Warning: {filename} not found")
        
        return np.array(X_train), np.array(y_train)
    
    def train(self, train_dir='train'):
        """Train the rank prediction model."""
        print("Loading training data...")
        X_train, y_train = self.load_training_data(train_dir)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Ranks: {y_train}")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("\nTraining models...")
        
        # Model 1: Random Forest with optimal params
        rf_model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Model 2: Extra Trees (more randomization)
        et_model = ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=43,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Model 3: Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            random_state=42
        )
        
        # Model 4: Logistic Regression with L2 regularization
        lr_model = LogisticRegression(
            C=100.0,
            max_iter=5000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced'
        )
        
        # Train each model
        print("\nTraining Random Forest...")
        rf_model.fit(X_train_scaled, y_train)
        rf_acc = rf_model.score(X_train_scaled, y_train)
        print(f"  Training accuracy: {rf_acc:.4f}")
        
        print("Training Extra Trees...")
        et_model.fit(X_train_scaled, y_train)
        et_acc = et_model.score(X_train_scaled, y_train)
        print(f"  Training accuracy: {et_acc:.4f}")
        
        print("Training Gradient Boosting...")
        gb_model.fit(X_train_scaled, y_train)
        gb_acc = gb_model.score(X_train_scaled, y_train)
        print(f"  Training accuracy: {gb_acc:.4f}")
        
        print("Training Logistic Regression...")
        lr_model.fit(X_train_scaled, y_train)
        lr_acc = lr_model.score(X_train_scaled, y_train)
        print(f"  Training accuracy: {lr_acc:.4f}")
        
        # Create ensemble (matching training.py)
        print("\nCreating super ensemble...")
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('et', et_model),
                ('gb', gb_model),
                ('lr', lr_model)
            ],
            voting='soft',
            weights=[3, 3, 3, 1],  # Give more weight to tree models
            n_jobs=-1
        )
        
        ensemble.fit(X_train_scaled, y_train)
        ensemble_acc = ensemble.score(X_train_scaled, y_train)
        print(f"Ensemble training accuracy: {ensemble_acc:.4f}")
        
        self.model = ensemble
        
        # Save the model and scaler
        self.save_model()
        
        print("\nTraining complete!")
        return ensemble_acc
    
    def predict(self, test_dir='test'):
        """Generate predictions for all test files."""
        if self.model is None:
            self.load_model()
        
        predictions = []
        
        # Get all test files
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
        test_files = sorted(test_files, key=lambda x: int(x.split('.')[0]))
        
        print(f"Processing {len(test_files)} test files...")
        
        for idx, filename in enumerate(test_files):
            filepath = os.path.join(test_dir, filename)
            file_id = filename.split('.')[0]
            
            # Extract features
            features = self.extract_features_from_file(filepath)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict rank
            pred_rank = self.model.predict(features_scaled)[0]
            
            # Get prediction probabilities for analysis
            pred_proba = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(pred_proba)
            
            predictions.append({'id': int(file_id), 'rank': pred_rank})
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(test_files)} files")
        
        return predictions
    
    def save_model(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """Save trained model and scaler."""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {model_path} and {scaler_path}")
    
    def load_model(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """Load trained model and scaler."""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Model files not found. Please run training first:\n"
                f"  python Q5.py --train --train_dir train"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Model loaded successfully")


def main():
    parser = argparse.ArgumentParser(description='Go Rank Prediction')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--train_dir', type=str, default='train', help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='test', help='Test data directory')
    args = parser.parse_args()
    
    predictor = ImprovedGoRankPredictor()
    
    if args.train:
        # Training mode
        print("=" * 60)
        print("IMPROVED TRAINING MODE")
        print("=" * 60)
        predictor.train(args.train_dir)
    else:
        # Prediction mode
        print("=" * 60)
        print("PREDICTION MODE")
        print("=" * 60)
        predictions = predictor.predict(args.test_dir)
        
        # Create submission DataFrame
        df = pd.DataFrame(predictions)
        df = df.sort_values('id')
        df.to_csv('submission.csv', index=False)
        
        print(f"\n✓ Predictions saved to submission.csv")
        print(f"✓ Total predictions: {len(predictions)}")
        print("\nFirst few predictions:")
        print(df.head(10))
        print("\nLast few predictions:")
        print(df.tail(10))
        print("\nRank distribution:")
        print(df['rank'].value_counts().sort_index())
        print("\n✓ Submission file ready for Kaggle upload!")


if __name__ == '__main__':
    main()