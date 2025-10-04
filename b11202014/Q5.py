"""
Go Rank Prediction ML Model - LightGBM Version (Anti-Overfitting)
Assignment 1 Q5 - Machine Learning Class Fall 2025

Key Changes to Prevent Overfitting:
- Reduced model complexity
- Stronger regularization
- Simplified feature engineering
- Conservative hyperparameters
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
import pickle
import os
import argparse


class LightGBMGoRankPredictor:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        
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
        
        return np.array(features_list) if features_list else np.zeros((0, 30))
    
    def engineer_advanced_features(self, move_features):
        """
        Simplified feature engineering to reduce overfitting.
        Focus on the most important features only.
        """
        if len(move_features) == 0:
            return np.zeros(150)  # Reduced from 300 to 150
        
        aggregated = []
        
        # Extract feature groups
        policy_probs = move_features[:, 0:9]
        value_preds = move_features[:, 9:18]
        rank_probs = move_features[:, 18:27]  # MOST IMPORTANT
        strength = move_features[:, 27]
        winrate = move_features[:, 28]
        lead = move_features[:, 29]
        
        rank_indices = np.arange(1, 10)
        
        # ===== RANK MODEL FEATURES (MOST CRITICAL) =====
        
        # 1. Mean rank probabilities (9) - Most stable
        aggregated.extend(np.mean(rank_probs, axis=0))
        
        # 2. Median rank probabilities (9) - Robust to outliers
        aggregated.extend(np.median(rank_probs, axis=0))
        
        # 3. Std rank probabilities (9) - Consistency measure
        aggregated.extend(np.std(rank_probs, axis=0))
        
        # 4. Weighted rank statistics (5) - Reduced from 15
        weighted_ranks = np.sum(rank_probs * rank_indices, axis=1)
        aggregated.extend([
            np.mean(weighted_ranks),
            np.median(weighted_ranks),
            np.std(weighted_ranks),
            np.percentile(weighted_ranks, 25),
            np.percentile(weighted_ranks, 75)
        ])
        
        # 5. Rank confidence (5) - Reduced from 10
        rank_max_probs = np.max(rank_probs, axis=1)
        aggregated.extend([
            np.mean(rank_max_probs),
            np.median(rank_max_probs),
            np.std(rank_max_probs),
            np.percentile(rank_max_probs, 25),
            np.percentile(rank_max_probs, 75)
        ])
        
        # 6. Rank entropy (3) - Reduced from 5
        rank_entropy = -np.sum(rank_probs * np.log(rank_probs + 1e-10), axis=1)
        aggregated.extend([
            np.mean(rank_entropy),
            np.median(rank_entropy),
            np.std(rank_entropy)
        ])
        
        # ===== POLICY FEATURES (Simplified) =====
        
        # 7. Policy summary (9)
        aggregated.extend(np.mean(policy_probs, axis=0))
        
        # 8. Policy confidence (3)
        policy_max_probs = np.max(policy_probs, axis=1)
        aggregated.extend([
            np.mean(policy_max_probs),
            np.median(policy_max_probs),
            np.std(policy_max_probs)
        ])
        
        # ===== VALUE FEATURES (Simplified) =====
        
        # 9. Value summary (9)
        aggregated.extend(np.mean(value_preds, axis=0))
        
        # ===== STRENGTH FEATURES =====
        
        # 10. Strength statistics (5)
        aggregated.extend([
            np.mean(strength),
            np.median(strength),
            np.std(strength),
            np.percentile(strength, 25),
            np.percentile(strength, 75)
        ])
        
        # ===== KATAGO FEATURES =====
        
        # 11. Winrate features (5)
        aggregated.extend([
            np.mean(winrate),
            np.median(winrate),
            np.std(winrate),
            np.percentile(winrate, 25),
            np.percentile(winrate, 75)
        ])
        
        # 12. Lead features (5)
        aggregated.extend([
            np.mean(lead),
            np.median(lead),
            np.std(lead),
            np.percentile(lead, 25),
            np.percentile(lead, 75)
        ])
        
        # ===== TEMPORAL FEATURES (Simplified) =====
        
        # 13. Early/middle/late game analysis (27)
        n_moves = len(move_features)
        third = max(1, n_moves // 3)
        
        phases = [
            move_features[:third],           # Early
            move_features[third:2*third],    # Middle
            move_features[2*third:]          # Late
        ]
        
        for phase_moves in phases:
            if len(phase_moves) > 0:
                phase_rank_probs = phase_moves[:, 18:27]
                # Mean rank probs per phase (9)
                aggregated.extend(np.mean(phase_rank_probs, axis=0))
            else:
                aggregated.extend([0.0] * 9)
        
        # ===== META FEATURES =====
        
        # 14. Game length (2)
        aggregated.append(np.log1p(n_moves))  # Log scale to reduce variance
        aggregated.append(min(n_moves / 300.0, 1.0))  # Normalized
        
        # 15. Consistency (3)
        rank_argmax = np.argmax(rank_probs, axis=1)
        aggregated.extend([
            np.std(rank_argmax),
            np.std(weighted_ranks),
            len(np.unique(rank_argmax)) / 9.0
        ])
        
        # Pad or truncate to 150
        result = np.array(aggregated)
        if len(result) < 150:
            result = np.pad(result, (0, 150 - len(result)), constant_values=0)
        else:
            result = result[:150]
        
        return result
    
    def extract_features_from_file(self, filepath):
        move_features = self.parse_game_file(filepath)
        return self.engineer_advanced_features(move_features)
    
    def load_training_data(self, train_dir='train'):
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
                print(f"  Rank {rank}: features shape = {features.shape}")
        
        return np.array(X_train), np.array(y_train)
    
    def train(self, train_dir='train'):
        print("=" * 70)
        print("LIGHTGBM TRAINING MODE (ANTI-OVERFITTING)")
        print("=" * 70)
        
        print("\nLoading training data...")
        X_train, y_train = self.load_training_data(train_dir)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Ranks: {y_train}")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("\n" + "=" * 70)
        print("Training Conservative Models (Reduced Complexity)...")
        print("=" * 70)
        
        # LightGBM Model 1: Very conservative
        lgb_model1 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='gbdt',
            n_estimators=200,  # Reduced from 5000
            max_depth=4,  # Reduced from 8
            learning_rate=0.05,  # Increased for faster convergence
            num_leaves=15,  # Reduced from 31
            min_child_samples=1,
            subsample=0.8,
            colsample_bytree=0.6,  # Reduced feature sampling
            reg_alpha=1.0,  # Increased regularization
            reg_lambda=1.0,  # Increased regularization
            min_split_gain=0.1,  # Prevent splitting on noise
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # LightGBM Model 2: Moderate
        lgb_model2 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='gbdt',
            n_estimators=300,  # Reduced from 5000
            max_depth=5,  # Reduced from 10
            learning_rate=0.03,
            num_leaves=20,  # Reduced from 63
            min_child_samples=1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.8,  # Increased regularization
            reg_lambda=0.8,
            min_split_gain=0.05,
            random_state=123,
            n_jobs=-1,
            verbose=-1
        )
        
        # Random Forest: Conservative
        rf_model = RandomForestClassifier(
            n_estimators=500,  # Reduced from 2000
            max_depth=8,  # Limited depth
            min_samples_split=3,  # Increased from 2
            min_samples_leaf=2,  # Increased from 1
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Train all models
        print("\n1. Training LightGBM Model 1 (Conservative)...")
        lgb_model1.fit(X_train_scaled, y_train)
        acc1 = lgb_model1.score(X_train_scaled, y_train)
        print(f"   Training accuracy: {acc1:.4f}")
        
        print("\n2. Training LightGBM Model 2 (Moderate)...")
        lgb_model2.fit(X_train_scaled, y_train)
        acc2 = lgb_model2.score(X_train_scaled, y_train)
        print(f"   Training accuracy: {acc2:.4f}")
        
        print("\n3. Training Random Forest (Conservative)...")
        rf_model.fit(X_train_scaled, y_train)
        acc3 = rf_model.score(X_train_scaled, y_train)
        print(f"   Training accuracy: {acc3:.4f}")
        
        # Create ensemble with fewer models
        print("\n" + "=" * 70)
        print("Creating Conservative Ensemble...")
        print("=" * 70)
        
        ensemble = VotingClassifier(
            estimators=[
                ('lgb1', lgb_model1),
                ('lgb2', lgb_model2),
                ('rf', rf_model)
            ],
            voting='soft',
            weights=[2, 2, 1],  # Equal-ish weights
            n_jobs=-1
        )
        
        ensemble.fit(X_train_scaled, y_train)
        ensemble_acc = ensemble.score(X_train_scaled, y_train)
        
        print(f"\nEnsemble Training Accuracy: {ensemble_acc:.4f}")
        
        if ensemble_acc > 0.95:
            print("\n⚠️  WARNING: Training accuracy very high (>95%)")
            print("    Model may still overfit. Consider further regularization.")
        
        self.model = ensemble
        self.save_model()
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Model: Conservative Ensemble (3 models)")
        print(f"Features: {X_train.shape[1]} (reduced from 300)")
        print(f"Training Accuracy: {ensemble_acc:.4f}")
        print(f"Individual accuracies: {acc1:.4f}, {acc2:.4f}, {acc3:.4f}")
        print("=" * 70)
        
        return ensemble_acc
    
    def predict(self, test_dir='test'):
        if self.model is None:
            self.load_model()
        
        predictions = []
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
        test_files = sorted(test_files, key=lambda x: int(x.split('.')[0]))
        
        print(f"\nProcessing {len(test_files)} test files...")
        
        for idx, filename in enumerate(test_files):
            filepath = os.path.join(test_dir, filename)
            file_id = filename.split('.')[0]
            
            features = self.extract_features_from_file(filepath)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            pred_rank = self.model.predict(features_scaled)[0]
            predictions.append({'id': int(file_id), 'rank': pred_rank})
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(test_files)} files")
        
        print(f"  Completed all {len(test_files)} files")
        
        return predictions
    
    def save_model(self, model_path='model_antioverfit.pkl', scaler_path='scaler_antioverfit.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\n✓ Model saved to {model_path} and {scaler_path}")
    
    def load_model(self, model_path='model_antioverfit.pkl', scaler_path='scaler_antioverfit.pkl'):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "Model files not found. Run training first:\n"
                "  python Q5.py --train --train_dir train"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✓ Model loaded successfully")


def main():
    parser = argparse.ArgumentParser(description='Go Rank Prediction with LightGBM')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--train_dir', type=str, default='train', help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='test', help='Test data directory')
    args = parser.parse_args()
    
    predictor = LightGBMGoRankPredictor()
    
    if args.train:
        predictor.train(args.train_dir)
    else:
        print("=" * 70)
        print("PREDICTION MODE - LightGBM")
        print("=" * 70)
        predictions = predictor.predict(args.test_dir)
        
        df = pd.DataFrame(predictions)
        df = df.sort_values('id')
        df.to_csv('submission.csv', index=False)
        
        print("\n" + "=" * 70)
        print("✓ PREDICTIONS SAVED")
        print("=" * 70)
        print(f"File: submission.csv")
        print(f"Total predictions: {len(predictions)}")
        print("\nFirst 10 predictions:")
        print(df.head(10))
        print("\nRank distribution:")
        print(df['rank'].value_counts().sort_index())
        print("\n" + "=" * 70)
        print("✓ Ready for Kaggle submission!")
        print("=" * 70)


if __name__ == '__main__':
    main()