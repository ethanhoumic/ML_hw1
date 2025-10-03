"""
Go Rank Prediction ML Model - LightGBM Version
Assignment 1 Q5 - Machine Learning Class Fall 2025

Uses LightGBM for superior performance on tabular data
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
        Advanced feature engineering optimized for LightGBM.
        Focus on rank model outputs and statistical aggregations.
        """
        if len(move_features) == 0:
            return np.zeros(300)
        
        aggregated = []
        
        # Extract feature groups
        policy_probs = move_features[:, 0:9]
        value_preds = move_features[:, 9:18]
        rank_probs = move_features[:, 18:27]  # MOST IMPORTANT
        strength = move_features[:, 27]
        winrate = move_features[:, 28]
        lead = move_features[:, 29]
        
        rank_indices = np.arange(1, 10)
        
        # ===== RANK MODEL FEATURES (CRITICAL) =====
        
        # 1. Mean rank probabilities (9)
        aggregated.extend(np.mean(rank_probs, axis=0))
        
        # 2. Median rank probabilities (9)
        aggregated.extend(np.median(rank_probs, axis=0))
        
        # 3. Std rank probabilities (9)
        aggregated.extend(np.std(rank_probs, axis=0))
        
        # 4. Max rank probabilities (9)
        aggregated.extend(np.max(rank_probs, axis=0))
        
        # 5. Min rank probabilities (9)
        aggregated.extend(np.min(rank_probs, axis=0))
        
        # 6. 25th percentile (9)
        aggregated.extend(np.percentile(rank_probs, 25, axis=0))
        
        # 7. 75th percentile (9)
        aggregated.extend(np.percentile(rank_probs, 75, axis=0))
        
        # 8. 90th percentile (9)
        aggregated.extend(np.percentile(rank_probs, 90, axis=0))
        
        # 9. 10th percentile (9)
        aggregated.extend(np.percentile(rank_probs, 10, axis=0))
        
        # 10. Mode distribution - which rank predicted most (9)
        rank_argmax = np.argmax(rank_probs, axis=1)
        rank_mode_dist = np.bincount(rank_argmax, minlength=9) / len(rank_argmax)
        aggregated.extend(rank_mode_dist)
        
        # 11. Weighted rank statistics (15)
        weighted_ranks = np.sum(rank_probs * rank_indices, axis=1)
        aggregated.extend([
            np.mean(weighted_ranks),
            np.median(weighted_ranks),
            np.std(weighted_ranks),
            np.min(weighted_ranks),
            np.max(weighted_ranks),
            np.percentile(weighted_ranks, 10),
            np.percentile(weighted_ranks, 25),
            np.percentile(weighted_ranks, 50),
            np.percentile(weighted_ranks, 75),
            np.percentile(weighted_ranks, 90),
            np.percentile(weighted_ranks, 95),
            np.max(weighted_ranks) - np.min(weighted_ranks),  # Range
            np.percentile(weighted_ranks, 75) - np.percentile(weighted_ranks, 25),  # IQR
            np.var(weighted_ranks),
            np.median(np.abs(weighted_ranks - np.median(weighted_ranks)))  # MAD
        ])
        
        # 12. Rank confidence metrics (10)
        rank_max_probs = np.max(rank_probs, axis=1)
        aggregated.extend([
            np.mean(rank_max_probs),
            np.median(rank_max_probs),
            np.std(rank_max_probs),
            np.min(rank_max_probs),
            np.max(rank_max_probs),
            np.percentile(rank_max_probs, 25),
            np.percentile(rank_max_probs, 75),
            np.percentile(rank_max_probs, 90),
            np.max(rank_max_probs) - np.min(rank_max_probs),
            np.mean(rank_max_probs > 0.5)  # Fraction with high confidence
        ])
        
        # 13. Rank entropy - prediction uncertainty (5)
        rank_entropy = -np.sum(rank_probs * np.log(rank_probs + 1e-10), axis=1)
        aggregated.extend([
            np.mean(rank_entropy),
            np.median(rank_entropy),
            np.std(rank_entropy),
            np.min(rank_entropy),
            np.max(rank_entropy)
        ])
        
        # ===== POLICY FEATURES =====
        
        # 14. Policy means (9)
        aggregated.extend(np.mean(policy_probs, axis=0))
        
        # 15. Policy medians (9)
        aggregated.extend(np.median(policy_probs, axis=0))
        
        # 16. Policy mode distribution (9)
        policy_argmax = np.argmax(policy_probs, axis=1)
        policy_mode_dist = np.bincount(policy_argmax, minlength=9) / len(policy_argmax)
        aggregated.extend(policy_mode_dist)
        
        # 17. Policy confidence (5)
        policy_max_probs = np.max(policy_probs, axis=1)
        aggregated.extend([
            np.mean(policy_max_probs),
            np.median(policy_max_probs),
            np.std(policy_max_probs),
            np.percentile(policy_max_probs, 25),
            np.percentile(policy_max_probs, 75)
        ])
        
        # 18. Policy-weighted rank (5)
        policy_weighted_rank = np.sum(policy_probs * rank_indices, axis=1)
        aggregated.extend([
            np.mean(policy_weighted_rank),
            np.median(policy_weighted_rank),
            np.std(policy_weighted_rank),
            np.percentile(policy_weighted_rank, 25),
            np.percentile(policy_weighted_rank, 75)
        ])
        
        # ===== VALUE FEATURES =====
        
        # 19. Value means (9)
        aggregated.extend(np.mean(value_preds, axis=0))
        
        # 20. Value medians (9)
        aggregated.extend(np.median(value_preds, axis=0))
        
        # ===== STRENGTH FEATURES =====
        
        # 21. Strength statistics (10)
        aggregated.extend([
            np.mean(strength),
            np.median(strength),
            np.std(strength),
            np.min(strength),
            np.max(strength),
            np.percentile(strength, 10),
            np.percentile(strength, 25),
            np.percentile(strength, 75),
            np.percentile(strength, 90),
            np.max(strength) - np.min(strength)
        ])
        
        # ===== KATAGO FEATURES =====
        
        # 22. Winrate features (8)
        aggregated.extend([
            np.mean(winrate),
            np.median(winrate),
            np.std(winrate),
            np.percentile(winrate, 25),
            np.percentile(winrate, 75),
            np.mean(np.abs(winrate - 0.5)),
            np.std(np.abs(winrate - 0.5)),
            np.mean(winrate > 0.5)  # Fraction winning
        ])
        
        # 23. Lead features (8)
        aggregated.extend([
            np.mean(lead),
            np.median(lead),
            np.std(lead),
            np.percentile(lead, 25),
            np.percentile(lead, 75),
            np.mean(np.abs(lead)),
            np.std(np.abs(lead)),
            np.mean(lead > 0)  # Fraction ahead
        ])
        
        # ===== CONSISTENCY FEATURES =====
        
        # 24. Prediction consistency (8)
        aggregated.extend([
            np.std(rank_argmax),
            np.std(policy_argmax),
            len(np.unique(rank_argmax)) / 9.0,
            len(np.unique(policy_argmax)) / 9.0,
            np.std(weighted_ranks),
            np.std(policy_weighted_rank),
            np.std(rank_max_probs),
            np.std(policy_max_probs)
        ])
        
        # ===== TEMPORAL FEATURES =====
        
        # 25. Early/middle/late game analysis (36)
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
                # Weighted rank per phase (3)
                phase_weighted = np.sum(phase_rank_probs * rank_indices, axis=1)
                aggregated.extend([
                    np.mean(phase_weighted),
                    np.median(phase_weighted),
                    np.std(phase_weighted)
                ])
            else:
                aggregated.extend([0.0] * 12)
        
        # ===== META FEATURES =====
        
        # 26. Game characteristics (5)
        aggregated.append(n_moves)
        aggregated.append(np.log1p(n_moves))  # Log scale
        # Game length categories (one-hot)
        if n_moves < 100:
            aggregated.extend([1, 0, 0])
        elif n_moves < 200:
            aggregated.extend([0, 1, 0])
        else:
            aggregated.extend([0, 0, 1])
        
        # 27. Agreement features (5)
        if len(weighted_ranks) > 1 and len(policy_weighted_rank) > 1:
            corr = np.corrcoef(weighted_ranks, policy_weighted_rank)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        aggregated.extend([
            corr,
            np.mean(np.abs(weighted_ranks - policy_weighted_rank)),
            np.std(weighted_ranks - policy_weighted_rank),
            np.median(np.abs(weighted_ranks - policy_weighted_rank)),
            np.max(np.abs(weighted_ranks - policy_weighted_rank))
        ])
        
        # Pad or truncate to 300
        result = np.array(aggregated)
        if len(result) < 300:
            result = np.pad(result, (0, 300 - len(result)), constant_values=0)
        else:
            result = result[:300]
        
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
        print("LIGHTGBM TRAINING MODE")
        print("=" * 70)
        
        print("\nLoading training data...")
        X_train, y_train = self.load_training_data(train_dir)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Ranks: {y_train}")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("\n" + "=" * 70)
        print("Training LightGBM models...")
        print("=" * 70)
        
        # LightGBM Model 1: Focus on accuracy
        lgb_model1 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='gbdt',
            n_estimators=5000,
            max_depth=8,
            learning_rate=0.01,
            num_leaves=31,
            min_child_samples=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # LightGBM Model 2: Different hyperparameters
        lgb_model2 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='gbdt',
            n_estimators=5000,
            max_depth=10,
            learning_rate=0.005,
            num_leaves=63,
            min_child_samples=1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=123,
            n_jobs=-1,
            verbose=-1
        )
        
        # LightGBM Model 3: DART boosting
        lgb_model3 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='dart',
            n_estimators=3000,
            max_depth=7,
            learning_rate=0.01,
            num_leaves=31,
            min_child_samples=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=456,
            n_jobs=-1,
            verbose=-1
        )
        
        # Random Forest as backup
        rf_model = RandomForestClassifier(
            n_estimators=2000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Train all models
        print("\n1. Training LightGBM Model 1 (GBDT, depth=8)...")
        lgb_model1.fit(X_train_scaled, y_train)
        print(f"   Training accuracy: {lgb_model1.score(X_train_scaled, y_train):.4f}")
        
        print("\n2. Training LightGBM Model 2 (GBDT, depth=10)...")
        lgb_model2.fit(X_train_scaled, y_train)
        print(f"   Training accuracy: {lgb_model2.score(X_train_scaled, y_train):.4f}")
        
        print("\n3. Training LightGBM Model 3 (DART)...")
        lgb_model3.fit(X_train_scaled, y_train)
        print(f"   Training accuracy: {lgb_model3.score(X_train_scaled, y_train):.4f}")
        
        print("\n4. Training Random Forest...")
        rf_model.fit(X_train_scaled, y_train)
        print(f"   Training accuracy: {rf_model.score(X_train_scaled, y_train):.4f}")
        
        # Create super ensemble
        print("\n" + "=" * 70)
        print("Creating LightGBM Super Ensemble...")
        print("=" * 70)
        
        ensemble = VotingClassifier(
            estimators=[
                ('lgb1', lgb_model1),
                ('lgb2', lgb_model2),
                ('lgb3', lgb_model3),
                ('rf', rf_model)
            ],
            voting='soft',
            weights=[4, 4, 3, 2],  # LightGBM models get more weight
            n_jobs=-1
        )
        
        ensemble.fit(X_train_scaled, y_train)
        ensemble_acc = ensemble.score(X_train_scaled, y_train)
        
        print(f"\nSuper Ensemble Training Accuracy: {ensemble_acc:.4f}")
        
        self.model = ensemble
        self.save_model()
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Model: LightGBM Super Ensemble (4 models)")
        print(f"Features: {X_train.shape[1]}")
        print(f"Training Accuracy: {ensemble_acc:.4f}")
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
    
    def save_model(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\n✓ Model saved to {model_path} and {scaler_path}")
    
    def load_model(self, model_path='model.pkl', scaler_path='scaler.pkl'):
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