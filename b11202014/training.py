"""
Ultra Advanced Training with LightGBM (Anti-Overfitting Version)
Optimized for better generalization on small datasets

Key Anti-Overfitting Strategies:
1. Reduced model complexity (fewer estimators, shallower trees)
2. Stronger regularization
3. Simpler feature set (150 features vs 300)
4. Conservative hyperparameters
5. Fewer ensemble members
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
import pickle
import os
import argparse


class UltraLightGBMPredictor:
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        
    def parse_game_file(self, filepath):
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
        """Simplified feature engineering - 150 features"""
        if len(move_features) == 0:
            return np.zeros(150)
        
        aggregated = []
        
        policy_probs = move_features[:, 0:9]
        value_preds = move_features[:, 9:18]
        rank_probs = move_features[:, 18:27]
        strength = move_features[:, 27]
        winrate = move_features[:, 28]
        lead = move_features[:, 29]
        
        rank_indices = np.arange(1, 10)
        
        # Rank model features - simplified
        aggregated.extend(np.mean(rank_probs, axis=0))
        aggregated.extend(np.median(rank_probs, axis=0))
        aggregated.extend(np.std(rank_probs, axis=0))
        
        weighted_ranks = np.sum(rank_probs * rank_indices, axis=1)
        aggregated.extend([
            np.mean(weighted_ranks),
            np.median(weighted_ranks),
            np.std(weighted_ranks),
            np.percentile(weighted_ranks, 25),
            np.percentile(weighted_ranks, 75)
        ])
        
        rank_max_probs = np.max(rank_probs, axis=1)
        aggregated.extend([
            np.mean(rank_max_probs),
            np.median(rank_max_probs),
            np.std(rank_max_probs),
            np.percentile(rank_max_probs, 25),
            np.percentile(rank_max_probs, 75)
        ])
        
        rank_entropy = -np.sum(rank_probs * np.log(rank_probs + 1e-10), axis=1)
        aggregated.extend([
            np.mean(rank_entropy),
            np.median(rank_entropy),
            np.std(rank_entropy)
        ])
        
        # Policy features - simplified
        aggregated.extend(np.mean(policy_probs, axis=0))
        
        policy_max_probs = np.max(policy_probs, axis=1)
        aggregated.extend([
            np.mean(policy_max_probs),
            np.median(policy_max_probs),
            np.std(policy_max_probs)
        ])
        
        # Value features
        aggregated.extend(np.mean(value_preds, axis=0))
        
        # Strength features
        aggregated.extend([
            np.mean(strength),
            np.median(strength),
            np.std(strength),
            np.percentile(strength, 25),
            np.percentile(strength, 75)
        ])
        
        # KataGo features
        aggregated.extend([
            np.mean(winrate),
            np.median(winrate),
            np.std(winrate),
            np.percentile(winrate, 25),
            np.percentile(winrate, 75)
        ])
        
        aggregated.extend([
            np.mean(lead),
            np.median(lead),
            np.std(lead),
            np.percentile(lead, 25),
            np.percentile(lead, 75)
        ])
        
        # Temporal features - simplified
        n_moves = len(move_features)
        third = max(1, n_moves // 3)
        
        phases = [
            move_features[:third],
            move_features[third:2*third],
            move_features[2*third:]
        ]
        
        for phase_moves in phases:
            if len(phase_moves) > 0:
                phase_rank_probs = phase_moves[:, 18:27]
                aggregated.extend(np.mean(phase_rank_probs, axis=0))
            else:
                aggregated.extend([0.0] * 9)
        
        # Meta features
        aggregated.append(np.log1p(n_moves))
        aggregated.append(min(n_moves / 300.0, 1.0))
        
        rank_argmax = np.argmax(rank_probs, axis=1)
        aggregated.extend([
            np.std(rank_argmax),
            np.std(weighted_ranks),
            len(np.unique(rank_argmax)) / 9.0
        ])
        
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
                print(f"  Rank {rank}: {features.shape}")
        
        return np.array(X_train), np.array(y_train)
    
    def train(self, train_dir='train'):
        print("=" * 80)
        print(" " * 15 + "ULTRA LIGHTGBM TRAINING (ANTI-OVERFITTING)")
        print("=" * 80)
        
        print("\nLoading training data...")
        X_train, y_train = self.load_training_data(train_dir)
        
        print(f"\nDataset Summary:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        print(f"  Ranks: {list(y_train)}")
        print(f"\n⚠️  Small dataset warning: Using conservative hyperparameters")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("\n" + "=" * 80)
        print("Training Conservative Models (Reduced Complexity)")
        print("=" * 80)
        
        # Model 1: Ultra conservative
        print("\n[1/4] LightGBM - Ultra Conservative")
        lgb1 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='gbdt',
            n_estimators=150,  # Drastically reduced from 10000
            max_depth=3,  # Very shallow from 6
            learning_rate=0.1,  # Higher for faster convergence
            num_leaves=7,  # Reduced from 31
            min_child_samples=1,
            subsample=0.8,
            colsample_bytree=0.5,  # Reduced from 0.8
            reg_alpha=2.0,  # Strong regularization from 0.5
            reg_lambda=2.0,  # Strong regularization from 0.5
            min_split_gain=0.2,  # High threshold from 0.01
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb1.fit(X_train_scaled, y_train)
        acc1 = lgb1.score(X_train_scaled, y_train)
        print(f"      Training accuracy: {acc1:.4f}")
        
        # Model 2: Conservative
        print("\n[2/4] LightGBM - Conservative")
        lgb2 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='gbdt',
            n_estimators=250,  # Reduced from 8000
            max_depth=4,  # Shallow from 12
            learning_rate=0.08,  # Higher from 0.008
            num_leaves=10,  # Reduced from 63
            min_child_samples=1,
            subsample=0.7,
            colsample_bytree=0.6,  # Reduced from 0.75
            reg_alpha=1.5,  # Increased from 0.2
            reg_lambda=1.5,  # Increased from 0.2
            min_split_gain=0.1,  # Increased
            random_state=123,
            n_jobs=-1,
            verbose=-1
        )
        lgb2.fit(X_train_scaled, y_train)
        acc2 = lgb2.score(X_train_scaled, y_train)
        print(f"      Training accuracy: {acc2:.4f}")
        
        # Model 3: Extra Trees (naturally resistant to overfitting)
        print("\n[3/4] Extra Trees")
        et = ExtraTreesClassifier(
            n_estimators=300,  # Reduced from 3000
            max_depth=6,  # Limited depth
            min_samples_split=3,  # Increased from 2
            min_samples_leaf=2,  # Increased from 1
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        et.fit(X_train_scaled, y_train)
        acc3 = et.score(X_train_scaled, y_train)
        print(f"      Training accuracy: {acc3:.4f}")
        
        # Model 4: Random Forest
        print("\n[4/4] Random Forest")
        rf = RandomForestClassifier(
            n_estimators=300,  # Reduced from 3000
            max_depth=6,  # Limited depth
            min_samples_split=3,  # Increased from 2
            min_samples_leaf=2,  # Increased from 1
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_train_scaled, y_train)
        acc4 = rf.score(X_train_scaled, y_train)
        print(f"      Training accuracy: {acc4:.4f}")
        
        # Create conservative ensemble
        print("\n" + "=" * 80)
        print("Creating Conservative Ensemble (4 models)")
        print("=" * 80)
        
        ensemble = VotingClassifier(
            estimators=[
                ('lgb1', lgb1),
                ('lgb2', lgb2),
                ('et', et),
                ('rf', rf)
            ],
            voting='soft',
            weights=[2, 2, 1, 1],  # Balanced weights
            n_jobs=-1
        )
        
        print("\nTraining ensemble...")
        ensemble.fit(X_train_scaled, y_train)
        ensemble_acc = ensemble.score(X_train_scaled, y_train)
        
        self.model = ensemble
        self.save_model()
        
        print("\n" + "=" * 80)
        print(" " * 25 + "✓ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\n  Model Type: Conservative Ensemble (4 models)")
        print(f"  Components: 2x LightGBM + ExtraTrees + RandomForest")
        print(f"  Features: {X_train.shape[1]} (reduced from 300)")
        print(f"  Training Accuracy: {ensemble_acc:.4f}")
        print(f"  Individual accuracies:")
        print(f"    - LGB Ultra Conservative: {acc1:.4f}")
        print(f"    - LGB Conservative: {acc2:.4f}")
        print(f"    - Extra Trees: {acc3:.4f}")
        print(f"    - Random Forest: {acc4:.4f}")
        
        if ensemble_acc > 0.95:
            print(f"\n  ⚠️  WARNING: Training accuracy very high (>95%)")
            print(f"      Model may still overfit with only {len(X_train)} samples")
            print(f"      Expected test accuracy: 60-80%")
        else:
            print(f"\n  ✓ Training accuracy reasonable for small dataset")
            print(f"    Expected test accuracy: 70-85%")
        
        print("\n" + "=" * 80)
        
        return ensemble_acc
    
    def save_model(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\n✓ Model saved to {model_path} and {scaler_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='train')
    args = parser.parse_args()
    
    predictor = UltraLightGBMPredictor()
    predictor.train(args.train_dir)


if __name__ == '__main__':
    main()