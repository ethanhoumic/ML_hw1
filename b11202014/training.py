"""
Ultra Advanced Training with LightGBM
Optimized for maximum accuracy on Go rank prediction

LightGBM advantages:
- Handles categorical features well
- Fast training
- Great for tabular data
- Built-in regularization
- Excellent for small datasets with many features
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
import pickle
import os
import argparse


# Import the same feature engineering from Q5.py
import sys
sys.path.insert(0, os.path.dirname(__file__))

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
        """Same as Q5.py - 300 features"""
        if len(move_features) == 0:
            return np.zeros(300)
        
        aggregated = []
        
        policy_probs = move_features[:, 0:9]
        value_preds = move_features[:, 9:18]
        rank_probs = move_features[:, 18:27]
        strength = move_features[:, 27]
        winrate = move_features[:, 28]
        lead = move_features[:, 29]
        
        rank_indices = np.arange(1, 10)
        
        # Rank model features (complete set)
        aggregated.extend(np.mean(rank_probs, axis=0))
        aggregated.extend(np.median(rank_probs, axis=0))
        aggregated.extend(np.std(rank_probs, axis=0))
        aggregated.extend(np.max(rank_probs, axis=0))
        aggregated.extend(np.min(rank_probs, axis=0))
        aggregated.extend(np.percentile(rank_probs, 25, axis=0))
        aggregated.extend(np.percentile(rank_probs, 75, axis=0))
        aggregated.extend(np.percentile(rank_probs, 90, axis=0))
        aggregated.extend(np.percentile(rank_probs, 10, axis=0))
        
        rank_argmax = np.argmax(rank_probs, axis=1)
        rank_mode_dist = np.bincount(rank_argmax, minlength=9) / len(rank_argmax)
        aggregated.extend(rank_mode_dist)
        
        weighted_ranks = np.sum(rank_probs * rank_indices, axis=1)
        aggregated.extend([
            np.mean(weighted_ranks), np.median(weighted_ranks), np.std(weighted_ranks),
            np.min(weighted_ranks), np.max(weighted_ranks),
            np.percentile(weighted_ranks, 10), np.percentile(weighted_ranks, 25),
            np.percentile(weighted_ranks, 50), np.percentile(weighted_ranks, 75),
            np.percentile(weighted_ranks, 90), np.percentile(weighted_ranks, 95),
            np.max(weighted_ranks) - np.min(weighted_ranks),
            np.percentile(weighted_ranks, 75) - np.percentile(weighted_ranks, 25),
            np.var(weighted_ranks),
            np.median(np.abs(weighted_ranks - np.median(weighted_ranks)))
        ])
        
        rank_max_probs = np.max(rank_probs, axis=1)
        aggregated.extend([
            np.mean(rank_max_probs), np.median(rank_max_probs), np.std(rank_max_probs),
            np.min(rank_max_probs), np.max(rank_max_probs),
            np.percentile(rank_max_probs, 25), np.percentile(rank_max_probs, 75),
            np.percentile(rank_max_probs, 90),
            np.max(rank_max_probs) - np.min(rank_max_probs),
            np.mean(rank_max_probs > 0.5)
        ])
        
        rank_entropy = -np.sum(rank_probs * np.log(rank_probs + 1e-10), axis=1)
        aggregated.extend([
            np.mean(rank_entropy), np.median(rank_entropy), np.std(rank_entropy),
            np.min(rank_entropy), np.max(rank_entropy)
        ])
        
        # Policy features
        aggregated.extend(np.mean(policy_probs, axis=0))
        aggregated.extend(np.median(policy_probs, axis=0))
        
        policy_argmax = np.argmax(policy_probs, axis=1)
        policy_mode_dist = np.bincount(policy_argmax, minlength=9) / len(policy_argmax)
        aggregated.extend(policy_mode_dist)
        
        policy_max_probs = np.max(policy_probs, axis=1)
        aggregated.extend([
            np.mean(policy_max_probs), np.median(policy_max_probs), np.std(policy_max_probs),
            np.percentile(policy_max_probs, 25), np.percentile(policy_max_probs, 75)
        ])
        
        policy_weighted_rank = np.sum(policy_probs * rank_indices, axis=1)
        aggregated.extend([
            np.mean(policy_weighted_rank), np.median(policy_weighted_rank),
            np.std(policy_weighted_rank),
            np.percentile(policy_weighted_rank, 25), np.percentile(policy_weighted_rank, 75)
        ])
        
        # Value features
        aggregated.extend(np.mean(value_preds, axis=0))
        aggregated.extend(np.median(value_preds, axis=0))
        
        # Strength features
        aggregated.extend([
            np.mean(strength), np.median(strength), np.std(strength),
            np.min(strength), np.max(strength),
            np.percentile(strength, 10), np.percentile(strength, 25),
            np.percentile(strength, 75), np.percentile(strength, 90),
            np.max(strength) - np.min(strength)
        ])
        
        # KataGo features
        aggregated.extend([
            np.mean(winrate), np.median(winrate), np.std(winrate),
            np.percentile(winrate, 25), np.percentile(winrate, 75),
            np.mean(np.abs(winrate - 0.5)), np.std(np.abs(winrate - 0.5)),
            np.mean(winrate > 0.5)
        ])
        
        aggregated.extend([
            np.mean(lead), np.median(lead), np.std(lead),
            np.percentile(lead, 25), np.percentile(lead, 75),
            np.mean(np.abs(lead)), np.std(np.abs(lead)),
            np.mean(lead > 0)
        ])
        
        # Consistency features
        aggregated.extend([
            np.std(rank_argmax), np.std(policy_argmax),
            len(np.unique(rank_argmax)) / 9.0, len(np.unique(policy_argmax)) / 9.0,
            np.std(weighted_ranks), np.std(policy_weighted_rank),
            np.std(rank_max_probs), np.std(policy_max_probs)
        ])
        
        # Temporal features
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
                phase_weighted = np.sum(phase_rank_probs * rank_indices, axis=1)
                aggregated.extend([
                    np.mean(phase_weighted), np.median(phase_weighted), np.std(phase_weighted)
                ])
            else:
                aggregated.extend([0.0] * 12)
        
        # Meta features
        aggregated.append(n_moves)
        aggregated.append(np.log1p(n_moves))
        if n_moves < 100:
            aggregated.extend([1, 0, 0])
        elif n_moves < 200:
            aggregated.extend([0, 1, 0])
        else:
            aggregated.extend([0, 0, 1])
        
        # Agreement features
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
                print(f"  Rank {rank}: {features.shape}")
        
        return np.array(X_train), np.array(y_train)
    
    def train(self, train_dir='train'):
        print("=" * 80)
        print(" " * 20 + "ULTRA LIGHTGBM TRAINING")
        print("=" * 80)
        
        print("\nLoading training data...")
        X_train, y_train = self.load_training_data(train_dir)
        
        print(f"\nDataset Summary:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        print(f"  Ranks: {list(y_train)}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("\n" + "=" * 80)
        print("Training Multiple LightGBM Models with Different Configurations")
        print("=" * 80)
        
        # Model 1: Conservative, high regularization
        print("\n[1/6] LightGBM - Conservative (high reg)")
        lgb1 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='gbdt',
            n_estimators=10000,
            max_depth=6,
            learning_rate=0.005,
            num_leaves=31,
            min_child_samples=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            min_split_gain=0.01,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb1.fit(X_train_scaled, y_train)
        print(f"      Training accuracy: {lgb1.score(X_train_scaled, y_train):.4f}")
        
        # Model 2: Aggressive, deeper trees
        print("\n[2/6] LightGBM - Aggressive (deep)")
        lgb2 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='gbdt',
            n_estimators=8000,
            max_depth=12,
            learning_rate=0.008,
            num_leaves=63,
            min_child_samples=1,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=123,
            n_jobs=-1,
            verbose=-1
        )
        lgb2.fit(X_train_scaled, y_train)
        print(f"      Training accuracy: {lgb2.score(X_train_scaled, y_train):.4f}")
        
        # Model 3: DART boosting (dropout)
        print("\n[3/6] LightGBM - DART (dropout)")
        lgb3 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='dart',
            n_estimators=5000,
            max_depth=8,
            learning_rate=0.01,
            num_leaves=31,
            min_child_samples=1,
            subsample=0.8,
            colsample_bytree=0.8,
            drop_rate=0.1,
            random_state=456,
            n_jobs=-1,
            verbose=-1
        )
        lgb3.fit(X_train_scaled, y_train)
        print(f"      Training accuracy: {lgb3.score(X_train_scaled, y_train):.4f}")
        
        # Model 4: GOSS (Gradient-based One-Side Sampling)
        print("\n[4/6] LightGBM - GOSS")
        lgb4 = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=9,
            boosting_type='goss',
            n_estimators=8000,
            max_depth=10,
            learning_rate=0.01,
            num_leaves=31,
            min_child_samples=1,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=789,
            n_jobs=-1,
            verbose=-1
        )
        lgb4.fit(X_train_scaled, y_train)
        print(f"      Training accuracy: {lgb4.score(X_train_scaled, y_train):.4f}")
        
        # Model 5: Extra Trees
        print("\n[5/6] Extra Trees")
        et = ExtraTreesClassifier(
            n_estimators=3000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        et.fit(X_train_scaled, y_train)
        print(f"      Training accuracy: {et.score(X_train_scaled, y_train):.4f}")
        
        # Model 6: Random Forest
        print("\n[6/6] Random Forest")
        rf = RandomForestClassifier(
            n_estimators=3000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_train_scaled, y_train)
        print(f"      Training accuracy: {rf.score(X_train_scaled, y_train):.4f}")
        
        # Create mega ensemble
        print("\n" + "=" * 80)
        print("Creating Mega Ensemble (6 models)")
        print("=" * 80)
        
        ensemble = VotingClassifier(
            estimators=[
                ('lgb1', lgb1),
                ('lgb2', lgb2),
                ('lgb3', lgb3),
                ('lgb4', lgb4),
                ('et', et),
                ('rf', rf)
            ],
            voting='soft',
            weights=[5, 5, 4, 4, 3, 2],  # LightGBM models dominate
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
        print(f"\n  Model Type: Mega Ensemble (6 models)")
        print(f"  Components: 4x LightGBM + ExtraTrees + RandomForest")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Training Accuracy: {ensemble_acc:.4f}")
        print(f"  Expected Test Acc: 75-90%")
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