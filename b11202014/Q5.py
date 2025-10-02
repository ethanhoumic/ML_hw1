"""
Go Rank Prediction ML Model
Assignment 1 Q5 - Machine Learning Class Fall 2025

Run with --train flag to train the model, otherwise it generates predictions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import argparse


class GoRankPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def parse_game_file(self, filepath):
        """Parse a single game file and extract features from all moves."""
        features_list = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
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
    
    def aggregate_features(self, move_features):
        """
        Aggregate move-level features into file-level features.
        Each move has 30 features, we compute 8 statistics per feature.
        Total output: 30 * 8 = 240 features
        """
        expected_dim = 240  # 30 features * 8 stats
        
        if len(move_features) == 0:
            return np.zeros(expected_dim)
        
        aggregated = []
        
        # For each of the 30 features per move
        for feat_idx in range(30):
            if feat_idx < move_features.shape[1]:
                feat_values = move_features[:, feat_idx]
                
                # 8 different statistics
                aggregated.extend([
                    np.mean(feat_values),
                    np.std(feat_values),
                    np.median(feat_values),
                    np.percentile(feat_values, 25),
                    np.percentile(feat_values, 75),
                    np.max(feat_values),
                    np.min(feat_values),
                    np.max(feat_values) - np.min(feat_values)  # Range
                ])
            else:
                # If somehow we don't have this feature, pad with zeros
                aggregated.extend([0.0] * 8)
        
        return np.array(aggregated[:expected_dim])
    
    def extract_features_from_file(self, filepath):
        """Extract and aggregate features from a single file."""
        move_features = self.parse_game_file(filepath)
        return self.aggregate_features(move_features)
    
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
                print(f"  Features shape: {features.shape}")
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
        
        # Model 1: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Model 2: Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        # Model 3: Logistic Regression
        lr_model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
            multi_class='multinomial'
        )
        
        # Train each model
        print("\nTraining Random Forest...")
        rf_model.fit(X_train_scaled, y_train)
        rf_acc = rf_model.score(X_train_scaled, y_train)
        print(f"  Training accuracy: {rf_acc:.4f}")
        
        print("Training Gradient Boosting...")
        gb_model.fit(X_train_scaled, y_train)
        gb_acc = gb_model.score(X_train_scaled, y_train)
        print(f"  Training accuracy: {gb_acc:.4f}")
        
        print("Training Logistic Regression...")
        lr_model.fit(X_train_scaled, y_train)
        lr_acc = lr_model.score(X_train_scaled, y_train)
        print(f"  Training accuracy: {lr_acc:.4f}")
        
        # Create ensemble
        print("\nCreating ensemble model...")
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('lr', lr_model)
            ],
            voting='soft',
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
        
        for filename in test_files:
            filepath = os.path.join(test_dir, filename)
            file_id = filename.split('.')[0]
            
            # Extract features
            features = self.extract_features_from_file(filepath)
            if (file_id % 50 == 0) :
                print(f"File {file_id}: features shape = {features.shape}")
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict rank
            pred_rank = self.model.predict(features_scaled)[0]
            predictions.append({'id': int(file_id), 'rank': pred_rank})
        
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
    
    predictor = GoRankPredictor()
    
    if args.train:
        # Training mode
        print("=" * 60)
        print("TRAINING MODE")
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
        
        print(f"\nPredictions saved to submission.csv")
        print(f"Total predictions: {len(predictions)}")
        print("\nFirst few predictions:")
        print(df.head(10))
        print("\nSubmission file ready for Kaggle upload!")


if __name__ == '__main__':
    main()