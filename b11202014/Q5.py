import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy import stats
import pickle
import argparse

def debug_file_format(filepath, num_lines=50):
    """Debug helper to inspect file format."""
    print(f"\n{'='*60}")
    print(f"DEBUG: First {num_lines} lines of {filepath}")
    print('='*60)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(f"Line {i}: {repr(line.strip())}")
    except Exception as e:
        print(f"Error reading file: {e}")
    print('='*60)

def parse_game_file(filepath, split_games=False):
    """Parse a single game file and extract features.
    
    Args:
        filepath: Path to the file
        split_games: If True, return features split by game. If False, return all features together.
    
    Returns:
        If split_games=True: List of lists (each inner list is features for one game)
        If split_games=False: Single list of all features
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if split_games:
        games_features = []
        current_game_features = []
    else:
        features_list = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Game header indicates start of new game
        if line.startswith('Game'):
            if split_games and current_game_features:
                # Save previous game
                games_features.append(current_game_features)
                current_game_features = []
            i += 1
            continue
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this line contains a move (starts with B[ or W[)
        if re.match(r'^[BW]\[', line):
            move = line
            color = 'B' if move[0] == 'B' else 'W'
            
            # Check if we have enough lines for a complete move
            if i + 4 < len(lines):
                try:
                    # Line i+1: Policy probabilities (9 values)
                    policy_probs = [float(x) for x in lines[i+1].strip().split()]
                    
                    # Line i+2: Value predictions (9 values with % signs)
                    value_line = lines[i+2].strip()
                    value_preds = [float(x.replace('%', '')) / 100.0 for x in value_line.split()]
                    
                    # Line i+3: Rank model outputs (9 values)
                    rank_outputs = [float(x) for x in lines[i+3].strip().split()]
                    
                    # Line i+4: Strength score (1 value)
                    strength_score = float(lines[i+4].strip())
                    
                    # Line i+5: KataGo values (3 values)
                    katago_parts = lines[i+5].strip().split()
                    winrate = float(katago_parts[0].replace('%', '')) / 100.0
                    lead = float(katago_parts[1])
                    uncertainty = float(katago_parts[2])
                    
                    # Adjust KataGo values based on player color
                    if color == 'W':
                        winrate = 1.0 - winrate
                        lead = -lead
                    
                    # Verify we have the correct number of features
                    if len(policy_probs) == 9 and len(value_preds) == 9 and len(rank_outputs) == 9:
                        # Construct feature vector (30 features total)
                        features = policy_probs + value_preds + rank_outputs + [strength_score, winrate, lead, uncertainty]
                        
                        if split_games:
                            current_game_features.append(features)
                        else:
                            features_list.append(features)
                    
                    i += 6  # Move to next move block (move + 5 feature lines)
                except (ValueError, IndexError) as e:
                    # Skip malformed entries
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    # Don't forget the last game if splitting
    if split_games and current_game_features:
        games_features.append(current_game_features)
        return games_features
    else:
        return features_list

def aggregate_features(features_list):
    """Aggregate features from multiple moves into a single sample with enhanced statistics."""
    if not features_list:
        return None
    
    features_array = np.array(features_list)
    
    # Basic statistics: mean, std, min, max, median
    mean_features = np.mean(features_array, axis=0)
    std_features = np.std(features_array, axis=0)
    min_features = np.min(features_array, axis=0)
    max_features = np.max(features_array, axis=0)
    median_features = np.median(features_array, axis=0)
    
    # Additional statistics for better discrimination
    q10_features = np.percentile(features_array, 10, axis=0)
    q25_features = np.percentile(features_array, 25, axis=0)
    q75_features = np.percentile(features_array, 75, axis=0)
    q90_features = np.percentile(features_array, 90, axis=0)
    
    # Range features
    range_features = max_features - min_features
    iqr_features = q75_features - q25_features
    
    # Skewness and kurtosis (distribution shape)
    skew_features = stats.skew(features_array, axis=0)
    kurt_features = stats.kurtosis(features_array, axis=0)
    
    # Replace any NaN values with 0
    skew_features = np.nan_to_num(skew_features, nan=0.0)
    kurt_features = np.nan_to_num(kurt_features, nan=0.0)
    
    # Cross-features: relationships between the 9 rank model outputs
    # Extract rank model outputs (indices 18-26)
    rank_outputs = features_array[:, 18:27]
    rank_mean = np.mean(rank_outputs, axis=0)
    rank_std = np.std(rank_outputs, axis=0)
    
    # Entropy of rank outputs (measure of uncertainty)
    rank_entropy = []
    for i in range(rank_outputs.shape[0]):
        probs = rank_outputs[i, :] + 1e-10  # Avoid log(0)
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs))
        rank_entropy.append(entropy)
    rank_entropy_stats = [np.mean(rank_entropy), np.std(rank_entropy), np.max(rank_entropy), np.min(rank_entropy)]
    
    # Concatenate all statistics
    aggregated = np.concatenate([
        mean_features, 
        std_features, 
        min_features, 
        max_features, 
        median_features,
        q10_features,
        q25_features,
        q75_features,
        q90_features,
        range_features,
        iqr_features,
        skew_features,
        kurt_features,
        rank_mean,
        rank_std,
        rank_entropy_stats
    ])
    
    return aggregated

def load_training_data(train_dir='train', games_per_sample=5):
    """Load and process training data.
    
    Args:
        train_dir: Directory containing training files
        games_per_sample: Number of games to aggregate into one training sample
    """
    X_train = []
    y_train = []
    
    for rank in range(1, 10):  # 1D to 9D
        filename = f'log_{rank}D_policy_train.txt'
        filepath = os.path.join(train_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        print(f"Processing {filename}...")
        # Parse games separately
        games_features = parse_game_file(filepath, split_games=True)
        print(f"  Found {len(games_features)} games")
        
        # Create multiple samples by grouping games
        if len(games_features) >= games_per_sample:
            # Group games into samples
            num_samples = len(games_features) // games_per_sample
            for sample_idx in range(num_samples):
                start_idx = sample_idx * games_per_sample
                end_idx = start_idx + games_per_sample
                
                # Combine features from multiple games
                combined_features = []
                for game_features in games_features[start_idx:end_idx]:
                    combined_features.extend(game_features)
                
                # Aggregate features
                if len(combined_features) > 0:
                    aggregated = aggregate_features(combined_features)
                    if aggregated is not None and len(aggregated) > 0:
                        X_train.append(aggregated)
                        y_train.append(rank - 1)  # Convert to 0-indexed (0-8)
            
            print(f"  -> Created {num_samples} training samples from {len(games_features)} games")
        else:
            # If too few games, use all games as one sample
            all_features = []
            for game_features in games_features:
                all_features.extend(game_features)
            
            if len(all_features) > 0:
                aggregated = aggregate_features(all_features)
                if aggregated is not None and len(aggregated) > 0:
                    X_train.append(aggregated)
                    y_train.append(rank - 1)
            print(f"  -> Created 1 training sample from {len(games_features)} games (too few to split)")
    
    if len(X_train) == 0:
        print("\nERROR: No training data loaded!")
        print(f"Please check that training files exist in '{train_dir}' directory")
        print("Expected files: log_1D_policy_train.txt, log_2D_policy_train.txt, ..., log_9D_policy_train.txt")
    else:
        print(f"\nSuccessfully loaded {len(X_train)} training samples")
        print(f"Class distribution: {np.bincount(y_train)}")
    
    return np.array(X_train), np.array(y_train)

def load_test_data(test_dir='test'):
    """Load and process test data."""
    X_test = []
    test_files = []
    
    for filename in sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0):
        if filename.endswith('.txt'):
            filepath = os.path.join(test_dir, filename)
            print(f"Processing {filename}...")
            
            # For test data, aggregate all moves in the file (don't split by game)
            features_list = parse_game_file(filepath, split_games=False)
            aggregated = aggregate_features(features_list)
            
            if aggregated is not None:
                X_test.append(aggregated)
                test_files.append(filename.replace('.txt', ''))
    
    return np.array(X_test), test_files

def train_single_model(X_train, y_train, params, num_boost_round=2000):
    """Train a single LightGBM model."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=params.get('seed', 42), stratify=y_train
    )
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)]
    )
    
    # Evaluate
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_val, y_pred_class)
    
    return model, accuracy

def train_model_ensemble(X_train, y_train, output_path='lgb_model.pkl', n_models=5):
    """Train an ensemble of LightGBM models with different configurations."""
    print(f"\nTraining ensemble of {n_models} LightGBM models...")
    
    models = []
    accuracies = []
    
    # Different configurations for diversity
    configs = [
        {'num_leaves': 63, 'max_depth': 8, 'learning_rate': 0.03, 'lambda_l1': 0.5, 'lambda_l2': 0.5},
        {'num_leaves': 95, 'max_depth': 9, 'learning_rate': 0.025, 'lambda_l1': 0.3, 'lambda_l2': 0.3},
        {'num_leaves': 47, 'max_depth': 7, 'learning_rate': 0.04, 'lambda_l1': 0.7, 'lambda_l2': 0.7},
        {'num_leaves': 80, 'max_depth': 10, 'learning_rate': 0.02, 'lambda_l1': 0.4, 'lambda_l2': 0.4},
        {'num_leaves': 55, 'max_depth': 8, 'learning_rate': 0.035, 'lambda_l1': 0.6, 'lambda_l2': 0.6},
    ]
    
    for i in range(n_models):
        print(f"\n{'='*60}")
        print(f"Training Model {i+1}/{n_models}")
        print('='*60)
        
        config = configs[i % len(configs)]
        params = {
            'objective': 'multiclass',
            'num_class': 9,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'seed': 42 + i,
            **config
        }
        
        model, acc = train_single_model(X_train, y_train, params)
        models.append(model)
        accuracies.append(acc)
        print(f"Model {i+1} Validation Accuracy: {acc:.4f}")
    
    avg_accuracy = np.mean(accuracies)
    print(f"\n{'='*60}")
    print(f"Ensemble Average Validation Accuracy: {avg_accuracy:.4f}")
    print('='*60)
    
    # Save ensemble
    with open(output_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Ensemble saved to {output_path}")
    
    return models

def predict_and_save(models, X_test, test_files, output_path='submission.csv'):
    """Generate predictions using ensemble and save to CSV."""
    print("\nGenerating ensemble predictions...")
    
    # Handle both single model and ensemble
    if not isinstance(models, list):
        models = [models]
    
    # Collect predictions from all models
    all_predictions = []
    for i, model in enumerate(models):
        pred = model.predict(X_test, num_iteration=model.best_iteration)
        all_predictions.append(pred)
        print(f"Model {i+1} predictions collected")
    
    # Average predictions
    avg_predictions = np.mean(all_predictions, axis=0)
    predicted_ranks = np.argmax(avg_predictions, axis=1) + 1  # Convert back to 1-9
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_files,
        'rank': predicted_ranks
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"\nSample predictions:")
    print(submission.head(10))

def main():
    parser = argparse.ArgumentParser(description='Go Rank Prediction')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--train_dir', type=str, default='train', help='Training data directory')
    parser.add_argument('--test_dir', type=str, default='test', help='Test data directory')
    parser.add_argument('--model_path', type=str, default='lgb_model.pkl', help='Model file path')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output CSV file')
    parser.add_argument('--debug', action='store_true', help='Debug mode: show file format')
    parser.add_argument('--n_models', type=int, default=5, help='Number of models in ensemble')
    
    args = parser.parse_args()
    
    # Debug mode
    if args.debug:
        print("DEBUG MODE: Inspecting file format...")
        test_file = os.path.join(args.train_dir, 'log_1D_policy_train.txt')
        if os.path.exists(test_file):
            debug_file_format(test_file, 50)
        else:
            print(f"File not found: {test_file}")
            print(f"Available files in {args.train_dir}:")
            if os.path.exists(args.train_dir):
                for f in os.listdir(args.train_dir):
                    print(f"  - {f}")
        return
    
    if args.train:
        # Training mode
        print("=" * 50)
        print("TRAINING MODE")
        print("=" * 50)
        X_train, y_train = load_training_data(args.train_dir)
        print(f"\nTraining data shape: {X_train.shape}")
        
        if len(X_train) == 0:
            print("\nERROR: No training data found. Exiting.")
            print("Try running with --debug flag to inspect file format:")
            print("  python Q5.py --debug")
            return
        
        models = train_model_ensemble(X_train, y_train, args.model_path, args.n_models)
    else:
        # Evaluation mode (default)
        print("=" * 50)
        print("EVALUATION MODE")
        print("=" * 50)
        
        # Load model
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found!")
            print("Please train the model first using: python Q5.py --train")
            return
        
        with open(args.model_path, 'rb') as f:
            models = pickle.load(f)
        print(f"Model(s) loaded from {args.model_path}")
        
        # Load test data and predict
        X_test, test_files = load_test_data(args.test_dir)
        print(f"\nTest data shape: {X_test.shape}")
        predict_and_save(models, X_test, test_files, args.output)

if __name__ == "__main__":
    main()