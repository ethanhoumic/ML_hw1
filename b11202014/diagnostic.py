"""
Diagnostic Script - Analyze what's actually in your data
Run this to understand why accuracy might be low
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def parse_file(filepath):
    """Parse a single file."""
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
            try:
                policy_line = lines[i + 1].strip().split()
                value_line = lines[i + 2].strip().split()
                rank_line = lines[i + 3].strip().split()
                
                policy_probs = [float(x) for x in policy_line]
                value_preds = [float(x.rstrip('%')) / 100.0 for x in value_line]
                rank_probs = [float(x) for x in rank_line]
                
                features_list.append(policy_probs + value_preds + rank_probs)
                i += 6
            except (IndexError, ValueError):
                i += 1
                continue
        else:
            i += 1
    
    return np.array(features_list) if features_list else np.zeros((0, 27))


def analyze_training_data(train_dir='train'):
    """Analyze training data to see if rank model outputs are discriminative."""
    
    print("=" * 80)
    print("DIAGNOSTIC ANALYSIS - Training Data")
    print("=" * 80)
    
    rank_data = {}
    
    for rank in range(1, 10):
        filename = f'log_{rank}D_policy_train.txt'
        filepath = os.path.join(train_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  {filename} not found!")
            continue
        
        print(f"\nAnalyzing {filename}...")
        features = parse_file(filepath)
        
        if len(features) == 0:
            print(f"  ⚠️  No moves found!")
            continue
        
        print(f"  Total moves: {len(features)}")
        
        # Extract rank probabilities
        rank_probs = features[:, 18:27]
        
        # Calculate statistics
        mean_probs = np.mean(rank_probs, axis=0)
        weighted_rank = np.sum(mean_probs * np.arange(1, 10))
        most_likely_rank = np.argmax(mean_probs) + 1
        confidence = np.max(mean_probs)
        
        rank_data[rank] = {
            'mean_probs': mean_probs,
            'weighted': weighted_rank,
            'most_likely': most_likely_rank,
            'confidence': confidence,
            'n_moves': len(features)
        }
        
        print(f"  Rank probabilities (mean across all moves):")
        for r in range(1, 10):
            bar = '#' * int(mean_probs[r-1] * 50)
            print(f"    {r}D: {mean_probs[r-1]:.4f} {bar}")
        
        print(f"  Weighted rank prediction: {weighted_rank:.2f}")
        print(f"  Most likely rank: {most_likely_rank}D (confidence: {confidence:.3f})")
        
        # Check if prediction matches actual rank
        if most_likely_rank == rank:
            print(f"  ✓ Rank model correctly identifies this as {rank}D")
        else:
            print(f"  ⚠️  Rank model thinks this is {most_likely_rank}D, but it's actually {rank}D")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Check if weighted predictions are monotonic
    print("\nWeighted rank predictions per file:")
    weighted_preds = []
    for rank in sorted(rank_data.keys()):
        w = rank_data[rank]['weighted']
        weighted_preds.append(w)
        match = "✓" if abs(w - rank) < 1.0 else "⚠️"
        print(f"  {rank}D file: {w:.2f} {match}")
    
    # Check if predictions are ordered correctly
    is_monotonic = all(weighted_preds[i] <= weighted_preds[i+1] 
                      for i in range(len(weighted_preds)-1))
    
    if is_monotonic:
        print("\n✓ Predictions are monotonically increasing - GOOD!")
        print("  The rank model can distinguish between ranks.")
    else:
        print("\n⚠️  Predictions are NOT monotonic - PROBLEM!")
        print("  The rank model may not be discriminative enough.")
        print("  Consider using different features or aggregations.")
    
    # Calculate correlation
    actual_ranks = list(rank_data.keys())
    predicted_ranks = [rank_data[r]['weighted'] for r in actual_ranks]
    correlation = np.corrcoef(actual_ranks, predicted_ranks)[0, 1]
    
    print(f"\nCorrelation between actual and predicted ranks: {correlation:.4f}")
    if correlation > 0.9:
        print("  ✓ Excellent correlation! Model should work well.")
    elif correlation > 0.7:
        print("  ✓ Good correlation. Model should work reasonably.")
    elif correlation > 0.5:
        print("  ⚠️  Moderate correlation. May need better features.")
    else:
        print("  ❌ Poor correlation. Features may not be discriminative.")
    
    # Visualize
    try:
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Weighted predictions vs actual
        plt.subplot(1, 2, 1)
        plt.plot(actual_ranks, actual_ranks, 'k--', label='Perfect prediction', alpha=0.5)
        plt.plot(actual_ranks, predicted_ranks, 'ro-', label='Model prediction', linewidth=2, markersize=8)
        plt.xlabel('Actual Rank')
        plt.ylabel('Predicted Rank (weighted)')
        plt.title('Rank Prediction Quality')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, 10))
        plt.yticks(range(1, 10))
        
        # Plot 2: Rank probability heatmap
        plt.subplot(1, 2, 2)
        heatmap_data = np.array([rank_data[r]['mean_probs'] for r in sorted(rank_data.keys())])
        plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        plt.colorbar(label='Probability')
        plt.xlabel('Predicted Rank')
        plt.ylabel('Actual Rank')
        plt.title('Rank Model Output Distribution')
        plt.xticks(range(9), [f'{i}D' for i in range(1, 10)])
        plt.yticks(range(len(rank_data)), [f'{r}D' for r in sorted(rank_data.keys())])
        
        plt.tight_layout()
        plt.savefig('diagnostic_plot.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved visualization to 'diagnostic_plot.png'")
    except Exception as e:
        print(f"\n⚠️  Could not create plot: {e}")
    
    return rank_data


def suggest_improvements(rank_data):
    """Suggest improvements based on analysis."""
    print("\n" + "=" * 80)
    print("SUGGESTIONS FOR IMPROVEMENT")
    print("=" * 80)
    
    # Check confidence levels
    avg_confidence = np.mean([d['confidence'] for d in rank_data.values()])
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    
    if avg_confidence < 0.3:
        print("  ⚠️  Low confidence in rank predictions")
        print("  → The rank model outputs may not be very discriminative")
        print("  → Try: Use more features beyond just rank model")
        print("  → Try: Ensemble with policy and value features")
    
    # Check if ranks are confused
    confusion_count = 0
    for rank, data in rank_data.items():
        if data['most_likely'] != rank:
            confusion_count += 1
            print(f"  ⚠️  Rank {rank}D confused with {data['most_likely']}D")
    
    if confusion_count > 3:
        print("\n  ❌ Many ranks are confused!")
        print("  → The rank model alone may not be sufficient")
        print("  → Try: Weight policy and strength features more heavily")
        print("  → Try: Use move-level features, not just aggregations")
    
    # Check variance
    weighted_preds = [d['weighted'] for d in rank_data.values()]
    variance = np.var(weighted_preds)
    print(f"\nVariance in predictions: {variance:.3f}")
    
    if variance < 1.0:
        print("  ⚠️  Low variance - predictions are too similar")
        print("  → Model may not distinguish ranks well")
        print("  → Try: Look at policy/value differences between ranks")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS:")
    print("=" * 80)
    print("1. Check diagnostic_plot.png to visualize rank separability")
    print("2. If correlation < 0.8, use more diverse features")
    print("3. If confidence < 0.4, weight strength/policy more")
    print("4. Consider using percentiles instead of just means")
    print("5. Try median aggregation (more robust to outliers)")
    print("=" * 80)


if __name__ == '__main__':
    import sys
    
    train_dir = sys.argv[1] if len(sys.argv) > 1 else 'train'
    
    rank_data = analyze_training_data(train_dir)
    suggest_improvements(rank_data)
    
    print("\n✓ Diagnostic complete!")