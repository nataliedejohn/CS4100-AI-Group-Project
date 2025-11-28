"""Optimized DataLoader based on feature research.

Uses the best 5 features identified through analysis for maximum DQN prediction accuracy.
5 features × 50 players = 250 inputs
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
# from sklearn.preprocessing import MinMaxScaler (commented out for now because sklearn package was being annoying lol)

from DQN_train import train_dqn, test_dqn

dataloader = None  # redelcared at top-level for convenience in other files
class OptimizedPlayerDataset(Dataset):
    """Dataset with optimized features for DQN prediction accuracy.
    
    Uses features identified through research as most predictive:
    1. prev_hr_rate_season - Season HR rate (best correlation: 0.148)
    2. prev_avg_ev_7 - Exit velocity (correlation: 0.074, importance: 0.203)
    3. prev_hr_rate_28 - 28-day HR rate (correlation: 0.079)
    4. prev_pa_roll_28 - 28-day PA (importance: 0.344)
    5. prev_avg_la_7 - Launch angle (importance: 0.165)
    """
    
    def __init__(self, data_path: Path, normalize: bool = True):
        self.df = pd.read_csv(data_path, parse_dates=['week_start', 'week_end'])
        
        top_players = self.df.groupby('player_id')['label_hr'].sum().sort_values(ascending=False).head(50).index.tolist()
        self.top_players = top_players
        
        self.feature_cols = [
            'prev_hr_rate_season',
            'prev_avg_ev_7',
            'prev_hr_rate_28',
            'prev_pa_roll_28',
            'prev_avg_la_7'
        ]
        
        self.df = self.df[self.df['player_id'].isin(self.top_players)].copy()
        self.weeks = sorted(self.df['week_id'].unique())
        
        self.normalize = normalize
        self.scaler = None
        
        if normalize:
            self._fit_normalizer()
    
    def _fit_normalizer(self):
        """Fit normalizer on all feature data."""
        all_values = []
        for col in self.feature_cols:
            values = self.df[col].fillna(0).values
            all_values.extend(values.tolist())
        
        self.feature_ranges = {}
        for col in self.feature_cols:
            values = self.df[col].fillna(0).values
            min_val = values.min()
            max_val = values.max()
            self.feature_ranges[col] = {'min': min_val, 'max': max_val}
            
            if max_val == min_val:
                self.feature_ranges[col] = {'min': 0, 'max': 1}
    
    def __len__(self) -> int:
        return len(self.weeks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return 250 features (5 × 50 players) and rewards.
        
        Features are normalized for optimal DQN training.
        
        Returns:
            state: (250,) tensor - 5 features × 50 players, normalized
            rewards: (50,) tensor - reward for each player
        """
        week_id = self.weeks[idx]
        week_data = self.df[self.df['week_id'] == week_id].copy()
        
        state = np.zeros(250)
        rewards = np.zeros(50)
        
        all_features = []
        for i, player_id in enumerate(self.top_players):
            player_data = week_data[week_data['player_id'] == player_id]
            player_features = []
            
            if not player_data.empty:
                row = player_data.iloc[0]
                
                for col in self.feature_cols:
                    val = row.get(col, 0.0)
                    if pd.isna(val):
                        val = 0.0
                    player_features.append(float(val))
                
                rewards[i] = float(row.get('reward', 0.0))
            else:
                player_features = [0.0] * len(self.feature_cols)
            
            all_features.append(player_features)
        
        if self.normalize and self.feature_ranges:
            normalized_features = []
            for player_feat in all_features:
                norm_player_feat = []
                for j, feat_val in enumerate(player_feat):
                    col = self.feature_cols[j]
                    feat_range = self.feature_ranges[col]
                    min_val, max_val = feat_range['min'], feat_range['max']
                    
                    if max_val > min_val:
                        norm_val = (feat_val - min_val) / (max_val - min_val)
                    else:
                        norm_val = 0.0
                    norm_player_feat.append(norm_val)
                normalized_features.append(norm_player_feat)
            all_features = normalized_features
        
        for i, player_features in enumerate(all_features):
            start_idx = i * 5
            for j, feature_val in enumerate(player_features):
                state[start_idx + j] = feature_val
        
        return torch.FloatTensor(state), torch.FloatTensor(rewards)


if __name__ == "__main__":
    print("="*70)
    print("OPTIMIZED DATALOADER - Based on Feature Research")
    print("="*70)
    
    data_path = Path("basebal_data/cleaned_output/player_week_features_clean.csv")
    dataset = OptimizedPlayerDataset(data_path, normalize=True)
    
    print(f"\nDataset Info:")
    print(f"  Size: {len(dataset)} weeks")
    print(f"  Features: 5 per player × 50 players = 250 total")
    print(f"  Normalization: Enabled (MinMaxScaler 0-1)")
    
    print(f"\nSelected Features (best for prediction):")
    for i, feat in enumerate(dataset.feature_cols, 1):
        print(f"  {i}. {feat}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    print(f"\nDataLoader created: {len(dataloader)} batches")
    
    for state, rewards in dataloader:
        print(f"\nFirst batch:")
        print(f"  State shape: {state.shape}    # (batch_size=1, 250 features)")
        print(f"  Rewards shape: {rewards.shape}  # (batch_size=1, 50 players)")
        print(f"  State range: [{state.min():.4f}, {state.max():.4f}] (normalized 0-1)")
        print(f"  First player features (5): {state[0][:5].tolist()}")
        break
    
    print("\n" + "="*70)
    print("Features selected based on:")
    print("  1. Correlation with target (reward)")
    print("  2. Feature importance (variance + predictive power)")
    print("  3. Low missing data percentage")
    print("="*70)

    train_dqn(dataloader, num_episodes=1000, decay=0.09)
    test_dqn(dataloader, num_episodes=1000, decay=0.09)
