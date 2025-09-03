"""
Connect 4 CV Distortion Experiment - Data Analysis & Visualization
===============================================================

This script analyzes the results from your CV distortion experiment and generates
publication-ready graphs for your EE project report.

Usage: python analyze_results.py [csv_filename]
If no filename provided, it will use the most recent CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
import sys
from datetime import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ExperimentAnalyzer:
    def __init__(self, csv_file=None):
        if csv_file is None:
            # Find the most recent CSV file
            csv_files = glob.glob("cv_distortion_experiment_*.csv")
            if not csv_files:
                raise FileNotFoundError("No experiment CSV files found!")
            csv_file = max(csv_files, key=os.path.getctime)
        
        print(f"📊 Loading data from: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.csv_file = csv_file
        
        # Create output directory for plots
        self.output_dir = "experiment_plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"📈 Loaded {len(self.df)} game records")
        print(f"🎯 Trials: {self.df['trial_number'].nunique()}")
        print(f"📁 Plots will be saved to: {self.output_dir}/")
    
    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("\n🎨 Generating plots...")
        
        # 1. Main performance metrics vs accuracy
        self.plot_performance_vs_accuracy()
        
        # 2. Move optimality degradation
        self.plot_move_optimality()
        
        # 3. Illegal moves analysis
        self.plot_illegal_moves()
        
        # 4. Game length analysis
        self.plot_game_length()
        
        # 5. Starting player advantage
        self.plot_starting_player_effect()
        
        # 6. Combined dashboard
        self.plot_dashboard()
        
        print(f"\n✅ All plots saved to {self.output_dir}/")
        print("📋 Generated plots:")
        for file in sorted(os.listdir(self.output_dir)):
            if file.endswith('.png'):
                print(f"   • {file}")
    
    def plot_performance_vs_accuracy(self):
        """Plot win rates vs CV accuracy"""
        # Group by accuracy level
        summary = self.df.groupby('accuracy_percent').agg({
            'minimax_ai_wins': 'mean',
            'deterministic_ai_wins': 'mean',
            'illegal_moves': 'sum',
            'game_length': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Win rates
        ax1.plot(summary['accuracy_percent'], summary['minimax_ai_wins'] * 100, 
                'o-', linewidth=3, markersize=8, label='Minimax AI (Distorted)', color='red')
        ax1.plot(summary['accuracy_percent'], summary['deterministic_ai_wins'] * 100, 
                'o-', linewidth=3, markersize=8, label='Deterministic AI (Perfect)', color='blue')
        
        ax1.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax1.set_ylabel('Win Rate (%)', fontsize=12)
        ax1.set_title('AI Performance vs CV Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add trend line for minimax AI
        z = np.polyfit(summary['accuracy_percent'], summary['minimax_ai_wins'] * 100, 1)
        p = np.poly1d(z)
        ax1.plot(summary['accuracy_percent'], p(summary['accuracy_percent']), 
                "--", alpha=0.8, color='red', linewidth=2)
        
        # Illegal moves
        ax2.bar(summary['accuracy_percent'], summary['illegal_moves'], 
               alpha=0.7, color='orange', edgecolor='black', linewidth=1)
        ax2.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax2.set_ylabel('Total Illegal Moves', fontsize=12)
        ax2.set_title('Illegal Moves vs CV Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/1_performance_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_move_optimality(self):
        """Plot move optimality degradation"""
        # Filter out games where minimax didn't make any moves
        df_filtered = self.df[self.df['total_minimax_moves'] > 0].copy()
        
        if len(df_filtered) == 0:
            print("⚠️  No move optimality data available")
            return
        
        summary = df_filtered.groupby('accuracy_percent').agg({
            'move_optimality_percent': ['mean', 'std'],
            'optimal_moves': 'sum',
            'total_minimax_moves': 'sum'
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['accuracy_percent', 'optimality_mean', 'optimality_std', 
                          'total_optimal', 'total_moves']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Move optimality percentage
        ax1.errorbar(summary['accuracy_percent'], summary['optimality_mean'], 
                    yerr=summary['optimality_std'], fmt='o-', linewidth=3, 
                    markersize=8, capsize=5, color='green')
        
        ax1.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax1.set_ylabel('Move Optimality (%)', fontsize=12)
        ax1.set_title('Move Quality Degradation vs CV Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add trend line
        z = np.polyfit(summary['accuracy_percent'], summary['optimality_mean'], 1)
        p = np.poly1d(z)
        ax1.plot(summary['accuracy_percent'], p(summary['accuracy_percent']), 
                "--", alpha=0.8, color='green', linewidth=2)
        
        # Optimal moves ratio
        optimal_ratio = summary['total_optimal'] / summary['total_moves'] * 100
        ax2.bar(summary['accuracy_percent'], optimal_ratio, alpha=0.7, 
               color='lightgreen', edgecolor='black', linewidth=1)
        ax2.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax2.set_ylabel('Optimal Moves (%)', fontsize=12)
        ax2.set_title('Overall Optimal Move Percentage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/2_move_optimality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_illegal_moves(self):
        """Analyze illegal move patterns"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Illegal moves by accuracy
        illegal_by_accuracy = self.df.groupby('accuracy_percent')['illegal_moves'].sum().reset_index()
        
        ax1.plot(illegal_by_accuracy['accuracy_percent'], illegal_by_accuracy['illegal_moves'], 
                'ro-', linewidth=3, markersize=8)
        ax1.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax1.set_ylabel('Total Illegal Moves', fontsize=12)
        ax1.set_title('Illegal Moves vs CV Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Games with illegal moves percentage
        games_with_illegal = self.df.groupby('accuracy_percent').apply(
            lambda x: (x['illegal_moves'] > 0).mean() * 100
        ).reset_index(name='pct_games_with_illegal')
        
        ax2.bar(games_with_illegal['accuracy_percent'], games_with_illegal['pct_games_with_illegal'], 
               alpha=0.7, color='red', edgecolor='black', linewidth=1)
        ax2.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax2.set_ylabel('Games with Illegal Moves (%)', fontsize=12)
        ax2.set_title('Percentage of Games with Illegal Moves', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/3_illegal_moves_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_game_length(self):
        """Analyze game length patterns"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average game length by accuracy
        length_summary = self.df.groupby('accuracy_percent').agg({
            'game_length': ['mean', 'std']
        }).reset_index()
        length_summary.columns = ['accuracy_percent', 'mean_length', 'std_length']
        
        ax1.errorbar(length_summary['accuracy_percent'], length_summary['mean_length'], 
                    yerr=length_summary['std_length'], fmt='o-', linewidth=3, 
                    markersize=8, capsize=5, color='purple')
        
        ax1.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax1.set_ylabel('Average Game Length (moves)', fontsize=12)
        ax1.set_title('Game Length vs CV Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Game length distribution
        accuracy_levels = sorted(self.df['accuracy_percent'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(accuracy_levels)))
        
        for i, acc in enumerate(accuracy_levels):
            data = self.df[self.df['accuracy_percent'] == acc]['game_length']
            ax2.hist(data, alpha=0.6, label=f'{acc:.0f}% accuracy', 
                    color=colors[i], bins=15, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Game Length (moves)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Game Length Distribution by Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/4_game_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_starting_player_effect(self):
        """Analyze starting player advantage"""
        if 'starting_player' not in self.df.columns:
            print("⚠️  Starting player data not available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Win rates by starting player
        starting_analysis = self.df.groupby(['accuracy_percent', 'starting_player']).agg({
            'minimax_ai_wins': 'mean'
        }).reset_index()
        
        # Pivot for easier plotting
        pivot_data = starting_analysis.pivot(index='accuracy_percent', 
                                           columns='starting_player', 
                                           values='minimax_ai_wins')
        
        ax1.plot(pivot_data.index, pivot_data['Minimax'] * 100, 'o-', 
                linewidth=3, markersize=8, label='Minimax starts', color='red')
        ax1.plot(pivot_data.index, pivot_data['Deterministic'] * 100, 's-', 
                linewidth=3, markersize=8, label='Deterministic starts', color='blue')
        
        ax1.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax1.set_ylabel('Minimax AI Win Rate (%)', fontsize=12)
        ax1.set_title('Starting Player Advantage Effect', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Starting player balance check
        start_counts = self.df.groupby(['accuracy_percent', 'starting_player']).size().unstack()
        ax2.bar(start_counts.index - 1, start_counts['Deterministic'], 
               width=2, alpha=0.7, label='Deterministic starts', color='blue')
        ax2.bar(start_counts.index + 1, start_counts['Minimax'], 
               width=2, alpha=0.7, label='Minimax starts', color='red')
        
        ax2.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax2.set_ylabel('Number of Games', fontsize=12)
        ax2.set_title('Starting Player Balance', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/5_starting_player_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dashboard(self):
        """Create a comprehensive dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Win rates (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        summary = self.df.groupby('accuracy_percent').agg({
            'minimax_ai_wins': 'mean',
            'deterministic_ai_wins': 'mean'
        }).reset_index()
        
        ax1.plot(summary['accuracy_percent'], summary['minimax_ai_wins'] * 100, 
                'o-', linewidth=2, label='Minimax AI', color='red')
        ax1.plot(summary['accuracy_percent'], summary['deterministic_ai_wins'] * 100, 
                'o-', linewidth=2, label='Deterministic AI', color='blue')
        ax1.set_title('Win Rates vs CV Accuracy', fontweight='bold')
        ax1.set_xlabel('CV Accuracy (%)')
        ax1.set_ylabel('Win Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Move optimality (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'move_optimality_percent' in self.df.columns:
            df_filtered = self.df[self.df['total_minimax_moves'] > 0]
            opt_summary = df_filtered.groupby('accuracy_percent')['move_optimality_percent'].mean()
            ax2.plot(opt_summary.index, opt_summary.values, 'o-', 
                    linewidth=2, color='green')
        ax2.set_title('Move Optimality', fontweight='bold')
        ax2.set_xlabel('CV Accuracy (%)')
        ax2.set_ylabel('Optimality (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Illegal moves (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        illegal_summary = self.df.groupby('accuracy_percent')['illegal_moves'].sum()
        ax3.bar(illegal_summary.index, illegal_summary.values, 
               alpha=0.7, color='orange')
        ax3.set_title('Illegal Moves', fontweight='bold')
        ax3.set_xlabel('CV Accuracy (%)')
        ax3.set_ylabel('Total Illegal Moves')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Game length (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        length_summary = self.df.groupby('accuracy_percent')['game_length'].mean()
        ax4.plot(length_summary.index, length_summary.values, 'o-', 
                linewidth=2, color='purple')
        ax4.set_title('Average Game Length', fontweight='bold')
        ax4.set_xlabel('CV Accuracy (%)')
        ax4.set_ylabel('Moves')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance correlation (middle center)
        ax5 = fig.add_subplot(gs[1, 1])
        minimax_wins = self.df.groupby('accuracy_percent')['minimax_ai_wins'].mean() * 100
        if 'move_optimality_percent' in self.df.columns:
            df_filtered = self.df[self.df['total_minimax_moves'] > 0]
            optimality = df_filtered.groupby('accuracy_percent')['move_optimality_percent'].mean()
            ax5.scatter(optimality, minimax_wins, s=100, alpha=0.7, color='red')
            
            # Add trend line
            if len(optimality) > 1:
                z = np.polyfit(optimality, minimax_wins, 1)
                p = np.poly1d(z)
                ax5.plot(optimality, p(optimality), "--", alpha=0.8, color='red')
        
        ax5.set_title('Win Rate vs Move Quality', fontweight='bold')
        ax5.set_xlabel('Move Optimality (%)')
        ax5.set_ylabel('Win Rate (%)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Experiment summary (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        total_games = len(self.df)
        total_trials = self.df['trial_number'].nunique()
        avg_minimax_wins = self.df['minimax_ai_wins'].mean() * 100
        total_illegal = self.df['illegal_moves'].sum()
        
        summary_text = f"""EXPERIMENT SUMMARY
        
Total Games: {total_games}
Trials: {total_trials}
Accuracy Levels: {sorted(self.df['accuracy_percent'].unique())}

Avg Minimax Win Rate: {avg_minimax_wins:.1f}%
Total Illegal Moves: {total_illegal}

Data File: {os.path.basename(self.csv_file)}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # 7. Accuracy trend (bottom span)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Combined metrics
        metrics_summary = self.df.groupby('accuracy_percent').agg({
            'minimax_ai_wins': 'mean',
            'illegal_moves': lambda x: x.sum() / len(x) * 10,  # Scale for visibility
        }).reset_index()
        
        if 'move_optimality_percent' in self.df.columns:
            df_filtered = self.df[self.df['total_minimax_moves'] > 0]
            opt_by_acc = df_filtered.groupby('accuracy_percent')['move_optimality_percent'].mean() / 100
            metrics_summary = metrics_summary.merge(
                opt_by_acc.reset_index().rename(columns={'move_optimality_percent': 'optimality'}), 
                on='accuracy_percent', how='left'
            )
            ax7.plot(metrics_summary['accuracy_percent'], metrics_summary['optimality'], 
                    'o-', linewidth=3, label='Move Optimality', color='green')
        
        ax7.plot(metrics_summary['accuracy_percent'], metrics_summary['minimax_ai_wins'], 
                'o-', linewidth=3, label='Win Rate', color='red')
        ax7.plot(metrics_summary['accuracy_percent'], metrics_summary['illegal_moves'], 
                's-', linewidth=3, label='Illegal Moves (×10)', color='orange')
        
        ax7.set_title('CV Distortion Impact on AI Performance', fontsize=16, fontweight='bold')
        ax7.set_xlabel('Computer Vision Accuracy (%)', fontsize=12)
        ax7.set_ylabel('Performance Metrics (normalized)', fontsize=12)
        ax7.legend(fontsize=11)
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(0, 1.1)
        
        plt.suptitle('Connect 4 CV Distortion Experiment - Complete Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(f'{self.output_dir}/6_complete_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        report = f"""
CONNECT 4 CV DISTORTION EXPERIMENT - ANALYSIS REPORT
==================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data File: {self.csv_file}

EXPERIMENT OVERVIEW:
- Total Games Played: {len(self.df)}
- Number of Trials: {self.df['trial_number'].nunique()}
- Accuracy Levels Tested: {sorted(self.df['accuracy_percent'].unique())}

PERFORMANCE SUMMARY BY ACCURACY LEVEL:
"""
        
        for acc in sorted(self.df['accuracy_percent'].unique()):
            subset = self.df[self.df['accuracy_percent'] == acc]
            minimax_wins = subset['minimax_ai_wins'].mean() * 100
            det_wins = subset['deterministic_ai_wins'].mean() * 100
            illegal_moves = subset['illegal_moves'].sum()
            avg_length = subset['game_length'].mean()
            
            report += f"""
{acc:.0f}% CV Accuracy:
  • Minimax AI Win Rate: {minimax_wins:.1f}%
  • Deterministic AI Win Rate: {det_wins:.1f}%
  • Total Illegal Moves: {illegal_moves}
  • Average Game Length: {avg_length:.1f} moves"""
            
            if 'move_optimality_percent' in subset.columns:
                filtered_subset = subset[subset['total_minimax_moves'] > 0]
                if len(filtered_subset) > 0:
                    avg_optimality = filtered_subset['move_optimality_percent'].mean()
                    report += f"""
  • Average Move Optimality: {avg_optimality:.1f}%"""
        
        report += f"""

KEY FINDINGS:
- Performance degradation as CV accuracy decreases
- Correlation between CV errors and illegal moves
- Impact of distortion on move quality
"""
        
        # Save report
        with open(f'{self.output_dir}/experiment_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        return report


def main():
    """Main execution function"""
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        analyzer = ExperimentAnalyzer(csv_file)
        analyzer.generate_all_plots()
        analyzer.generate_summary_report()
        
        print(f"\n🎉 Analysis complete!")
        print(f"📊 Check the '{analyzer.output_dir}' folder for all generated plots and reports.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you have run the experiment first to generate CSV data.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
