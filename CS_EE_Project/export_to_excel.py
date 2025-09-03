import pandas as pd
import numpy as np
import glob
import os
import sys

def export_for_excel(csv_file=None):
    if csv_file is None:
        csv_files = glob.glob("cv_distortion_experiment_*.csv")
        if not csv_files:
            print("No experiment CSV files found!")
            return
        csv_file = max(csv_files, key=os.path.getctime)
    
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Create Excel filename
    base_name = os.path.splitext(csv_file)[0]
    excel_file = f"{base_name}_excel_ready.xlsx"
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        # Sheet 1: Raw Data (cleaned up)
        raw_data = df[[
            'trial_number', 'accuracy_percent', 'board_index', 'starting_player',
            'minimax_ai_wins', 'deterministic_ai_wins', 'illegal_moves', 
            'game_length', 'move_optimality_percent', 'avg_move_time'
        ]].copy()
        
        raw_data.columns = [
            'Trial', 'CV_Accuracy_%', 'Board_ID', 'Starting_Player',
            'Minimax_Win', 'Deterministic_Win', 'Illegal_Moves',
            'Game_Length', 'Move_Optimality_%', 'Avg_Move_Time_sec'
        ]
        
        raw_data.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        # Sheet 2: Summary by Accuracy Level
        summary = df.groupby('accuracy_percent').agg({
            'minimax_ai_wins': ['sum', 'mean', 'count'],
            'deterministic_ai_wins': ['sum', 'mean'],
            'illegal_moves': 'sum',
            'game_length': 'mean',
            'move_optimality_percent': 'mean',
            'avg_move_time': 'mean'
        }).round(2)
        
        # Flatten column names
        summary.columns = [
            'Minimax_Total_Wins', 'Minimax_Win_Rate', 'Total_Games',
            'Deterministic_Total_Wins', 'Deterministic_Win_Rate',
            'Total_Illegal_Moves', 'Avg_Game_Length', 
            'Avg_Move_Optimality_%', 'Avg_Move_Time_sec'
        ]
        
        summary.reset_index(inplace=True)
        summary.columns = ['CV_Accuracy_%'] + list(summary.columns[1:])
        
        # Convert win rates to percentages
        summary['Minimax_Win_Rate_%'] = (summary['Minimax_Win_Rate'] * 100).round(1)
        summary['Deterministic_Win_Rate_%'] = (summary['Deterministic_Win_Rate'] * 100).round(1)
        
        # Reorder columns for clarity
        summary_clean = summary[[
            'CV_Accuracy_%', 'Total_Games',
            'Minimax_Total_Wins', 'Minimax_Win_Rate_%',
            'Deterministic_Total_Wins', 'Deterministic_Win_Rate_%',
            'Total_Illegal_Moves', 'Avg_Game_Length',
            'Avg_Move_Optimality_%', 'Avg_Move_Time_sec'
        ]]
        
        summary_clean.to_excel(writer, sheet_name='Summary_by_Accuracy', index=False)
        
        # Sheet 3: Starting Player Analysis
        if 'starting_player' in df.columns:
            starting_analysis = df.groupby(['accuracy_percent', 'starting_player']).agg({
                'minimax_ai_wins': 'mean',
                'game_length': 'mean',
                'illegal_moves': 'sum',
                'move_optimality_percent': 'mean'
            }).round(2)
            
            starting_analysis.columns = [
                'Minimax_Win_Rate', 'Avg_Game_Length', 
                'Total_Illegal_Moves', 'Avg_Move_Optimality_%'
            ]
            
            starting_analysis['Minimax_Win_Rate_%'] = (starting_analysis['Minimax_Win_Rate'] * 100).round(1)
            starting_analysis = starting_analysis.drop('Minimax_Win_Rate', axis=1)
            
            starting_analysis.reset_index(inplace=True)
            starting_analysis.columns = [
                'CV_Accuracy_%', 'Starting_Player', 'Minimax_Win_Rate_%',
                'Avg_Game_Length', 'Total_Illegal_Moves', 'Avg_Move_Optimality_%'
            ]
            
            starting_analysis.to_excel(writer, sheet_name='Starting_Player_Analysis', index=False)
        
        # Sheet 4: Move Optimality Details
        if 'move_optimality_percent' in df.columns:
            optimality_data = df[df['total_minimax_moves'] > 0].groupby('accuracy_percent').agg({
                'optimal_moves': 'sum',
                'total_minimax_moves': 'sum',
                'move_optimality_percent': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            optimality_data.columns = [
                'Total_Optimal_Moves', 'Total_Minimax_Moves',
                'Avg_Optimality_%', 'Std_Optimality_%', 'Min_Optimality_%', 'Max_Optimality_%'
            ]
            
            optimality_data['Overall_Optimality_Rate_%'] = (
                optimality_data['Total_Optimal_Moves'] / 
                optimality_data['Total_Minimax_Moves'] * 100
            ).round(1)
            
            optimality_data.reset_index(inplace=True)
            optimality_data.columns = ['CV_Accuracy_%'] + list(optimality_data.columns[1:])
            
            optimality_data.to_excel(writer, sheet_name='Move_Optimality_Details', index=False)
        
        # Sheet 5: Chart Data (Pre-formatted for Excel charts)
        chart_data = summary_clean[['CV_Accuracy_%', 'Minimax_Win_Rate_%', 'Deterministic_Win_Rate_%']].copy()
        chart_data.columns = ['CV_Accuracy', 'Minimax_AI', 'Deterministic_AI']
        
        # Add illegal moves data
        illegal_data = summary_clean[['CV_Accuracy_%', 'Total_Illegal_Moves']].copy()
        illegal_data.columns = ['CV_Accuracy', 'Illegal_Moves']
        
        # Add optimality data
        if 'move_optimality_percent' in df.columns:
            opt_data = optimality_data[['CV_Accuracy_%', 'Overall_Optimality_Rate_%']].copy()
            opt_data.columns = ['CV_Accuracy', 'Move_Optimality']
            
            chart_combined = chart_data.merge(illegal_data, on='CV_Accuracy').merge(opt_data, on='CV_Accuracy')
        else:
            chart_combined = chart_data.merge(illegal_data, on='CV_Accuracy')
        
        chart_combined.to_excel(writer, sheet_name='Chart_Data', index=False)
        
        # Sheet 6: Experiment Info
        experiment_info = pd.DataFrame([
            ['Total Games', len(df)],
            ['Trials', df['trial_number'].nunique()],
            ['Accuracy Levels', ', '.join(map(str, sorted(df['accuracy_percent'].unique())))],
            ['Games per Trial', len(df) // df['trial_number'].nunique()],
            ['Average Game Length', df['game_length'].mean().round(1)],
            ['Total Illegal Moves', df['illegal_moves'].sum()],
            ['Experiment Duration (est)', f"{df['avg_move_time'].sum()/60:.1f} minutes"],
            ['Data Generated', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        ], columns=['Metric', 'Value'])
        
        experiment_info.to_excel(writer, sheet_name='Experiment_Info', index=False)
    
    print(f"✅ Excel file created: {excel_file}")
    print("\n📊 Excel sheets created:")
    print("   1. Raw_Data - Individual game results")
    print("   2. Summary_by_Accuracy - Main results table")
    print("   3. Starting_Player_Analysis - First move advantage")
    print("   4. Move_Optimality_Details - AI decision quality")
    print("   5. Chart_Data - Ready for Excel charts")
    print("   6. Experiment_Info - Experiment metadata")
    
    print(f"\n📈 Recommended Excel charts:")
    print("   • Line chart: CV_Accuracy vs Win Rates (Sheet 5: Chart_Data)")
    print("   • Bar chart: CV_Accuracy vs Illegal_Moves (Sheet 5: Chart_Data)")
    print("   • Line chart: CV_Accuracy vs Move_Optimality (Sheet 5: Chart_Data)")
    print("   • Comparison chart: Starting player effects (Sheet 3)")
    
    return excel_file

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        excel_file = export_for_excel(csv_file)
        print(f"\n🎉 Ready for Excel! Open: {excel_file}")
        
    except FileNotFoundError:
        print("❌ No experiment CSV files found!")
        print("Run your experiment first: python final_cv_experiment.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
