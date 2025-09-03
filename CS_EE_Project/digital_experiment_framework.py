"""
Digital Connect 4 Experiment Framework
=====================================

This framework tests the impact of computer vision distortions on AI decision-making
by simulating CV errors (blur, misclassification) on digital board states.

Experimental Design:
1. Create 30+ ground truth board states (baseline perfect information)
2. Apply controlled distortions to simulate CV errors
3. Compare AI performance: Perfect Info AI vs Distorted Info AI
4. Measure: move quality, illegal moves, win rate, decision time

Author: Connect4-CV-AI Project
"""

import numpy as np
import random
import json
import csv
import time
import math
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import copy

# Import your existing modules
import game
import ai
from enhanced_deterministic_ai import create_deterministic_ai, StrongDeterministicAI

class BoardDistortor:
    """Applies controlled distortions to simulate CV errors"""
    
    def __init__(self):
        self.distortion_log = []
    
    def apply_blur(self, board: np.ndarray, blur_rate: float) -> np.ndarray:
        """
        Simulate blur by randomly removing pieces (turn them to 0)
        blur_rate: 0.0 to 1.0 (0% to 100% of pieces may disappear)
        """
        distorted = board.copy()
        rows, cols = board.shape
        
        blur_count = 0
        for r in range(rows):
            for c in range(cols):
                if board[r][c] != 0 and random.random() < blur_rate:
                    distorted[r][c] = 0  # Piece disappears due to blur
                    blur_count += 1
                    
        self.distortion_log.append({
            'type': 'blur',
            'rate': blur_rate,
            'affected_pieces': blur_count
        })
        return distorted
    
    def apply_misclassification(self, board: np.ndarray, misclass_rate: float) -> np.ndarray:
        """
        Simulate color misclassification by swapping piece colors
        misclass_rate: 0.0 to 1.0 (0% to 100% of pieces may be misclassified)
        """
        distorted = board.copy()
        rows, cols = board.shape
        
        misclass_count = 0
        for r in range(rows):
            for c in range(cols):
                if board[r][c] != 0 and random.random() < misclass_rate:
                    # Swap colors: 1 <-> 2
                    distorted[r][c] = 2 if board[r][c] == 1 else 1
                    misclass_count += 1
                    
        self.distortion_log.append({
            'type': 'misclassification', 
            'rate': misclass_rate,
            'affected_pieces': misclass_count
        })
        return distorted
    
    def apply_combined_distortion(self, board: np.ndarray, blur_rate: float, misclass_rate: float) -> np.ndarray:
        """Apply both blur and misclassification distortions"""
        # Apply blur first
        distorted = self.apply_blur(board, blur_rate)
        # Then apply misclassification to the blurred board
        distorted = self.apply_misclassification(distorted, misclass_rate)
        return distorted
    
    def clear_log(self):
        """Clear distortion log for new experiment"""
        self.distortion_log = []

class GroundTruthGenerator:
    """Generate representative Connect 4 board states for testing"""
    
    @staticmethod
    def create_early_game_boards() -> List[np.ndarray]:
        """Create early game scenarios (few pieces, 1-10 moves)"""
        boards = []
        
        # Empty board
        boards.append(np.zeros((6, 7)))
        
        # Single piece scenarios
        for col in [0, 3, 6]:  # Left, center, right
            board = np.zeros((6, 7))
            board[0][col] = 1  # Player 1 starts
            boards.append(board.copy())
        
        # Two piece scenarios
        board = np.zeros((6, 7))
        board[0][3] = 1  # Player 1 center
        board[0][2] = 2  # Player 2 adjacent
        boards.append(board.copy())
        
        board = np.zeros((6, 7))
        board[0][3] = 1  # Player 1 center
        board[0][4] = 2  # Player 2 adjacent other side
        boards.append(board.copy())
        
        # Multiple pieces, early stage
        board = np.zeros((6, 7))
        board[0][3] = 1
        board[1][3] = 2
        board[0][2] = 1
        board[0][4] = 2
        boards.append(board.copy())
        
        return boards
    
    @staticmethod
    def create_mid_game_boards() -> List[np.ndarray]:
        """Create mid-game scenarios (moderate pieces, strategic positions)"""
        boards = []
        
        # Typical mid-game with some stacking
        board = np.zeros((6, 7))
        board[0][3] = 1
        board[1][3] = 2
        board[2][3] = 1
        board[0][2] = 2
        board[1][2] = 1
        board[0][4] = 1
        board[1][4] = 2
        board[0][1] = 2
        board[0][5] = 1
        boards.append(board.copy())
        
        # Different column distributions
        board = np.zeros((6, 7))
        # Left side heavy
        board[0][0] = 1
        board[1][0] = 2
        board[0][1] = 2
        board[1][1] = 1
        board[2][1] = 2
        board[0][2] = 1
        # Sparse right side
        board[0][5] = 2
        board[0][6] = 1
        boards.append(board.copy())
        
        # Center control scenario
        board = np.zeros((6, 7))
        board[0][3] = 1
        board[1][3] = 2
        board[2][3] = 1
        board[3][3] = 2
        board[0][2] = 2
        board[0][4] = 1
        board[1][2] = 1
        board[1][4] = 2
        boards.append(board.copy())
        
        return boards
    
    @staticmethod
    def create_near_win_boards() -> List[np.ndarray]:
        """Create near-win scenarios (critical decision points)"""
        boards = []
        
        # Horizontal threat - 3 in a row, need to block
        board = np.zeros((6, 7))
        board[0][1] = 1
        board[0][2] = 1  
        board[0][3] = 1
        # Player 2 must block at [0][0] or [0][4]
        board[0][5] = 2  # Some other pieces
        board[0][6] = 2
        boards.append(board.copy())
        
        # Vertical threat - 3 stacked, need to block
        board = np.zeros((6, 7))
        board[0][3] = 1
        board[1][3] = 1
        board[2][3] = 1
        # Player 2 must block at [3][3]
        board[0][2] = 2
        board[0][4] = 2
        boards.append(board.copy())
        
        # Diagonal threat
        board = np.zeros((6, 7))
        board[0][0] = 1
        board[1][1] = 1
        board[2][2] = 1
        # Player 2 should block at [3][3] if possible
        board[0][1] = 2  # Stack to make [1][1] possible
        board[1][2] = 2  # Stack to make [2][2] possible  
        board[2][3] = 2  # Stack to make [3][3] possible
        boards.append(board.copy())
        
        # Winning opportunity - player can win immediately
        board = np.zeros((6, 7))
        board[0][1] = 2
        board[0][2] = 2
        board[0][3] = 2
        # Player 2 can win at [0][0] or [0][4]
        board[1][1] = 1  # Some blocking pieces
        board[1][3] = 1
        boards.append(board.copy())
        
        return boards
    
    @staticmethod
    def create_complex_boards() -> List[np.ndarray]:
        """Create complex strategic scenarios"""
        boards = []
        
        # Multiple threats scenario
        board = np.zeros((6, 7))
        board[0][0] = 1
        board[0][1] = 1
        board[0][2] = 1  # Horizontal threat
        board[1][0] = 2
        board[2][0] = 2  # Vertical threat for P2
        board[0][4] = 2
        board[0][5] = 1
        board[0][6] = 1  # Another horizontal setup
        boards.append(board.copy())
        
        # Full column scenarios
        board = np.zeros((6, 7))
        for r in range(6):  # Fill leftmost column
            board[r][0] = 1 if r % 2 == 0 else 2
        # Sparse other columns
        board[0][3] = 1
        board[1][3] = 2
        board[0][6] = 2
        boards.append(board.copy())
        
        # Endgame scenario
        board = np.zeros((6, 7))
        # Fill most of the board randomly but strategically
        positions = [(0,1), (0,2), (0,3), (0,4), (0,5),
                    (1,1), (1,2), (1,3), (1,4), (1,5),
                    (2,2), (2,3), (2,4),
                    (3,3)]
        
        for i, (r, c) in enumerate(positions):
            board[r][c] = 1 if i % 2 == 0 else 2
            
        boards.append(board.copy())
        
        return boards

    @classmethod
    def generate_full_dataset(cls, save_to_file: Optional[str] = None) -> List[np.ndarray]:
        """Generate complete dataset of ground truth boards"""
        all_boards = []
        
        all_boards.extend(cls.create_early_game_boards())
        all_boards.extend(cls.create_mid_game_boards())  
        all_boards.extend(cls.create_near_win_boards())
        all_boards.extend(cls.create_complex_boards())
        
        # Add some random valid boards
        for _ in range(5):
            board = cls._create_random_valid_board()
            all_boards.append(board)
        
        print(f"Generated {len(all_boards)} ground truth board states")
        
        if save_to_file:
            cls._save_boards_to_file(all_boards, save_to_file)
        
        return all_boards
    
    @staticmethod
    def _create_random_valid_board() -> np.ndarray:
        """Create a random but valid Connect 4 board state"""
        board = np.zeros((6, 7))
        num_moves = random.randint(8, 25)  # Random number of moves
        
        for move in range(num_moves):
            valid_cols = [c for c in range(7) if board[5][c] == 0]  # Top row empty
            if not valid_cols:
                break
                
            col = random.choice(valid_cols)
            # Find bottom-most empty row in this column
            for row in range(6):
                if board[row][col] == 0:
                    board[row][col] = 1 if move % 2 == 0 else 2
                    break
        
        return board
    
    @staticmethod
    def _save_boards_to_file(boards: List[np.ndarray], filename: str):
        """Save boards to JSON file"""
        boards_list = [board.tolist() for board in boards]
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_boards': len(boards),
                'boards': boards_list
            }, f, indent=2)
        print(f"Saved {len(boards)} boards to {filename}")

class ExperimentRunner:
    """Main experiment runner for testing AI performance under CV distortions"""
    
    def __init__(self):
        self.ground_truth_boards = []
        self.distortor = BoardDistortor()
        self.results = []
        
    def load_ground_truth_boards(self, boards: List[np.ndarray]):
        """Load the ground truth dataset"""
        self.ground_truth_boards = boards
        print(f"Loaded {len(boards)} ground truth boards")
    
    def run_single_game_experiment(self, perfect_board: np.ndarray, distorted_board: np.ndarray, 
                                 experiment_id: str) -> Dict:
        """
        Run a single game: Perfect Info AI vs Distorted Info AI
        Returns detailed metrics about the game
        """
        
        # Create game boards for both AIs
        perfect_game_board = perfect_board.copy()
        distorted_game_board = distorted_board.copy() 
        
        # Initialize AIs
        perfect_ai = ai  # Using your minimax AI
        deterministic_opponent = create_deterministic_ai(strength="strong", piece=1)
        deterministic_opponent.reset()
        
        # Game metrics
        game_metrics = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'moves_made': 0,
            'illegal_moves_attempted': 0,
            'perfect_ai_wins': 0,
            'distorted_ai_wins': 0,
            'draws': 0,
            'move_times': [],
            'move_quality_scores': [],
            'final_board_state': None,
            'game_length': 0
        }
        
        # Play the game
        current_board = perfect_board.copy()  # Start with perfect board
        turn = 0  # 0 = Perfect AI (minimax), 1 = Deterministic AI
        moves_made = 0
        max_moves = 42  # Maximum possible moves in Connect 4
        
        while moves_made < max_moves:
            if turn == 0:  # Perfect AI (Minimax) turn
                start_time = time.time()
                
                try:
                    # Perfect AI sees perfect board
                    result = perfect_ai.minimax(current_board, 4, -math.inf, math.inf, True)
                    col, score = result if result[0] is not None else (None, 0)
                    
                    move_time = time.time() - start_time
                    game_metrics['move_times'].append(move_time)
                    game_metrics['move_quality_scores'].append(score)
                    
                    if col is not None and game.is_valid_location(current_board, col):
                        row = game.for_next_open_row(current_board, col)
                        if row is not None:
                            game.drop_piece(current_board, row, col, 2)  # Perfect AI is player 2
                            moves_made += 1
                            
                            if game.win_checks(current_board, 2):
                                game_metrics['perfect_ai_wins'] = 1
                                break
                        else:
                            game_metrics['illegal_moves_attempted'] += 1
                    else:
                        game_metrics['illegal_moves_attempted'] += 1
                        # Fallback: pick first valid move
                        valid_locations = perfect_ai.get_valid_locations(current_board)
                        if valid_locations:
                            col = valid_locations[0]
                            row = game.for_next_open_row(current_board, col)
                            if row is not None:
                                game.drop_piece(current_board, row, col, 2)
                                moves_made += 1
                        
                except Exception as e:
                    print(f"Perfect AI error: {e}")
                    game_metrics['illegal_moves_attempted'] += 1
                    
                turn = 1
                
            else:  # Deterministic AI turn
                try:
                    # Deterministic AI sees distorted board
                    col = deterministic_opponent.get_move(distorted_board)
                    
                    if col is not None and game.is_valid_location(current_board, col):
                        row = game.for_next_open_row(current_board, col)
                        if row is not None:
                            game.drop_piece(current_board, row, col, 1)  # Deterministic AI is player 1
                            # Also update the distorted board for next turn
                            distorted_row = game.for_next_open_row(distorted_board, col) 
                            if distorted_row is not None:
                                game.drop_piece(distorted_board, distorted_row, col, 1)
                            moves_made += 1
                            
                            if game.win_checks(current_board, 1):
                                game_metrics['distorted_ai_wins'] = 1
                                break
                        else:
                            game_metrics['illegal_moves_attempted'] += 1
                    else:
                        game_metrics['illegal_moves_attempted'] += 1
                        
                except Exception as e:
                    print(f"Deterministic AI error: {e}")
                    game_metrics['illegal_moves_attempted'] += 1
                    
                turn = 0
                
        # Game finished
        game_metrics['moves_made'] = moves_made
        game_metrics['game_length'] = moves_made
        game_metrics['final_board_state'] = current_board.tolist()
        
        # Check for draw
        if game_metrics['perfect_ai_wins'] == 0 and game_metrics['distorted_ai_wins'] == 0:
            game_metrics['draws'] = 1
            
        return game_metrics
    
    def run_distortion_experiment(self, distortion_levels: List[Tuple[float, float]], 
                                games_per_level: int = 5) -> List[Dict]:
        """
        Run experiments across different distortion levels
        distortion_levels: List of (blur_rate, misclass_rate) tuples
        """
        all_results = []
        
        print(f"Running experiment with {len(distortion_levels)} distortion levels")
        print(f"Playing {games_per_level} games per level")
        print(f"Total games: {len(distortion_levels) * games_per_level * len(self.ground_truth_boards)}")
        
        for blur_rate, misclass_rate in distortion_levels:
            print(f"\n--- Testing distortion: blur={blur_rate:.2f}, misclass={misclass_rate:.2f} ---")
            
            level_results = []
            
            for board_idx, perfect_board in enumerate(self.ground_truth_boards):
                for game_num in range(games_per_level):
                    
                    # Apply distortions
                    self.distortor.clear_log()
                    distorted_board = self.distortor.apply_combined_distortion(
                        perfect_board, blur_rate, misclass_rate
                    )
                    
                    experiment_id = f"blur{blur_rate:.2f}_misclass{misclass_rate:.2f}_board{board_idx}_game{game_num}"
                    
                    # Run single game
                    game_result = self.run_single_game_experiment(
                        perfect_board, distorted_board, experiment_id
                    )
                    
                    # Add distortion info
                    game_result['blur_rate'] = blur_rate
                    game_result['misclass_rate'] = misclass_rate
                    game_result['board_index'] = board_idx
                    game_result['game_number'] = game_num
                    game_result['distortions_applied'] = self.distortor.distortion_log.copy()
                    
                    level_results.append(game_result)
                    all_results.append(game_result)
                    
                    print(f"  Board {board_idx}, Game {game_num}: "
                          f"Perfect AI {'Won' if game_result['perfect_ai_wins'] else 'Lost/Draw'}, "
                          f"Moves: {game_result['moves_made']}, "
                          f"Illegal: {game_result['illegal_moves_attempted']}")
            
            # Calculate summary stats for this distortion level
            self._print_level_summary(level_results, blur_rate, misclass_rate)
        
        self.results = all_results
        return all_results
    
    def _print_level_summary(self, level_results: List[Dict], blur_rate: float, misclass_rate: float):
        """Print summary statistics for a distortion level"""
        total_games = len(level_results)
        perfect_wins = sum(r['perfect_ai_wins'] for r in level_results)
        distorted_wins = sum(r['distorted_ai_wins'] for r in level_results)
        draws = sum(r['draws'] for r in level_results)
        total_illegal = sum(r['illegal_moves_attempted'] for r in level_results)
        avg_game_length = sum(r['game_length'] for r in level_results) / total_games
        avg_move_time = np.mean([t for r in level_results for t in r['move_times']])
        
        print(f"  SUMMARY - Blur: {blur_rate:.2f}, Misclass: {misclass_rate:.2f}")
        print(f"    Perfect AI Win Rate: {perfect_wins}/{total_games} ({perfect_wins/total_games*100:.1f}%)")
        print(f"    Deterministic AI Win Rate: {distorted_wins}/{total_games} ({distorted_wins/total_games*100:.1f}%)")
        print(f"    Draws: {draws}/{total_games} ({draws/total_games*100:.1f}%)")
        print(f"    Total Illegal Moves: {total_illegal}")
        print(f"    Average Game Length: {avg_game_length:.1f} moves")
        print(f"    Average Move Time: {avg_move_time:.3f} seconds")
    
    def save_results_to_csv(self, filename: str):
        """Save experimental results to CSV for analysis"""
        if not self.results:
            print("No results to save")
            return
            
        fieldnames = ['experiment_id', 'timestamp', 'blur_rate', 'misclass_rate', 
                     'board_index', 'game_number', 'moves_made', 'illegal_moves_attempted',
                     'perfect_ai_wins', 'distorted_ai_wins', 'draws', 'game_length',
                     'avg_move_time', 'avg_move_quality_score']
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'experiment_id': result['experiment_id'],
                    'timestamp': result['timestamp'],
                    'blur_rate': result['blur_rate'],
                    'misclass_rate': result['misclass_rate'],
                    'board_index': result['board_index'],
                    'game_number': result['game_number'],
                    'moves_made': result['moves_made'],
                    'illegal_moves_attempted': result['illegal_moves_attempted'],
                    'perfect_ai_wins': result['perfect_ai_wins'],
                    'distorted_ai_wins': result['distorted_ai_wins'],
                    'draws': result['draws'],
                    'game_length': result['game_length'],
                    'avg_move_time': np.mean(result['move_times']) if result['move_times'] else 0,
                    'avg_move_quality_score': np.mean(result['move_quality_scores']) if result['move_quality_scores'] else 0
                }
                writer.writerow(row)
        
        print(f"Results saved to {filename}")

def main():
    """Run the complete experimental framework"""
    print("🔬 CONNECT 4 COMPUTER VISION DISTORTION EXPERIMENT")
    print("=" * 60)
    
    # Step 1: Generate ground truth dataset
    print("📋 Generating ground truth dataset...")
    ground_truth_generator = GroundTruthGenerator()
    boards = ground_truth_generator.generate_full_dataset("ground_truth_boards.json")
    
    # Step 2: Initialize experiment runner
    print("\n🎯 Initializing experiment runner...")
    runner = ExperimentRunner()
    runner.load_ground_truth_boards(boards)
    
    # Step 3: Define distortion levels to test
    distortion_levels = [
        (0.0, 0.0),   # No distortion (baseline)
        (0.05, 0.0),  # 5% blur only
        (0.0, 0.05),  # 5% misclassification only
        (0.05, 0.05), # 5% both
        (0.1, 0.0),   # 10% blur only
        (0.0, 0.1),   # 10% misclassification only
        (0.1, 0.1),   # 10% both
        (0.15, 0.05), # 15% blur, 5% misclass
        (0.05, 0.15), # 5% blur, 15% misclass
        (0.2, 0.1),   # 20% blur, 10% misclass
    ]
    
    # Step 4: Run experiments
    print(f"\n🚀 Running experiments with {len(distortion_levels)} distortion levels...")
    results = runner.run_distortion_experiment(distortion_levels, games_per_level=3)
    
    # Step 5: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"experiment_results_{timestamp}.csv"
    runner.save_results_to_csv(csv_filename)
    
    print(f"\n✅ Experiment complete!")
    print(f"📊 Results saved to: {csv_filename}")
    print(f"📋 Ground truth boards saved to: ground_truth_boards.json")
    print(f"🔢 Total games played: {len(results)}")

if __name__ == "__main__":
    main()
