import json
import time
import numpy as np
from datetime import datetime
import csv

class GameDataLogger:
    """
    Logs detailed data for each game and move evaluation
    """
    
    def __init__(self, trial_name, accuracy_level=100):
        self.trial_name = trial_name
        self.accuracy_level = accuracy_level
        self.games_data = []
        self.current_game_data = None
        self.move_evaluations = []
        
    def start_new_game(self, game_id):
        """Start logging a new game"""
        self.current_game_data = {
            "game_id": game_id,
            "trial_name": self.trial_name,
            "accuracy_level": self.accuracy_level,
            "timestamp": datetime.now().isoformat(),
            "moves": [],
            "game_result": None,  # Will be filled at end
            "total_moves": 0,
            "minimax_illegal_moves": 0,
            "deterministic_illegal_moves": 0,
            "minimax_total_time": 0,
            "vision_errors": 0
        }
    
    def log_move_evaluation(self, player, board_state, move_data):
        """
        Log a single move evaluation
        move_data should contain:
        - chosen_column: int
        - is_legal: bool
        - evaluation_time: float
        - minimax_score: float (if applicable)
        - vision_board: numpy array (detected board state)
        - true_board: numpy array (actual board state)
        - vision_accuracy: float (0-1)
        """
        if self.current_game_data is None:
            return  # No active game
            
        evaluation = {
            "move_number": len(self.current_game_data["moves"]) + 1,
            "player": player,  # 1 for minimax, 2 for deterministic
            "timestamp": time.time(),
            "chosen_column": move_data.get("chosen_column"),
            "is_legal": move_data.get("is_legal", True),
            "evaluation_time": move_data.get("evaluation_time", 0),
            "minimax_score": move_data.get("minimax_score"),
            "board_state_hash": hash(str(board_state.flatten())),
            "vision_accuracy": move_data.get("vision_accuracy", 1.0),
            "vision_errors": move_data.get("vision_errors", 0)
        }
        
        self.current_game_data["moves"].append(evaluation)
        
        # Track illegal moves
        if not move_data.get("is_legal", True):
            if player == 1:  # Minimax
                self.current_game_data["minimax_illegal_moves"] += 1
            else:
                self.current_game_data["deterministic_illegal_moves"] += 1
        
        # Track timing for minimax
        if player == 1:
            self.current_game_data["minimax_total_time"] += move_data.get("evaluation_time", 0)
    
    def finish_game(self, winner, reason="normal"):
        """Finish current game logging"""
        if self.current_game_data:
            self.current_game_data["game_result"] = winner
            self.current_game_data["end_reason"] = reason
            self.current_game_data["total_moves"] = len(self.current_game_data["moves"])
            
            # Calculate statistics
            self.current_game_data["illegal_move_rate"] = (
                (self.current_game_data["minimax_illegal_moves"] + 
                 self.current_game_data["deterministic_illegal_moves"]) / 
                max(1, self.current_game_data["total_moves"])
            )
            
            self.current_game_data["avg_minimax_time"] = (
                self.current_game_data["minimax_total_time"] / 
                max(1, sum(1 for move in self.current_game_data["moves"] if move["player"] == 1))
            )
            
            self.games_data.append(self.current_game_data)
            self.current_game_data = None
    
    def get_trial_statistics(self):
        """Get comprehensive statistics for the trial"""
        if not self.games_data:
            return {}
            
        total_games = len(self.games_data)
        minimax_wins = sum(1 for game in self.games_data if game["game_result"] == 1)
        deterministic_wins = sum(1 for game in self.games_data if game["game_result"] == 2)
        draws = sum(1 for game in self.games_data if game["game_result"] == 0)
        
        # Move evaluation statistics
        all_moves = []
        for game in self.games_data:
            all_moves.extend(game["moves"])
        
        minimax_moves = [move for move in all_moves if move["player"] == 1]
        total_illegal_moves = sum(1 for move in all_moves if not move["is_legal"])
        
        statistics = {
            "trial_name": self.trial_name,
            "accuracy_level": self.accuracy_level,
            "total_games": total_games,
            "total_move_evaluations": len(all_moves),
            "minimax_move_evaluations": len(minimax_moves),
            
            # Win rate statistics
            "minimax_win_rate": minimax_wins / total_games if total_games > 0 else 0,
            "deterministic_win_rate": deterministic_wins / total_games if total_games > 0 else 0,
            "draw_rate": draws / total_games if total_games > 0 else 0,
            
            # Move quality statistics
            "illegal_move_rate": total_illegal_moves / len(all_moves) if all_moves else 0,
            "avg_minimax_evaluation_time": np.mean([move["evaluation_time"] for move in minimax_moves]) if minimax_moves else 0,
            "std_minimax_evaluation_time": np.std([move["evaluation_time"] for move in minimax_moves]) if minimax_moves else 0,
            
            # Vision accuracy statistics
            "avg_vision_accuracy": np.mean([move.get("vision_accuracy", 1.0) for move in all_moves]) if all_moves else 0,
            "vision_error_rate": sum(move.get("vision_errors", 0) for move in all_moves) / len(all_moves) if all_moves else 0
        }
        
        return statistics
    
    def save_to_files(self, base_filename):
        """Save data to JSON and CSV files"""
        # Save raw data to JSON
        json_filename = f"{base_filename}_raw_data.json"
        with open(json_filename, 'w') as f:
            json.dump({
                "trial_info": {
                    "name": self.trial_name,
                    "accuracy_level": self.accuracy_level,
                    "total_games": len(self.games_data)
                },
                "games": self.games_data,
                "statistics": self.get_trial_statistics()
            }, f, indent=2)
        
        # Save statistics to CSV
        csv_filename = f"{base_filename}_statistics.csv"
        stats = self.get_trial_statistics()
        
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        
        # Save move-by-move data to CSV
        moves_csv_filename = f"{base_filename}_moves.csv"
        all_moves = []
        for game in self.games_data:
            for move in game["moves"]:
                move_data = move.copy()
                move_data["game_id"] = game["game_id"]
                move_data["trial_name"] = self.trial_name
                move_data["accuracy_level"] = self.accuracy_level
                all_moves.append(move_data)
        
        if all_moves:
            with open(moves_csv_filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_moves[0].keys())
                writer.writeheader()
                writer.writerows(all_moves)
        
        print(f"Data saved to:")
        print(f"  - {json_filename}")
        print(f"  - {csv_filename}")
        print(f"  - {moves_csv_filename}")
