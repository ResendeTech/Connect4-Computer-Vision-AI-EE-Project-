# Connect 4 Computer Vision AI Project

## Project Overview
This project investigates how computer vision distortions (blur, misclassification) affect AI decision-making in Connect 4 gameplay. We simulate CV errors and measure their impact on AI performance.

## Key Components

### Core Game Logic
- `game.py`: Connect 4 game mechanics, board operations, win detection
- `ai.py`: Minimax AI with alpha-beta pruning (depth 5-6)

### AI Players
- `deterministic_ai.py`: Basic rule-based AI (simple strategies)
- `enhanced_deterministic_ai.py`: Advanced deterministic AIs with position scoring
  - `WeakDeterministicAI`: Simple center-out strategy
  - `ModerateDeterministicAI`: Basic rule following
  - `StrongDeterministicAI`: Uses scoring algorithm, deterministic choices

### Experimental Frameworks
- `final_cv_experiment.py`: Main experiment runner
  - Tests 5 accuracy levels: 100%, 98%, 96%, 94%, 90%
  - 30 games per accuracy level (150 total games)
  - Measures: win rates, illegal moves, game length, move optimality
  - Exports data to CSV for analysis

- `digital_experiment_framework.py`: Advanced experimental framework
  - Ground truth board generation (early/mid/late game scenarios)
  - Controlled distortion application (blur + misclassification)
  - Comprehensive metrics collection

## Experimental Design

### CV Distortion Simulation
1. **Blur**: Randomly removes pieces from the board (simulates unclear vision)
2. **Misclassification**: Swaps piece colors (simulates color detection errors)
3. **Combined**: Both distortions applied simultaneously

### Accuracy Levels Tested
- 100%: Perfect vision (control group)
- 98%: 1% blur + 1% misclassification  
- 96%: 2% blur + 2% misclassification
- 94%: 3% blur + 3% misclassification
- 90%: 5% blur + 5% misclassification

### Key Metrics
- **Win Rate**: Percentage of games won by distorted vs perfect AI
- **Illegal Moves**: Moves that violate game rules due to distorted vision
- **Game Length**: Number of moves per game
- **Move Optimality**: Quality of decisions compared to oracle AI
- **Move Time**: Decision-making speed

## Key Findings
- As CV accuracy decreases, AI performance degrades significantly
- 100% accuracy: AI wins consistently with 0 illegal moves
- Lower accuracies: Dramatic drop in win rate, increase in illegal moves
- Game lengths become shorter as distorted AI makes poor decisions

## Data Output
Results exported to CSV with columns:
- Trial number, accuracy percentage, distortion levels
- Board index, starting player, moves made
- Win/loss outcomes, illegal move counts
- Game length, move times, optimality scores

## Usage
```bash
# Run main experiment
python final_cv_experiment.py

# Run advanced framework
python digital_experiment_framework.py
```

## Technical Details
- Python 3.11 with numpy for board operations
- Minimax with alpha-beta pruning for strategic AI
- Deterministic AIs ensure reproducible experiments
- CSV export for Excel/statistical analysis

## Files Structure
```
CS_EE_Project/
├── game.py                           # Core Connect 4 logic
├── ai.py                            # Minimax AI implementation  
├── deterministic_ai.py              # Basic deterministic AI
├── enhanced_deterministic_ai.py     # Advanced deterministic AIs
├── final_cv_experiment.py           # Main experiment runner
├── digital_experiment_framework.py  # Advanced experiment framework
├── *.csv                           # Experimental results
└── __pycache__/                    # Python cache files
```

This project demonstrates how computer vision accuracy directly impacts AI decision-making quality in strategic games, with applications to robotics and automated systems.
