# run_optimization.py
import json
from backtesting.parameter_optimization import optimize_parameters

def main():
    study = optimize_parameters(n_trials=50)
    best_trial = study.best_trial
    print("===== 최적 파라미터 =====")
    print(best_trial)
    with open("best_params.json", "w") as f:
        json.dump(best_trial.params, f, indent=4)

if __name__ == "__main__":
    main()
