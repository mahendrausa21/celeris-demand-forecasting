"""One command rebuilds every output:   python run_pipeline.py"""
import time
from pipeline import clean, anomalies, features, forecast, planner


def main():
    t0 = time.time()
    clean.run_all()                       # step 1: cleaning (all files)
    anomalies.run_anomalies()             # step 2: anomaly register, unconstrained demand, regimes
    features.run_features()               # step 3: macro lead/lag, EV shift, intermittency classes
    forecast.run_forecast()               # step 4: backtest + forecast + P10/P50/P90 + submission.csv
    planner.build_exceptions()            # exception list for planners
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()