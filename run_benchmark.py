"""run_benchmark.py

Command-line runner for the translated MATLAB/Simulink generated-code benchmark.

Examples
--------
python run_benchmark.py --time 0.01
python run_benchmark.py --time 1.5 --output_dir benchmark_results

The script saves numerical history and MATLAB-comparable plots.  It does not
modify the benchmark equations.
"""

from __future__ import annotations

import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import MatlabFullBenchmark, BenchmarkConfig, BenchmarkScenario
from parameters import PARAMS


def save_plot(path, time, series, xlabel="Time (s)", ylabel="", title=""):
    plt.figure(figsize=(10, 4.5))
    for y, label in series:
        plt.plot(time, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if any(label for _, label in series):
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def finite_report(arrays):
    print("\n" + "=" * 78)
    print("NaN / INF CHECK")
    print("=" * 78)
    names = [
        "frequency", "P", "Q", "P_filtered", "Q_filtered",
        "vod", "voq", "iod", "ioq", "ifd", "ifq",
        "ifd_ref", "ifq_ref", "vd_command", "vq_command",
        "md", "mq", "battery_current",
    ]
    ok = True
    for name in names:
        x = arrays[name]
        finite = np.isfinite(x).all()
        ok = ok and finite
        print(
            f"{name:<16} finite={str(finite):<5} "
            f"min={np.nanmin(x):>14.6f} "
            f"max={np.nanmax(x):>14.6f} "
            f"final={x[-1]:>14.6f}"
        )
    print("=" * 78)
    print("PASS: all selected signals are finite." if ok else "FAIL: NaN/Inf detected.")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Run the translated MATLAB generated-code GFM benchmark."
    )
    parser.add_argument("--time", type=float, default=1.5,
                        help="Simulation duration in seconds (default: 1.5).")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory for plots/history.")
    parser.add_argument("--grid_voltage", type=float, default=13800.0,
                        help="Upstream source line-line RMS voltage in V.")
    parser.add_argument("--grid_frequency", type=float, default=60.0,
                        help="Upstream source frequency in Hz.")
    parser.add_argument("--vdc", type=float, default=800.0,
                        help="Fixed DC terminal voltage in V for current benchmark scope.")
    parser.add_argument("--load_initial", type=int, default=50000,
                        choices=[40000, 50000, 60000],
                        help="Initial exact MATLAB-generated load in W.")
    parser.add_argument(
        "--load_disturbance",
        type=int,
        default=None,
        choices=[40000, 50000, 60000],
        help="Temporary disturbed load in W."
    )

    parser.add_argument(
        "--load_step_time",
        type=float,
        default=None,
        help="Time at which the load disturbance is applied."
    )

    parser.add_argument(
        "--load_restore_time",
        type=float,
        default=None,
        help="Time at which load returns to its initial value."
    )    
    parser.add_argument("--progress", type=float, default=0.1,
                        help="Progress-print interval in seconds; <=0 disables it.")
    args = parser.parse_args()

    if args.time <= 0:
        raise ValueError("--time must be positive")

    os.makedirs(args.output_dir, exist_ok=True)

    config = BenchmarkConfig(
        grid_voltage_ll_rms=args.grid_voltage,
        grid_frequency=args.grid_frequency,
        dc_voltage=args.vdc,
    )

    scenario = BenchmarkScenario(
        load_initial=args.load_initial,
        load_disturbance=args.load_disturbance,
        load_step_time=args.load_step_time,
        load_restore_time=args.load_restore_time,
    )

    sim = MatlabFullBenchmark(
        config=config,
        scenario=scenario,
    )

    n_steps = int(round(args.time / PARAMS.dt))
    progress_every = None
    if args.progress > 0:
        progress_every = max(1, int(round(args.progress / PARAMS.dt)))

    print("=" * 78)
    print("MATLAB GENERATED-CODE BENCHMARK")
    print("=" * 78)
    print(f"dt                 : {PARAMS.dt:.9g} s")
    print(f"duration           : {args.time:.6f} s")
    print(f"steps              : {n_steps}")
    print(f"grid voltage LL RMS: {args.grid_voltage:.3f} V")
    print(f"grid frequency     : {args.grid_frequency:.3f} Hz")
    print(f"DC voltage         : {args.vdc:.3f} V")
    print(f"P_ref              : {PARAMS.P_ref:.3f} W")
    print(f"Q_ref              : {PARAMS.Q_ref:.3f} var")
    print(f"V_ref              : {PARAMS.voltage_ref:.3f} V")
    print(f"initial load       : {args.load_initial:.3f} W")
    if args.load_disturbance is not None:
        print(f"disturbed load     : {args.load_disturbance:.3f} W")
        print(f"disturbance time   : {args.load_step_time:.6f} s")
        print(f"restore time       : {args.load_restore_time:.6f} s")
    else:
        print("load disturbance   : none")

    print("=" * 78)

    history = sim.run(args.time, progress_every=progress_every)
    a = history.arrays()

    if len(a["time"]) == 0:
        raise RuntimeError("Benchmark produced no samples")

    finite_report(a)

    print("\n" + "=" * 78)
    print("FINAL BENCHMARK VALUES")
    print("=" * 78)
    print(f"time        : {a['time'][-1]:.6f} s")
    print(f"frequency   : {a['frequency'][-1]:.6f} Hz")
    print(f"P_B3        : {a['P'][-1]:.6f} W")
    print(f"P_filtered  : {a['P_filtered'][-1]:.6f} W")
    print(f"Q_B3        : {a['Q'][-1]:.6f} var")
    print(f"Q_filtered  : {a['Q_filtered'][-1]:.6f} var")
    print(f"Vod         : {a['vod'][-1]:.6f} V")
    print(f"Voq         : {a['voq'][-1]:.6f} V")
    print(f"Iod         : {a['iod'][-1]:.6f} A")
    print(f"Ioq         : {a['ioq'][-1]:.6f} A")
    print(f"Ifd         : {a['ifd'][-1]:.6f} A")
    print(f"Ifq         : {a['ifq'][-1]:.6f} A")
    print(f"Ifd_ref     : {a['ifd_ref'][-1]:.6f} A")
    print(f"Ifq_ref     : {a['ifq_ref'][-1]:.6f} A")
    print(f"Vd_command  : {a['vd_command'][-1]:.6f} V")
    print(f"Vq_command  : {a['vq_command'][-1]:.6f} V")
    print(f"md          : {a['md'][-1]:.6f}")
    print(f"mq          : {a['mq'][-1]:.6f}")
    print(f"load_power  : {a['load_power'][-1]:.3f} W")
    print("=" * 78)

    npz_path = os.path.join(args.output_dir, "benchmark_history.npz")
    np.savez_compressed(npz_path, **a)

    t = a["time"]
    plot_specs = [
        ("load_power.png", [(a["load_power"] / 1000.0, "Load")], "Load (kW)", "Load Scenario"),
        ("frequency.png", [(a["frequency"], "Benchmark")], "Frequency (Hz)", "Frequency Response"),
        ("active_power.png", [(a["P"], "P_B3"), (a["P_filtered"], "P_filtered")], "Power (W)", "Active Power"),
        ("reactive_power.png", [(a["Q"], "Q_B3"), (a["Q_filtered"], "Q_filtered")], "Reactive power (var)", "Reactive Power"),
        ("b3_voltage_dq.png", [(a["vod"], "Vod"), (a["voq"], "Voq"), (a["vod_ref"], "Vod_ref")], "Voltage (V)", "B3 Voltage and Reference"),
        ("b3_current_dq.png", [(a["iod"], "Iod"), (a["ioq"], "Ioq")], "Current (A)", "B3 Current"),
        ("b1_current_dq.png", [(a["ifd"], "Ifd"), (a["ifq"], "Ifq"), (a["ifd_ref"], "Ifd_ref"), (a["ifq_ref"], "Ifq_ref")], "Current (A)", "B1 Current and References"),
        ("voltage_commands.png", [(a["vd_command"], "Vd_command"), (a["vq_command"], "Vq_command")], "Voltage command", "Current-Controller Voltage Commands"),
        ("modulation_dq.png", [(a["md"], "md"), (a["mq"], "mq")], "Modulation", "dq Modulation Commands"),
        ("pwm_carrier.png", [(a["carrier"], "carrier"), (a["ma"], "ma"), (a["mb"], "mb"), (a["mc"], "mc")], "Normalized value", "PWM References and Carrier"),
    ]

    for filename, series, ylabel, title in plot_specs:
        save_plot(
            os.path.join(args.output_dir, filename),
            t,
            series,
            ylabel=ylabel,
            title=title,
        )

    print("\nSaved results:")
    print(f"  {npz_path}")
    for filename, *_ in plot_specs:
        print(f"  {os.path.join(args.output_dir, filename)}")

    return history


if __name__ == "__main__":
    main()
