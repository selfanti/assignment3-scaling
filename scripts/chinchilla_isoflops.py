import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * (x**b)


def main():
    # Load training data
    with open("/home/tao/assignment3-scaling/data/isoflops_curves.json", "r") as f:
        data = json.load(f)

    # Group runs by compute budget
    points = {}
    for item in data:
        compute_budget = item["compute_budget"]
        if compute_budget not in points:
            points[compute_budget] = []
        # (N, loss, D) where D = C / (6N)
        parameters = item["parameters"]
        loss = item["final_loss"]
        tokens = compute_budget / (6 * parameters)
        points[compute_budget].append((parameters, loss, tokens))

    # Plot IsoFLOPs profiles: loss vs model size for each compute budget
    plt.figure(figsize=(10, 6))
    for compute_budget, values in sorted(points.items()):
        x_N = [x[0] for x in values]
        y_loss = [x[1] for x in values]
        plt.plot(x_N, y_loss, marker="o", label=f"C={compute_budget:.0e}")
    plt.xlabel("Model Size (parameters)")
    plt.ylabel("Final Loss")
    plt.title("IsoFLOPs Profiles: Loss vs Model Size")
    plt.legend()
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("/home/tao/assignment3-scaling/isoflops_curves.png")
    plt.close()

    # For each compute budget, pick the run with the lowest loss as the optimum
    c_N_opt = []  # (C, N_opt)
    c_D_opt = []  # (C, D_opt)
    for compute_budget, values in sorted(points.items()):
        best = min(values, key=lambda x: x[1])
        n_opt, _, d_opt = best
        c_N_opt.append((compute_budget, n_opt))
        c_D_opt.append((compute_budget, d_opt))
        print(
            f"Compute budget {compute_budget:.0e}: "
            f"optimal N={n_opt:.2e}, optimal D={d_opt:.2e}, loss={best[1]:.4f}"
        )

    # Fit power laws: N_opt = a * C^b, D_opt = a * C^b
    (a_n, b_n), _ = curve_fit(
        power_law,
        [x[0] for x in c_N_opt],
        [x[1] for x in c_N_opt],
        p0=[1.0, 0.5],
        maxfev=10000,
    )
    (a_d, b_d), _ = curve_fit(
        power_law,
        [x[0] for x in c_D_opt],
        [x[1] for x in c_D_opt],
        p0=[1.0, 0.5],
        maxfev=10000,
    )

    print(f"\nPower law fit for model size: N = {a_n:.4e} * C^{b_n:.4f}")
    print(f"Power law fit for dataset size: D = {a_d:.4e} * C^{b_d:.4f}")

    # Plot scaling law for model size
    plt.figure(figsize=(10, 6))
    c_values = [x[0] for x in c_N_opt]
    n_values = [x[1] for x in c_N_opt]
    plt.scatter(c_values, n_values, color="red", zorder=5, label="Data points")

    # Extrapolation curve
    c_extrap = [min(c_values)]
    while c_extrap[-1] < max(max(c_values), 1e24):
        c_extrap.append(c_extrap[-1] * 1.5)
    n_extrap = [power_law(c, a_n, b_n) for c in c_extrap]
    plt.plot(c_extrap, n_extrap, color="blue", label="Power law fit")

    plt.xlabel("Compute Budget (FLOPs)")
    plt.ylabel("Optimal Model Size (parameters)")
    plt.title("Compute-Optimal Model Size Scaling Law")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("/home/tao/assignment3-scaling/scaling_law_model_size.png")
    plt.close()

    # Plot scaling law for dataset size
    plt.figure(figsize=(10, 6))
    c_values_d = [x[0] for x in c_D_opt]
    d_values = [x[1] for x in c_D_opt]
    plt.scatter(c_values_d, d_values, color="red", zorder=5, label="Data points")

    d_extrap = [power_law(c, a_d, b_d) for c in c_extrap]
    plt.plot(c_extrap, d_extrap, color="blue", label="Power law fit")

    plt.xlabel("Compute Budget (FLOPs)")
    plt.ylabel("Optimal Dataset Size (tokens)")
    plt.title("Compute-Optimal Dataset Size Scaling Law")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("/home/tao/assignment3-scaling/scaling_law_dataset_size.png")
    plt.close()

    # Predictions for 1e23 and 1e24 FLOPs
    n_1e23 = power_law(1e23, a_n, b_n)
    n_1e24 = power_law(1e24, a_n, b_n)
    d_1e23 = power_law(1e23, a_d, b_d)
    d_1e24 = power_law(1e24, a_d, b_d)

    print(
        f"\nOptimal model size for 1e23 FLOPs: {n_1e23:.2e}, "
        f"for 1e24 FLOPs: {n_1e24:.2e}"
    )
    print(
        f"Optimal dataset size for 1e23 FLOPs: {d_1e23:.2e}, "
        f"for 1e24 FLOPs: {d_1e24:.2e}"
    )


if __name__ == "__main__":
    main()
