import numpy as np


def run_snr_sweep(executor, snr_db_list, verbose=True):
    out = {
        "snr_db": [],
        "ber": [],
        "bler": [],
        "iSE": [],
        "nominal_se": [],
        "tx_power_linear": [],
    }

    for snr_db in snr_db_list:
        metrics = executor(float(snr_db))

        out["snr_db"].append(float(snr_db))
        out["ber"].append(metrics["ber"])
        out["bler"].append(metrics["bler"])
        out["iSE"].append(metrics["iSE"])
        out["nominal_se"].append(metrics["nominal_se"])
        out["tx_power_linear"].append(metrics["tx_power_linear"])

        if verbose:
            print(
                f"SNR={snr_db:>5.1f} dB | "
                f"BER={metrics['ber']:.4e} | "
                f"BLER={metrics['bler']:.4e} | "
                f"iSE={metrics['iSE']:.4f}"
            )

    for key in out:
        out[key] = np.array(out[key])

    return out


def run_snr_sweep_mc(executor, snr_db_list, num_iter=10, verbose=True):
    results_accum = []

    for i in range(num_iter):
        if verbose:
            print(f"Iteration {i+1}/{num_iter}")
        results_accum.append(run_snr_sweep(executor, snr_db_list, verbose=verbose))

    avg_results = {}
    for key in results_accum[0].keys():
        avg_results[key] = np.mean([r[key] for r in results_accum], axis=0)

    return avg_results
