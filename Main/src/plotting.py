import matplotlib.pyplot as plt


def plot_results(results, title_prefix="OpenNTN"):
    plt.figure(figsize=(7, 5))
    plt.semilogy(results["snr_db"], results["ber"], marker="o")
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title(f"{title_prefix}: BER vs SNR")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.semilogy(results["snr_db"], results["bler"], marker="s")
    plt.xlabel("SNR [dB]")
    plt.ylabel("BLER")
    plt.title(f"{title_prefix}: BLER vs SNR")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(results["snr_db"], results["iSE"], marker="^")
    plt.xlabel("SNR [dB]")
    plt.ylabel("iSE [bits/s/Hz]")
    plt.title(f"{title_prefix}: iSE vs SNR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
