from dataclasses import dataclass, field
from typing import List


@dataclass
class ActionConfig:
    modulation: str = "QPSK"   # QPSK, 16QAM, 64QAM, 256QAM
    code_rate: float = 0.5
    power_boost_db: float = 0.0


@dataclass
class SimulationConfig:
    scenario: str = "dur"
    channel_model_name: str = "DenseUrban"  # DenseUrban, Urban, SubUrban
    perfect_csi: bool = True
    doppler_enabled: bool = True
    elevation_angle: float = 50.0
    carrier_frequency: float = 2.0e9
    batch_size: int = 512
    num_ut: int = 1
    num_ut_ant: int = 1
    num_bs_ant: int = 1
    num_ofdm_symbols: int = 14
    fft_size: int = 256
    subcarrier_spacing: float = 15e3
    cyclic_prefix_length: int = 16
    pilot_ofdm_symbol_indices: List[int] = field(default_factory=lambda: [2, 11])
