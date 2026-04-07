import numpy as np
import tensorflow as tf

from sionna.phy import Block
from sionna.phy.utils import ebnodb2no
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.ofdm import (
    ResourceGrid,
    ResourceGridMapper,
    RemoveNulledSubcarriers,
    LSChannelEstimator,
    LMMSEEqualizer,
)
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38811 import AntennaArray, DenseUrban, Urban, SubUrban
from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_ntn_topology


class OpenNTNPhyExecutor(Block):
    def __init__(self, action_cfg, sim_cfg):
        super().__init__()

        self.action_cfg = action_cfg
        self.sim_cfg = sim_cfg

        self.batch_size = sim_cfg.batch_size
        self.perfect_csi = sim_cfg.perfect_csi
        self.num_ut = sim_cfg.num_ut
        self.num_ut_ant = sim_cfg.num_ut_ant
        self.num_bs_ant = sim_cfg.num_bs_ant

        self.modulation = action_cfg.modulation
        self.code_rate = float(action_cfg.code_rate)
        self.power_boost_db = float(action_cfg.power_boost_db)

        self.num_bits_per_symbol = self._bits_per_symbol(self.modulation)

        direction = "downlink"

        # One TX, one UT link
        num_tx = 1
        num_streams_per_tx = self.num_bs_ant

        rx_tx_association = np.zeros([self.num_ut, num_tx])
        rx_tx_association[:, 0] = 1

        self.rg = ResourceGrid(
            num_ofdm_symbols=sim_cfg.num_ofdm_symbols,
            fft_size=sim_cfg.fft_size,
            subcarrier_spacing=sim_cfg.subcarrier_spacing,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=sim_cfg.cyclic_prefix_length,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=sim_cfg.pilot_ofdm_symbol_indices,
        )

        self.sm = StreamManagement(rx_tx_association, num_streams_per_tx)

        ut_array = AntennaArray(
            num_rows=1,
            num_cols=self.num_ut_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=sim_cfg.carrier_frequency,
        )

        bs_array = AntennaArray(
            num_rows=1,
            num_cols=self.num_bs_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=sim_cfg.carrier_frequency,
        )

        if sim_cfg.channel_model_name == "DenseUrban":
            self.channel_model = DenseUrban(
                carrier_frequency=sim_cfg.carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=direction,
                elevation_angle=sim_cfg.elevation_angle,
                doppler_enabled=sim_cfg.doppler_enabled,
            )
        elif sim_cfg.channel_model_name == "Urban":
            self.channel_model = Urban(
                carrier_frequency=sim_cfg.carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=direction,
                elevation_angle=sim_cfg.elevation_angle,
                doppler_enabled=sim_cfg.doppler_enabled,
            )
        elif sim_cfg.channel_model_name == "SubUrban":
            self.channel_model = SubUrban(
                carrier_frequency=sim_cfg.carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=direction,
                elevation_angle=sim_cfg.elevation_angle,
                doppler_enabled=sim_cfg.doppler_enabled,
            )
        else:
            raise ValueError("channel_model_name must be DenseUrban, Urban, or SubUrban")

        topology = gen_ntn_topology(
            batch_size=sim_cfg.batch_size,
            num_ut=sim_cfg.num_ut,
            scenario=sim_cfg.scenario,
        )
        self.channel_model.set_topology(*topology)

        self.binary_source = BinarySource()

        self.n = int(self.rg.num_data_symbols * self.num_bits_per_symbol)
        self.k = int(self.n * self.code_rate)

        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.rg_mapper = ResourceGridMapper(self.rg)

        self.ofdm_channel = OFDMChannel(
            self.channel_model,
            self.rg,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True,
        )

        self.remove_nulled_subcarriers = RemoveNulledSubcarriers(self.rg)
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")
        self.lmmse_equ = LMMSEEqualizer(self.rg, self.sm)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)

    @staticmethod
    def _bits_per_symbol(modulation):
        table = {
            "QPSK": 2,
            "16QAM": 4,
            "64QAM": 6,
            "256QAM": 8,
        }
        if modulation not in table:
            raise ValueError(f"Unsupported modulation: {modulation}")
        return table[modulation]

    def call(self, ebno_db):
        no = ebnodb2no(
            ebno_db,
            self.num_bits_per_symbol,
            self.code_rate,
            self.rg,
        )

        tx_power_linear = 10.0 ** (self.power_boost_db / 10.0)
        no_eff_input = no / tx_power_linear

        b = self.binary_source([self.batch_size, 1, self.num_bs_ant, self.k])

        c = self.encoder(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)

        x_rg = x_rg * tf.cast(np.sqrt(tx_power_linear), x_rg.dtype)

        y, h = self.ofdm_channel(x_rg, no)

        if self.perfect_csi:
            h_hat = self.remove_nulled_subcarriers(h)
            err_var = 0.0
        else:
            h_hat, err_var = self.ls_est(y, no_eff_input)

        x_hat, no_eff = self.lmmse_equ(y, h_hat, err_var, no_eff_input)

        llr = self.demapper(x_hat, no_eff)
        b_hat = self.decoder(llr)

        b_true = tf.cast(b, tf.float32)
        b_hat = tf.cast(b_hat, tf.float32)

        bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(b_true, b_hat), tf.float32))
        total_bits = tf.cast(tf.size(b_true), tf.float32)
        ber = float((bit_errors / total_bits).numpy())

        block_errors = tf.reduce_any(tf.not_equal(b_true, b_hat), axis=-1)
        bler = float(tf.reduce_mean(tf.cast(block_errors, tf.float32)).numpy())

        nominal_se = self.code_rate * self.num_bits_per_symbol
        iSE = nominal_se * (1.0 - bler)

        return {
            "ber": ber,
            "bler": bler,
            "iSE": float(iSE),
            "nominal_se": float(nominal_se),
            "tx_power_linear": float(tx_power_linear),
        }
