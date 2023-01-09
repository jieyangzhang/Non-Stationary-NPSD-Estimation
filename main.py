from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
import sys
import soundfile as sf
from nara_wpe.utils import stft
from nara_wpe.utils import istft
from numpy import dtype, seterr
from scipy.special import expm1
from sympy import expint

seterr(all='raise')
epison = 1e-3
max_val = 1e3

class NonStationaryNPSDEstimator:
    def __init__(self, K, M, r):
        self.l_min = 0.1
        self.l_max = 0.998
        self.rho = 2.5
        self.c = 3
        self.w = 10 # CDR, 频点平均用的窗长为2*w+1, 使用hamming窗
        self.CDR_window = np.hamming(2*self.w+1)
        self.q_thr1 = 0.95
        self.q_thr2 = 0.9
        self.K = K # 频点个数
        self.M = M # 通道数
        self.hat_Phi_y = np.kron(np.ones([K, 1]), np.eye(M) * 1e-4).reshape([K, M, M])
        self.hat_Phi_v = np.kron(np.ones([K, 1]), np.eye(M) * 1e-4).reshape([K, M, M])
        self.alpha_y = 0.95
        self.alpha_v = 0.95
        self.buffer_size = 8
        self.auto_PSD_buffer = np.zeros([self.buffer_size, K, M])
        self.cross_PSD_buffer = np.zeros([self.buffer_size, K, M - 1], dtype=np.complex128) # [1, 2], [1, 3], [1, 4]
        self.buffer_index = 0
        self.count = 0
        self.r = r # 相邻麦的距离
        c = 343 # 声速
        freqs = np.linspace(0, 8000, 513, endpoint=True)
        freqs[0] = 1e-10
        self.wavelength = 1 / freqs * c
        self.K_high1 = 52
        self.K_low1 = 1
        self.K_high2 = 448
        self.K_low2 = 192
        self.q_list = []
        self.p_list = []
        self.lambd = 0.85
        self.chi = np.zeros(K, dtype=np.float32)
        
    def calculate_hat_Gamma(self, Y):
        """_summary_

        Args:
            Y ([K, M]): _description_
        """
        # 更新自功率谱缓存以及互功率谱缓存
        idx = self.buffer_index % self.buffer_size
        self.auto_PSD_buffer[idx] = np.power(np.abs(Y), 2)
        
        for i in range(self.M - 1):
            self.cross_PSD_buffer[idx, :, i] = Y[:, 0] * np.conj(Y[:, i+1])
        self.buffer_index += 1
        
        # 计算自功率谱以及互功率谱的均值
        auto_PSD = np.average(self.auto_PSD_buffer, axis=0)
        cross_PSD = np.average(self.cross_PSD_buffer, axis=0)
        
        ''' 
            |    freq_range    |      microphone_used    |
            |      [1,650]     |           [1,4]         |
            |    (650, 1500]   |       [1,4] & [1,3]     |
            |    (1500, 8000]  |   [1,2] & [1,3] & [1,4] | 
        '''
        # 计算[1,4]对麦克风的CDR， [1,4]对参与了全频带计算。
        hat_gamma_1_4 = cross_PSD[:, 2] / np.sqrt(auto_PSD[:, 0] * auto_PSD[:, 3])
        hat_theta_1_4 = np.angle(cross_PSD[:, 2]) # doa
        gamma_1_4_v = np.sinc(2 * np.pi / self.wavelength * self.r * 3)
        try:
            hat_Gamma_1_4 = np.real((gamma_1_4_v - hat_gamma_1_4) / (hat_gamma_1_4 - np.exp(1j * hat_theta_1_4)))
        except FloatingPointError:
            hat_Gamma_1_4 = np.real((gamma_1_4_v - hat_gamma_1_4) / (hat_gamma_1_4 - np.exp(1j * hat_theta_1_4) + epison))
        hat_Gamma_1_4 = np.maximum(hat_Gamma_1_4, np.zeros_like(hat_Gamma_1_4))
            

        # 计算[1,3]对麦克风的CDR， [1,3]对参与了（650, 8000]频段范围的计算。
        hat_gamma_1_3 = cross_PSD[42:, 1] / np.sqrt(auto_PSD[42:, 0] * auto_PSD[42:, 2])
        hat_theta_1_3 = np.angle(cross_PSD[42:, 1]) # doa
        gamma_1_3_v = np.sinc(2 * np.pi / self.wavelength[42:] * self.r * 2)
        try:
            hat_Gamma_1_3 = np.real((gamma_1_3_v - hat_gamma_1_3) / (hat_gamma_1_3 - np.exp(1j * hat_theta_1_3)))
        except FloatingPointError:
            hat_Gamma_1_3 = np.real((gamma_1_3_v - hat_gamma_1_3) / (hat_gamma_1_3 - np.exp(1j * hat_theta_1_3) + epison))
        hat_Gamma_1_3 = np.maximum(hat_Gamma_1_3, np.zeros_like(hat_Gamma_1_3))
        
        # 计算[1,2]对麦克风CDR， [1,2]对参与了(1500, 8000]频段范围的计算。
        hat_gamma_1_2 = cross_PSD[96:, 1] / np.sqrt(auto_PSD[96:, 0] * auto_PSD[96:, 1])
        hat_theta_1_2 = np.angle(cross_PSD[96:, 0]) # doa
        gamma_1_2_v = np.sinc(2 * np.pi / self.wavelength[96:] * self.r * 1)
        try:
            hat_Gamma_1_2 = np.real((gamma_1_2_v - hat_gamma_1_2) / (hat_gamma_1_2 - np.exp(1j * hat_theta_1_2)))
        except FloatingPointError:
            hat_Gamma_1_2 = np.real((gamma_1_2_v - hat_gamma_1_2) / (hat_gamma_1_2 - np.exp(1j * hat_theta_1_2) + epison))
        hat_Gamma_1_2 = np.maximum(hat_Gamma_1_2, np.zeros_like(hat_Gamma_1_2))
        
        hat_Gamma = hat_Gamma_1_4.copy()
        hat_Gamma[42:] += hat_Gamma_1_3
        hat_Gamma[96:] += hat_Gamma_1_2
        hat_Gamma[42:96] /= 2
        hat_Gamma[96:] /= 3
        
        # 计算frequency-average CDR
        padded_hat_Gamma = np.pad(hat_Gamma, (self.w, self.w), mode='constant')
        hat_Gamma_frequency_average = np.convolve(padded_hat_Gamma, self.CDR_window, mode='valid')
        
        return hat_Gamma, hat_Gamma_frequency_average
    
    def calculate_priori_SAP(self, hat_Gamma, hat_Gamma_frequency_average):
        tmp = np.power(10, self.c * self.rho / 10)
        q_local = self.l_min + (self.l_max - self.l_min) * (tmp) / (tmp + np.power(hat_Gamma, self.rho))
        q_global = self.l_min + (self.l_max - self.l_min) * (tmp) / (tmp + np.power(hat_Gamma_frequency_average, self.rho))
        # q_frame的计算是将q_local分别在[1, 800]:[0, 52), [3000, 7000]:[192:448)这两个频段的均值和q_thr1和q_thr2做比较
        q_low = np.average(q_local[self.K_low1:self.K_high1])
        q_high = np.average(q_local[self.K_low2:self.K_high2])
        q_frame = (q_low > self.q_thr1) and (q_high > self.q_thr2)
        
        q_v = 1 - (1 - q_local) * (1 - q_global) * (1 - q_frame)
        self.q_list.append(1 - q_v)
        
        return q_v
    
    def calculate_posteriori_SAP(self, Phi_x, Phi_v, q_v, Y):
        try:
            Phi_v_inv = np.linalg.inv(Phi_v)
        except np.linalg.LinAlgError as e:
            print(e)
            maxtrix_epison = np.kron(np.ones([self.K, 1]), np.eye(self.M) * epison).reshape([self.K, self.M, self.M])
            Phi_v += maxtrix_epison
            Phi_v_inv = np.linalg.inv(Phi_v)
            
        xi = 1 + np.trace(np.einsum('kij,kjl->kil', Phi_v_inv, Phi_x), axis1=1, axis2=2)
        tmp1 = np.einsum('kij,kj->ki', Phi_v_inv, Y)
        tmp1_conj = np.conj(tmp1)
        tmp2 = np.einsum('kij,kj->ki', Phi_x, tmp1)
        beta = np.einsum('ki,ki->k', tmp1_conj, tmp2)
        ex = np.minimum(expm1(-np.real(beta / xi)) + 1, max_val * np.ones(self.K))
        ex = np.maximum(ex, np.ones(self.K) * epison)
        try:
            post_SPP = 1 / (1 + q_v / (1 - q_v) * xi * ex)
        except FloatingPointError as e:
            print(e)
            print(ex)
            post_SPP = 1 / (1 + q_v / (1 - q_v) * xi * ex)
        post_SAP = 1 - np.real(post_SPP)
        
        return post_SAP
    
    def process_frame(self, Y):
        K, M = Y.shape
        if (K != self.K) or (M != self.M):
            print('shape of y[%d:%d] diamatch!' % {K, M})
            return
        
        Y_Conj = Y.conj()
        
        YYH = np.einsum('ki,kj->kij', Y, Y_Conj)
        
        # update hat_Phi_y
        self.hat_Phi_y = self.alpha_y * self.hat_Phi_y + (1 - self.alpha_y) * YYH
        
        # estimate hat_Phi_x
        hat_Phi_x = self.hat_Phi_y - self.hat_Phi_v
        
        # estimate hat_Gamma, i.e. CDR
        hat_Gamma, hat_Gamma_frequency_average = self.calculate_hat_Gamma(Y)
        
        q_v = self.calculate_priori_SAP(hat_Gamma, hat_Gamma_frequency_average)
        
        tilde_p_v = self.calculate_posteriori_SAP(hat_Phi_x, self.hat_Phi_v, q_v, Y)
        self.hat_Phi_v = np.einsum('k,kij->kij', (1 - tilde_p_v / (self.lambd * self.chi + tilde_p_v)), self.hat_Phi_v) \
            + np.einsum('k,kij->kij', tilde_p_v / (self.lambd * self.chi + tilde_p_v), YYH)
        self.chi = self.lambd * self.chi + tilde_p_v # eq.24
        
        # p_v = self.calculate_posteriori_SAP(hat_Phi_x, self.hat_Phi_v, q_v, Y)
        # self.chi = self.lambd * self.chi + p_v # eq.24
        # self.hat_Phi_v = np.einsum('k,kij->kij', (1 - p_v / (self.lambd * self.chi + p_v)), self.hat_Phi_v) \
        #     + np.einsum('k,kij->kij', p_v / (self.lambd * self.chi + p_v), YYH)

        self.p_list.append(1 - tilde_p_v)
           
        self.count += 1
        
        return
    
def example():
    audio_path = sys.argv[1]
    y, sample_rate = sf.read(audio_path)
    y = y.T
    stft_options = dict(size=1024, shift=512) # 分帧长度，分帧偏移
    Y = stft(y, **stft_options, window=np.hanning).transpose(1, 2, 0)
    num_frames, num_freqs, num_channels = Y.shape
    demo = NonStationaryNPSDEstimator(num_freqs, num_channels, 0.03)
    for i in range(num_frames):
        demo.process_frame(Y[i])
        
    img = np.array(demo.p_list)
    plt.subplot(211)
    plt.imshow(np.transpose(img), origin="lower", cmap="magma", aspect="auto", interpolation="none", vmin=0, vmax=1)
    
    plt.subplot(212)
    img = np.array(demo.q_list)
    plt.imshow(np.transpose(img), origin="lower", cmap="magma", aspect="auto", interpolation="none", vmin=0, vmax=1)
    plt.show()
    
    
    return

if __name__ == '__main__':
    example()
    
    
''' 
reference:
[1] : https://github.com/jmh216/FYP-Neural-Beamformer/blob/4cb5287d55c1869443be3c2026dcb768512172a3/MVDR/submodules/sap-elobes-utilities/estnoicov.m
'''