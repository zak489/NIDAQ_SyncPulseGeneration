from nidaqmx import Task
from nidaqmx import stream_writers
from nidaqmx import stream_readers
from nidaqmx import constants
import numpy as np
import time
from matplotlib import pyplot as plt


class NIWriter(Task):

    def __init__(self, AO_channels=['/Dev1/ao0', '/Dev1/ao1'], min_val=0., max_val=1., sampling_rate=20, finite_sampling=True):
        Task.__init__(self)
        self.AO_channels = AO_channels
        self.no_AO_channels = len(AO_channels)
        self.min_val = min_val
        self.max_val = max_val
        self.sampling_rate = sampling_rate
        self.finite_sampling = finite_sampling
        self.run_time = 0
        self.pulse_sequence = np.array([np.linspace(0, 1, 100), np.linspace(1, 0, 100)])
        for idx in range(0, self.no_AO_channels):
            self.ao_channels.add_ao_voltage_chan(physical_channel=AO_channels[idx],
                                                 min_val=min_val, max_val=max_val)
        self.streamer = stream_writers.AnalogMultiChannelWriter(self.out_stream, auto_start=True)

    def create_pulse_sequence(self, input_sequence=np.asarray([[100, 0.5], [100, 0], [200, 1.], [500, 0]])):
        """
        Takes as input as a run time encoded pulse sequence and outputs required continuous stream format
        P1 = rm.create_pulse_sequence(input_sequence=[[time,voltage],[100,0.7]])
        rm.stream(pulse_sequence=[P1,np.flip(P1)]
        """
        input_sequence = np.asarray(input_sequence)
        output_sequence = np.zeros(int(input_sequence.sum(axis=0)[0]))
        sequence_start = 0  # perhaps this should be len(self.pulse_seqence)
        for idx in range(0, input_sequence.shape[0]):
            output_sequence[int(sequence_start):int(input_sequence[idx][0] + sequence_start)] = input_sequence[idx][1]
            sequence_start += input_sequence[idx][0]
        return output_sequence

    def voltage_sweep(self, start=0., stop=1., step=0.05, pulse_length=100):
        """
        Useful for finding correct excitation power
        """
        block = [[pulse_length,i] for i in np.arange(start, stop+step, step)]
        return block

    def upload_sequence(self, pulse_sequence):
        self.pulse_sequence = np.asarray(pulse_sequence)
        if self.finite_sampling == True:
            self.run_time = (len(self.pulse_sequence[0])/self.sampling_rate)
            print('expected time ', self.run_time, 's')
            sample_mode = constants.AcquisitionType.FINITE
            self.timing.cfg_samp_clk_timing(rate=self.sampling_rate, sample_mode=sample_mode,
                                            samps_per_chan=len(self.pulse_sequence[0]))
            return self.run_time

    def clear_stream(self):
        # Doesnt seem to work that well...
        self.streamer.write_many_sample([0],[0])

    def stream(self, sleep=False):
        self.streamer.write_many_sample(self.pulse_sequence)
        if sleep == True:
            print('waiting for', self.run_time)
            time.sleep(1.*self.run_time)
            print('finished wait')

    def set_sampling_rate(self, rate=100):
        # Will this actually work after streamer object has been made?
        self.sampling_rate = rate
        self.timing.cfg_samp_clk_timing(rate=rate)

    def continuous_sampling(self):
        # what does this actually do?
        sample_mode = constants.AcquisitionType.CONTINUOUS
        self.timing.cfg_samp_clk_timing(rate=self.sampling_rate, sample_mode=sample_mode)

    def get_channel_names(self):
        return self.channel_names

    def plot_pulse_sequence(self):
        time_axis = np.arange(0, len(self.pulse_sequence[0]))/self.sampling_rate
        line_style = ['b--', 'g:']
        for idx in range(0, self.number_of_channels):
            plt.plot(time_axis, self.pulse_sequence[idx], line_style[idx], label='AO'+str(idx))
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.ylim(0, 1.1*max(self.pulse_sequence[0]))
        plt.title('Pulse sequence')

    def done(self):
        return self.is_task_done()

    def close_devices(self):
        self.close()


class NI_AO(Task):
    def __init__(self,AO_channels=['/Dev1/ao0'], min_val=-10., max_val=10.):
        Task.__init__(self)
        self.AO_channels = AO_channels
        self.min_val = min_val
        self.max_val = max_val
        self.ao_channels.add_ao_voltage_chan(AO_channels[0], min_val=-10., max_val=10.)
        self.start()
    def writer(self,voltage):
        self.write(voltage,auto_start=True)
    def close_devices(self):
        self.close()

class NI_AO_SW(Task):
    def __init__(self,AO_channels=['/Dev1/ao0'], min_val=-10., max_val=10.):
        Task.__init__(self)
        self.AO_channels = AO_channels
        self.min_val = min_val
        self.max_val = max_val
        self.ao_channels.add_ao_voltage_chan(AO_channels[0], min_val=-10., max_val=10.)
        self.streamer = stream_writers.AnalogSingleChannelWriter(self.out_stream, auto_start=True)
        self.start()
    def writer(self,voltage):
        self.streamer.write_one_sample(voltage,timeout=10)

    def close_devices(self):
        self.close()






class NIReader(Task):
    def __init__(self, AI_channels=['/Dev1/ai0'], sampling_rate=50):
        Task.__init__(self)
        self.AI_channels = AI_channels
        self.no_AI_channels = len(AI_channels)
        self.sampling_rate = sampling_rate
        self.number_of_samples = 100
        self.data = np.zeros(0)

        for idx in range(0, self.no_AI_channels):
            self.ai_channels.add_ai_voltage_chan(physical_channel=AI_channels[idx])
        self.reader = stream_readers.AnalogMultiChannelReader(self.in_stream)

    def get_channel_names(self):
        return self.channel_names

    def readout(self, number_of_samples=200):
        self.number_of_samples = number_of_samples

        self.timing.cfg_samp_clk_timing(rate=self.sampling_rate)
        # sample_mode = constants.AcquisitionType.(?) to specify if the task acquires or generates samples
        # active_edge = constants.Edge.RISING        to specify edge counting
        # continuously or if it acquires or generates a finite number of samples.
        self.data = np.zeros((self.no_AI_channels, number_of_samples))
        self.reader.read_many_sample(self.data, number_of_samples_per_channel=number_of_samples)
        return self.data

    def plot_readout(self):
        # Need to have this calculate the readout time depending on the sampling rate as done in writer class
        plt.clf()
        time_axis = np.arange(0, self.number_of_samples)
        line_style = ['r:', 'm--']
        for idx in range(0, self.number_of_channels):
            plt.plot(time_axis, self.data[idx], line_style[idx], label='AI'+str(idx))
        plt.xlabel('time (s)'); plt.ylabel('Voltage (V)')
        plt.legend()

    def close_devices(self):
        self.close()

if __name__ == "__main__":

    # Analogue Input reader
    Reader = NIReader(AI_channels=['/Dev1/ai0'], sampling_rate=10000)
    data = Reader.readout(number_of_samples=1000)
    Reader.plot_readout()
    Reader.close_devices()
    plt.show()
