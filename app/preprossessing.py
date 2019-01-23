'''
Created on 19 Jan 2019

@author: anthonyanyanwu
'''
import math
import skvideo.io
import numpy as np
import pandas as pd
from filters import butter_highpass_filter
from settings import ROI_SIZE, BREATHING_CUTOFF_FREQUENCY

RED = 0
GREEN = 1
BLUE = 2

def get_rgb_channels(video_path):
    '''Get Red, Green and Blue channels of the video at the given path.
    '''
    video = skvideo.io.vread(video_path)
    red = video.copy()
    red[:,:,:,1] = 0
    red[:,:,:,2] = 0
    green = video.copy()
    green[:,:,:,0] = 0
    green[:,:,:,2] = 0
    blue = video.copy()
    blue[:,:,:,0] = 0
    blue[:,:,:,1] = 0
    return (red, green, blue)


def get_roi(video):
    xstart = int((video.shape[1] - ROI_SIZE) / 2) 
    xend = int((video.shape[1] + ROI_SIZE) / 2)
    ystart = int((video.shape[2] - ROI_SIZE) / 2) 
    yend = int((video.shape[2] + ROI_SIZE) / 2) 
    return video[:,xstart:xend,ystart:yend,:]


def get_channel_average(video, channel):
    return np.mean(video[:,:,:,channel], axis=(1, 2))


def get_frequency_properties(channel_signal, frequency, sample_rate):
    channel_signal = np.array(channel_signal)
    # if T is period of signal, then sample_rate * T = no_of_frames/cycle = sample_rate / frequency
    frames_per_cycle = int(round(sample_rate / frequency))
    # now get how many full cycles are in the channel_signal
    cycles = int(math.floor(channel_signal.size / frames_per_cycle))
    return channel_signal[:frames_per_cycle * cycles], frames_per_cycle, cycles


def get_nominal_values(channel_signal, frequency, sample_rate):
    '''Returns the nominal values of given channel_signal derived by imposing given frequency on given signal
    
    Parameters
    ----------
    channel_signal: The numpy array derived by getting channel average of a video
    frequency: the frequency which to use for grouping the frame averages
    sample_rate: the sample rate (frames/seconds) of the channel_signal
    
    Typical use case as follows 
    >>> import math
    >>> from skvideo.io import ffprobe
    >>> import numpy as np 
    >>> import pandas as pd
    >>> from app.preprossessing import (get_rgb_channels, get_roi, get_channel_average, get_nominal_values, BLUE)
    >>> from filters import butter_highpass_filter
    >>> from settings import BREATHING_CUTOFF_FREQUENCY
    >>> filepath = 'fixtures/video/phone-cam-plus-incandecent-bulb-1.MOV'
    >>> rs = ffprobe(filepath)['video']['@avg_frame_rate']  
    >>> sample_rate = int(rs.split('/')[0])/int(rs.split('/')[1]) 
    >>> _, _, blue_channel_video = get_rgb_channels(filepath)
    >>> blue_channel = get_channel_average(get_roi(blue_channel_video), BLUE)
    >>> fblue_channel = butter_highpass_filter(blue_channel, BREATHING_CUTOFF_FREQUENCY, sample_rate)
    >>> fforms = np.fft.fft(fblue_channel)
    >>> freqs = np.fft.fftfreq(len(fblue_channel), d=(1/sample_rate))   
    >>> df = pd.DataFrame({'peak': np.abs(fforms), 'freqs': freqs})
    >>> fpeak_max = df.query('freqs > 0')['peak'].max()
    >>> frequency_max = df.query('peak == @fpeak_max and freqs > 0')['freqs'].max()
    >>> blue_nominal_values = get_nominal_values(blue_channel, frequency_max*3/4, sample_rate)
    '''
    channel_signal, frames_per_cycle, cycles = get_frequency_properties(channel_signal, frequency, sample_rate)
    # resize channel_signal as per the number frames per cycle, discarding uncompleted channels
    return np.max(channel_signal.reshape((cycles, frames_per_cycle)), axis=1)


def get_lowest_values(channel_signal, frequency, sample_rate):
    '''Returns the trough values of given channel_signal derived by imposing given frequency on given signal
    
    Parameters
    ----------
    channel_signal: The numpy array derived by getting channel average of a video
    frequency: the frequency which to use for grouping the frame averages
    sample_rate: the sample rate (frames/seconds) of the channel_signal
    
    Typical use case as follows 
    >>> import math
    >>> from skvideo.io import ffprobe
    >>> import numpy as np 
    >>> import pandas as pd
    >>> from app.preprossessing import (get_rgb_channels, get_roi, get_channel_average, get_nominal_values, BLUE)
    >>> from filters import butter_highpass_filter
    >>> from settings import BREATHING_CUTOFF_FREQUENCY
    >>> filepath = 'fixtures/video/phone-cam-plus-incandecent-bulb-1.MOV'
    >>> rs = ffprobe(filepath)['video']['@avg_frame_rate']  
    >>> sample_rate = int(rs.split('/')[0])/int(rs.split('/')[1]) 
    >>> _, _, blue_channel_video = get_rgb_channels(filepath)
    >>> blue_channel = get_channel_average(get_roi(blue_channel_video), BLUE)
    >>> fblue_channel = butter_highpass_filter(blue_channel, BREATHING_CUTOFF_FREQUENCY, sample_rate)
    >>> fforms = np.fft.fft(fblue_channel)
    >>> freqs = np.fft.fftfreq(len(fblue_channel), d=(1/sample_rate))   
    >>> df = pd.DataFrame({'peak': np.abs(fforms), 'freqs': freqs})
    >>> fpeak_max = df.query('freqs > 0')['peak'].max()
    >>> frequency_max = df.query('peak == @fpeak_max and freqs > 0')['freqs'].max()
    >>> blue_lowest_values = get_lowest_values(blue_channel, frequency_max*3/4, sample_rate)
    '''
    channel_signal, frames_per_cycle, cycles = get_frequency_properties(channel_signal, frequency, sample_rate)
    # resize channel_signal as per the number frames per cycle, discarding uncompleted channels
    return np.min(channel_signal.reshape((cycles, frames_per_cycle)), axis=1)


def get_channel_ave_nominal_values(channel_signal, frame_rate, cutoff_frequency=BREATHING_CUTOFF_FREQUENCY, nominal_frequency_adjustment=0.75):
    '''Returns the nominal values for given channel signal. channel_signal is expected to be derived from get_channel_average'''
    filtered_channel = butter_highpass_filter(channel_signal, cutoff_frequency, frame_rate)     # remove frequency related to breathing
    frequency_peaks = np.fft.fft(filtered_channel)      # get fourier transform
    freqencies = np.fft.fftfreq(len(filtered_channel), d=(1/frame_rate))            # list of frequencies
    df = pd.DataFrame({'peak': np.abs(frequency_peaks), 'freqs': freqencies})       
    fpeak_max = df.query('freqs > 0')['peak'].max()                   # left for clearity. Actualy butter_highpass_filter is expected to remove zero freqs
    frequency_max = df.query('peak == @fpeak_max and freqs > 0')['freqs'].max()     # this is estimate of heart beat
    return get_nominal_values(channel_signal, frequency_max*nominal_frequency_adjustment, frame_rate)


def get_channel_ave_trough_values(channel_signal, frame_rate, cutoff_frequency=BREATHING_CUTOFF_FREQUENCY, nominal_frequency_adjustment=0.75):
    '''Returns the nominal values for given channel signal. channel_signal is expected to be derived from get_channel_average'''
    filtered_channel = butter_highpass_filter(channel_signal, cutoff_frequency, frame_rate)     # remove frequency related to breathing
    frequency_peaks = np.fft.fft(filtered_channel)      # get fourier transform
    freqencies = np.fft.fftfreq(len(filtered_channel), d=(1/frame_rate))            # list of frequencies
    df = pd.DataFrame({'peak': np.abs(frequency_peaks), 'freqs': freqencies})       
    fpeak_max = df.query('freqs > 0')['peak'].max()                   # left for clearity. Actualy butter_highpass_filter is expected to remove zero freqs
    frequency_max = df.query('peak == @fpeak_max and freqs > 0')['freqs'].max()     # this is estimate of heart beat
    return get_lowest_values(channel_signal, frequency_max*nominal_frequency_adjustment, frame_rate)
    
    
class VideoAttributes(object):
    
    def __init__(self, filepath):
        ave_frame_rate = skvideo.io.ffprobe(filepath)['video']['@avg_frame_rate'] 
        self.channels = get_rgb_channels(filepath)
        self.frame_rate = int(ave_frame_rate.split('/')[0])/int(ave_frame_rate.split('/')[1]) 
        self.frame_count = len(self.channels[0])     # no reason, just picked red to get the frame size
        red_channel, green_channel, blue_channel = self.channels
        self.red_channel = get_channel_average(get_roi(red_channel), RED)
        self.green_channel = get_channel_average(get_roi(green_channel), GREEN)
        self.blue_channel = get_channel_average(get_roi(blue_channel), BLUE)
        self.nominal_red = get_channel_ave_nominal_values(self.red_channel, self.frame_rate)
        self.nominal_blue = get_channel_ave_nominal_values(self.blue_channel, self.frame_rate)
        self.nominal_green = get_channel_ave_nominal_values(self.green_channel, self.frame_rate)
        self.trough_red = get_channel_ave_trough_values(self.red_channel, self.frame_rate)
        self.trough_blue = get_channel_ave_trough_values(self.blue_channel, self.frame_rate)
        self.trough_green = get_channel_ave_trough_values(self.green_channel, self.frame_rate)
