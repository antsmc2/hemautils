'''
Created on 18 Jan 2019

@author: anthonyanyanwu
'''
import pytest
import tempfile
import numpy as np
import pandas as pd
import skvideo.io
from settings import ROI_SIZE, BREATHING_CUTOFF_FREQUENCY
from app.preprossessing import (get_rgb_channels, get_roi, RED, GREEN, BLUE, get_channel_average, get_nominal_values, 
                                get_lowest_values, get_frequency_properties, get_channel_ave_nominal_values, get_channel_ave_trough_values,
                                VideoAttributes)
from filters import butter_highpass_filter



@pytest.fixture
def dummy_video_path():
    '''Returns Basically return path to a 64x64 RGB video containing 50 frames for test purpose'''
    dummy_video = np.random.randint(0, high=255, size=(50, 64, 64, 3), dtype=np.uint8)
    file = tempfile.NamedTemporaryFile()
    videopath = '%s.mpeg' % file.name
    skvideo.io.vwrite(videopath, dummy_video)
    return videopath
   
    
@pytest.fixture    
def dummy_video():
    '''returns a randomly generated 256x256 RGB video containing 50 frames for test purpose'''
    return np.random.randint(0, high=255, size=(50, 312, 256, 3), dtype=np.uint8)

@pytest.fixture 
def test_video():
    return 'fixtures/video/phone-cam-plus-incandecent-bulb-1.MOV' 


def mix_signal_generator(starting_value, fs, duration, *frequencies):
    nsamples = fs * duration
    t_sine = np.linspace(0, duration, nsamples, endpoint=False)
    y_sine = starting_value      # start with a high static signal
    for frequency in frequencies:
        w = 2. * np.pi * frequency
        y_sine += np.sin(w * t_sine)
    result = pd.DataFrame({'data' : y_sine} , index=t_sine)
    return result


def mix_freq_mod_signal_generator(starting_value, fs, duration, *frequencies):
    '''Same as mix_signal_generator except that each sign wave is having amplitude equal to signal frequency'''
    nsamples = fs * duration
    t_sine = np.linspace(0, duration, nsamples, endpoint=False)
    y_sine = starting_value
    for frequency in frequencies:
        w = 2. * np.pi * frequency
        y_sine += frequency * np.sin(w * t_sine)
    result = pd.DataFrame({'data' : y_sine} , index=t_sine)
    return result


def mix_freq_mod_signal_generator2(starting_value, fs, duration, *frequencies):
    '''Similar to mix_freq_mod_signal_generator except that amplitude is derived from 100/frequency'''
    nsamples = fs * duration
    t_sine = np.linspace(0, duration, nsamples, endpoint=False)
    y_sine = starting_value
    for frequency in frequencies:
        w = 2. * np.pi * frequency
        y_sine += 100/frequency * np.sin(w * t_sine)
    result = pd.DataFrame({'data' : y_sine} , index=t_sine)
    return result


class TestVideoChannelExtraction(object):

    def test_get_rgb_channels_returns_three_channel_of_video(self, dummy_video_path):
        assert len(get_rgb_channels(dummy_video_path)) == 3
        
    def test_get_rgb_channels_returns_three_channels_of_same_size_as_original_video(self, dummy_video_path):
        red, green, blue = get_rgb_channels(dummy_video_path)
        test_video_array = skvideo.io.vread(dummy_video_path)
        assert red.shape == test_video_array.shape
        assert green.shape == test_video_array.shape
        assert blue.shape == test_video_array.shape
        
    def test_get_rgb_channels_returns_first_part_as_red_only_channel(self, dummy_video_path):
        red, _, _ = get_rgb_channels(dummy_video_path)
        test_video_array = skvideo.io.vread(dummy_video_path)
        assert red.shape == test_video_array.shape
        assert np.any(red[:,:,:,0] != 0)
        assert np.all(red[:,:,:,1] == 0)
        assert np.all(red[:,:,:,2] == 0)
        
        
    def test_get_rgb_channels_returns_second_part_as_green_only_channel(self, dummy_video_path):
        _, green, _ = get_rgb_channels(dummy_video_path)
        test_video_array = skvideo.io.vread(dummy_video_path)
        assert green.shape == test_video_array.shape
        assert np.all(green[:,:,:,0] == 0)
        assert np.any(green[:,:,:,1] != 0)
        assert np.any(green[:,:,:,2] == 0)
        
    def test_get_rgb_channels_returns_second_part_as_blue_only_channel(self, dummy_video_path):
        _, _, blue = get_rgb_channels(dummy_video_path)
        test_video_array = skvideo.io.vread(dummy_video_path)
        assert blue.shape == test_video_array.shape
        assert np.all(blue[:,:,:,0] == 0)
        assert np.any(blue[:,:,:,1] == 0)
        assert np.any(blue[:,:,:,2] != 0)
        
        
class TestExtractRegionOfInterest(object):
     
    def test_get_roi_returns_frames_with_size_according_to_settings_ROI(self, dummy_video):
        roi_video = get_roi(dummy_video)
        assert roi_video[1].shape == (ROI_SIZE, ROI_SIZE, 3)
    
    def test_get_roi_returns_centered_frames_for_each_frame(self, dummy_video):
        roi_video = get_roi(dummy_video) 
        #confirm the boxes at the 4 corners at ROI
        xstart_corner = int((dummy_video.shape[1] - ROI_SIZE) / 2) 
        xend_corner = int((dummy_video.shape[1] + ROI_SIZE) / 2)
        ystart_corner = int((dummy_video.shape[2] - ROI_SIZE) / 2) 
        yend_corner = int((dummy_video.shape[2] + ROI_SIZE) / 2) 
        assert np.all(dummy_video[:, xstart_corner, ystart_corner] == roi_video[:, 0, 0])
        assert np.all(dummy_video[:, xstart_corner, yend_corner-1] == roi_video[:, 0, ROI_SIZE-1])
        assert np.all(dummy_video[:, xend_corner-1, yend_corner-1] == roi_video[:, ROI_SIZE-1, ROI_SIZE-1])
        assert np.all(dummy_video[:, xend_corner-1, ystart_corner] == roi_video[:, ROI_SIZE-1, 0])
        
    def test_get_channel_average_returns_series_of_average_values_of_each_video_frame_on_red_channel(self, dummy_video):
        red_frames = get_channel_average(dummy_video, RED)
        assert np.all(np.mean(dummy_video[:,:,:,RED], axis=(1, 2)) == red_frames)
        
    def test_get_channel_average_returns_series_of_average_values_of_each_video_frame_on_green_channel(self, dummy_video):
        green_frames = get_channel_average(dummy_video, GREEN)
        assert np.all(np.mean(dummy_video[:,:,:,GREEN], axis=(1, 2)) == green_frames)
        
    def test_get_channel_average_returns_series_of_average_values_of_each_video_frame_on_blue_channel(self, dummy_video):
        blue_frames = get_channel_average(dummy_video, BLUE)
        assert np.all(np.mean(dummy_video[:,:,:,BLUE], axis=(1, 2)) == blue_frames)
        
        
class TestFilters(object):
    
    def test_signal_after_higpass_filter_leads_to_cutoff_signal_not_exceeding_3_percent(self):
        duration = 50   # 50 secs
        sample_rate = 100       # 100 samples per sec
        cutoff = BREATHING_CUTOFF_FREQUENCY            # basically 0.5hz to test scenario of filtering noise due to breathing (with has frequency of about 0.2 to 0.3hz) 
        mixed_signal = mix_signal_generator(500, sample_rate, duration, 0.3, 5, 10, 15)      # create mixed signal of 0.3hz, 5hz, 10hz and 15hz   
        # assert via fourier transform presence of each signal
        fforms = np.fft.fft(mixed_signal.data)
        freqs = np.fft.fftfreq(len(mixed_signal), d=(1/sample_rate))   
        df = pd.DataFrame({'peak': np.abs(fforms), 'freqs': freqs})
        fpeak_max = df.query('freqs > 0')['peak'].max()         # need to exclude unwanted 0 frequency
        assert len(df.query('peak > (@fpeak_max * 0.99) and freqs == 0.3')) == 1
        assert len(df.query('peak > (@fpeak_max * 0.99) and freqs == 5')) == 1
        assert len(df.query('peak > (@fpeak_max * 0.99) and freqs == 10')) == 1
        assert len(df.query('peak > (@fpeak_max * 0.99) and freqs == 15')) == 1
        filtered = butter_highpass_filter(mixed_signal.data, cutoff, sample_rate) 
        # now confirm after applying filter
        fforms_filtered = np.fft.fft(filtered)
        freqs_filtered = np.fft.fftfreq(len(fforms_filtered), d=(1/sample_rate))  
        df_filtered = pd.DataFrame({'peak': np.abs(fforms_filtered), 'freqs': freqs_filtered})
        # confirm no cutoff frequency component has peak about 5%
        assert len(df_filtered.query('peak >= (@fpeak_max * 0.03) and abs(freqs) <= 0.3')) == 0     
        
    def test_signal_after_higpass_filter_leads_to_pass_band_signal_above_97_percent(self):
        duration = 50   # 50 secs
        sample_rate = 100       # 100 samples per sec
        cutoff = 0.5            # basically 0.5hz to test scenario of filtering noise due to breathing (with has frequency of about 0.2 to 0.3hz) 
        mixed_signal = mix_signal_generator(300, sample_rate, duration, 0.3, 5, 10, 15)      # create mixed signal of 0.3hz, 5hz, 10hz and 15hz   
        # assert via fourier transform presence of each signal
        fforms = np.fft.fft(mixed_signal.data)
        freqs = np.fft.fftfreq(len(mixed_signal), d=(1/sample_rate))   
        df = pd.DataFrame({'peak': np.abs(fforms), 'freqs': freqs})
        fpeak_max = df.query('freqs > 0')['peak'].max()     # need to exclude unwanted 0 frequency
        assert len(df.query('peak > (@fpeak_max * 0.99) and freqs == 0.3')) == 1
        assert len(df.query('peak > (@fpeak_max * 0.99) and freqs == 5')) == 1
        assert len(df.query('peak > (@fpeak_max * 0.99) and freqs == 10')) == 1
        assert len(df.query('peak > (@fpeak_max * 0.99) and freqs == 15')) == 1
        filtered = butter_highpass_filter(mixed_signal.data, cutoff, sample_rate) 
        # now confirm after applying filter
        fforms_filtered = np.fft.fft(filtered)
        freqs_filtered = np.fft.fftfreq(len(fforms_filtered), d=(1/sample_rate))  
        df_filtered = pd.DataFrame({'peak': np.abs(fforms_filtered), 'freqs': freqs_filtered}) 
        # confirm the pass band filter does not depreciate below 85% 
        assert len(df_filtered.query('peak > (@fpeak_max * 0.97)  and freqs == 5')) == 1
        assert len(df_filtered.query('peak > (@fpeak_max * 0.97)  and freqs == 10')) == 1
        assert len(df_filtered.query('peak > (@fpeak_max * 0.97)  and freqs == 15')) == 1   
        
    def test_get_frequency_properties_returns_expected_frames_per_cycle_and_no_of_cycles(self):
        duration = 50
        sample_rate = 100
        test_frequency = 20
        mixed_signal = mix_freq_mod_signal_generator(700, sample_rate, duration, int(test_frequency/4), int(test_frequency/2), test_frequency)
        mmixed_signal, frames_per_cycle, cycles = get_frequency_properties(mixed_signal, test_frequency, sample_rate)
        assert frames_per_cycle == round(sample_rate/test_frequency)
        assert cycles == mmixed_signal.size / frames_per_cycle
        
    def test_peak_extractor_returns_correct_nominal_peaks_within_each_given_frequency(self):
        #first create sign waves having multiple frequencies and multiple peaks
        duration = 50
        sample_rate = 100
        zero_peak = 200
        mixed_signal = mix_freq_mod_signal_generator(zero_peak, sample_rate, duration, 5, 10, 20)
        peak_values, _ = get_nominal_values(mixed_signal, 20, sample_rate)
        df = pd.DataFrame({'peak': peak_values})
        assert len(df.query('peak > (@zero_peak + 5) and peak < (@zero_peak + 10)')) == 5 * duration      # there should be 5 peaks meeting this criteria per cycle
        assert len(df.query('peak > (@zero_peak + 10) and peak < (@zero_peak + 20)')) == 5 * duration     # there should be 5 peaks meeting this criteria per cycle
        assert len(df.query('peak > (@zero_peak + 20) and peak < (@zero_peak + 25)')) == 5 * duration
        assert len(df.query('peak > (@zero_peak + 25) and peak < (@zero_peak + 30)')) == 5 * duration
        
    def test_nominal_peak_return_frame_ids_in_sync_with_peaks(self):
        #first create sign waves having multiple frequencies and multiple peaks
        duration = 50
        sample_rate = 100
        zero_peak = 200
        mixed_signal = mix_freq_mod_signal_generator(zero_peak, sample_rate, duration, 5, 10, 20)
        mixed_signal = np.array(mixed_signal.data)       # to enable sequential indexing
        peak_values, frame_ids = get_nominal_values(mixed_signal, 20, sample_rate)
        assert len(peak_values) == len(frame_ids)
        for index, id in enumerate(frame_ids):
            assert mixed_signal[id] == peak_values[index]        
            
    def test_trough_peak_return_frame_ids_in_sync_with_peaks(self):
        #first create sign waves having multiple frequencies and multiple peaks
        duration = 50
        sample_rate = 100
        zero_peak = 200
        mixed_signal = mix_freq_mod_signal_generator(zero_peak, sample_rate, duration, 5, 10, 20)
        mixed_signal = np.array(mixed_signal.data)       # to enable sequential indexing
        trough_values, frame_ids = get_lowest_values(mixed_signal, 20, sample_rate)
        assert len(trough_values) == len(frame_ids)
        for index, id in enumerate(frame_ids):
            assert mixed_signal[id] == trough_values[index]      
        
    def test_trough_extractor_returns_correct_nominal_troughs_within_each_given_frequency(self):
        #first create sign waves having multiple frequencies and multiple peaks
        duration = 50
        sample_rate = 100
        zero_peak = 100
        mixed_signal = mix_freq_mod_signal_generator(zero_peak, sample_rate, duration, 5, 10, 20)
        trough_values, _ = get_lowest_values(mixed_signal, 20, sample_rate)
        df = pd.DataFrame({'trough': trough_values})
        assert len(df.query('trough < (@zero_peak - 5) and trough > (@zero_peak - 10)')) == 5 * duration      # there should be 5 peaks meeting this criteria per cycle
        assert len(df.query('trough < (@zero_peak - 10) and trough > (@zero_peak - 20)')) == 5 * duration     # there should be 5 peaks meeting this criteria per cycle
        assert len(df.query('trough < (@zero_peak - 20) and trough > (@zero_peak - 25)')) == 5 * duration
        assert len(df.query('trough < (@zero_peak - 25) and trough > (@zero_peak - 30)')) == 5 * duration    
     
    def test_get_channel_nominal_values_returns_nominal_values_for_given_channel_with_filtered_frequency(self):
        duration = 50
        sample_rate = 100
        zero_peak = 250
        mixed_signal = mix_freq_mod_signal_generator2(zero_peak, sample_rate, duration, 5, 10, 20)
        peak_values, _ = get_channel_ave_nominal_values(mixed_signal.data, sample_rate, 7, nominal_frequency_adjustment=1)      
        df = pd.DataFrame({'peak': peak_values})
        # Since 5hz is filtered off, the next nominal frequency is 10Hz. 
        # The filtered wave form we should have only one peak within peak > (@zero_peak + 10) and peak < (@zero_peak + 20) every 0.1 seconds.
        # While this is true, applying 10hz on original wave yeilds two different peaks every 0.2 seconds. 
        # one peak within first 0.1 seconds  with criteria peak >= (@zero_peak + 20)
        # second in next 0.1 seconds  with criteria peak >= @zero_peak and peak <= (@zero_peak + 10)
        assert len(df.query('peak > (@zero_peak + 10) and peak < (@zero_peak + 20)')) == 0    
        assert len(df.query('peak >= @zero_peak and peak <= (@zero_peak + 10)')) == 5 * duration     # there should be 5 peaks meeting this criteria per cycle
        assert len(df.query('peak >= (@zero_peak + 20)')) == 5 * duration 
        
    def test_get_channel_trough_values_returns_max_trough_values_for_given_channel_with_filtered_frequency(self):
        duration = 50
        sample_rate = 100
        zero_peak = 250
        mixed_signal = mix_freq_mod_signal_generator2(zero_peak, sample_rate, duration, 5, 10, 20)
        trough_values, _ = get_channel_ave_trough_values(mixed_signal.data, sample_rate, 7, nominal_frequency_adjustment=1)      
        df = pd.DataFrame({'trough': trough_values})
        # Similar to get_channel_ave_nominal_values  but inverted.
        assert len(df.query('trough < (@zero_peak - 10) and trough > (@zero_peak - 20)')) == 0    
        assert len(df.query('trough < @zero_peak and trough > (@zero_peak - 10)')) == 5 * duration     # there should be 5 peaks meeting this criteria per cycle
        assert len(df.query('trough < (@zero_peak - 20)')) == 5 * duration 
        
    def test_video_attributes_video_value_is_same_as_get_rgb_channels(self, test_video):
        red, green, blue = get_rgb_channels(test_video)
        video_attributes = VideoAttributes(test_video)
        assert np.all(video_attributes.channels[0] == red)
        assert np.all(video_attributes.channels[1] == green)
        assert np.all(video_attributes.channels[2] == blue)
        
    def test_initiate_video_attributes_returns_frame_count_same_as_each_channel(self, test_video):
        video_attributes = VideoAttributes(test_video)
        red, green, blue = video_attributes.channels
        assert video_attributes.frame_count == len(red)
        assert video_attributes.frame_count == len(green)
        assert video_attributes.frame_count == len(blue)
        
    def test_initiate_video_attributes_instances_frame_rate_value(self, test_video):
        ave_frame_rate = skvideo.io.ffprobe(test_video)['video']['@avg_frame_rate'] 
        frame_rate = int(ave_frame_rate.split('/')[0])/int(ave_frame_rate.split('/')[1]) 
        video_attributes = VideoAttributes(test_video)
        assert round(frame_rate) == round(video_attributes.frame_rate)
        
    def test_initiate_video_attributes_has_all_channel_averages_available(self, test_video): 
        video_attributes = VideoAttributes(test_video)
        red, green, blue = video_attributes.channels
        assert np.all(video_attributes.red_channel == get_channel_average(get_roi(red), RED))
        assert np.all(video_attributes.green_channel == get_channel_average(get_roi(green), GREEN))
        assert np.all(video_attributes.blue_channel == get_channel_average(get_roi(blue), BLUE))
        
    def test_initiate_video_attributes_has_correct_channel_nominal_values_available(self, test_video):
        video_attributes = VideoAttributes(test_video)
        red, green, blue = video_attributes.channels
        frame_rate = video_attributes.frame_rate
        assert np.all(video_attributes.nominal_red.peak == get_channel_ave_nominal_values(get_channel_average(get_roi(red), RED), frame_rate)[0])
        assert np.all(video_attributes.nominal_green.peak == get_channel_ave_nominal_values(get_channel_average(get_roi(green), GREEN), frame_rate)[0])
        assert np.all(video_attributes.nominal_blue.peak == get_channel_ave_nominal_values(get_channel_average(get_roi(blue), BLUE), frame_rate)[0])
            
    def test_initiate_video_attributes_has_correct_channel_through_values_available(self, test_video):
        video_attributes = VideoAttributes(test_video)
        red, green, blue = video_attributes.channels
        frame_rate = video_attributes.frame_rate
        assert np.all(video_attributes.trough_red.trough == get_channel_ave_trough_values(get_channel_average(get_roi(red), RED), frame_rate)[0])
        assert np.all(video_attributes.trough_green.trough == get_channel_ave_trough_values(get_channel_average(get_roi(green), GREEN), frame_rate)[0])
        assert np.all(video_attributes.trough_blue.trough == get_channel_ave_trough_values(get_channel_average(get_roi(blue), BLUE), frame_rate)[0])
        
    
        

        
        
        