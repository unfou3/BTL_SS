# Required libraries for audio processing and visualization
import numpy as np          # For numerical computations and array operations
import sounddevice as sd    # For recording audio from microphone
import matplotlib.pyplot as plt  # For creating visualizations
from scipy.fft import fft, fftfreq, ifft  # For frequency analysis (FFT) and synthesis (IFFT)
from scipy.io import wavfile    # For reading/writing WAV files
import time
import os

class AudioTransform:
    def __init__(self):
        """Initialize audio parameters"""
        # Standard CD-quality sampling rate (44.1 kHz = 44100 samples per second)
        self.sample_rate = 44100  
        # Recording duration in seconds
        self.duration = 5  
        
    def record_audio(self):
        """Record audio from microphone
        
        Returns:
            numpy.ndarray: Recorded audio data as 1D array of int16 values
        """
        print("Recording...")
        print("Please make some sound...")
        
        # Calculate total samples needed (sample_rate * duration)
        # Use int16 for 16-bit audio resolution (-32768 to 32767)
        recording = sd.rec(int(self.duration * self.sample_rate),
                         samplerate=self.sample_rate,
                         channels=1,          # Mono recording
                         dtype=np.int16)      # 16-bit audio
        
        # Wait until recording is complete
        sd.wait()
        print("Recording finished")
        
        # Convert to float for processing
        recording = recording.astype(float)
        
        # Amplify the signal (increase by 5x)
        amplification = 5.0
        recording = recording * amplification
        
        # Prevent clipping by normalizing if amplitude exceeds maximum
        max_possible = np.iinfo(np.int16).max
        if np.max(np.abs(recording)) > max_possible:
            recording = recording * (max_possible / np.max(np.abs(recording)))
        
        # Print debug information
        print(f"Original max amplitude: {np.max(np.abs(recording/amplification))}")
        print(f"Amplified max amplitude: {np.max(np.abs(recording))}")
        print(f"Recording shape: {recording.shape}")
        
        # Convert back to int16
        recording = recording.astype(np.int16)
        
        # Convert 2D array to 1D and return
        return recording.flatten()

    def load_audio(self, filename):
        """Load audio from WAV file
        
        Args:
            filename (str): Path to WAV file
            
        Returns:
            numpy.ndarray: Audio data from file
        """
        # Read WAV file and get sample rate and data
        sample_rate, data = wavfile.read(filename)
        self.sample_rate = sample_rate
        return data

    def apply_fft(self, audio_data):
        """Apply Fast Fourier Transform to convert audio to frequency domain
        
        Args:
            audio_data (numpy.ndarray): Input audio time-domain data
            
        Returns:
            tuple: (frequencies, magnitudes, fft_data)
                - frequencies: The frequency values for each FFT bin
                - magnitudes: The magnitude of each frequency component
                - fft_data: Complex FFT results
        """
        # Store original data type for later use
        self.original_dtype = audio_data.dtype
        
        # Normalize audio to range [-1, 1]
        audio_data = audio_data.astype(float) / np.iinfo(np.int16).max
        
        # Calculate FFT
        n = len(audio_data)
        fft_data = fft(audio_data)  # Convert to frequency domain
        
        # Calculate frequency bins
        # fftfreq returns the frequencies corresponding to FFT results
        freq = fftfreq(n, 1 / self.sample_rate)
        
        # Get positive frequencies only (negative frequencies are mirror images)
        pos_mask = freq >= 0
        freqs = freq[pos_mask]
        
        # Calculate magnitude spectrum
        # 2.0/n normalizes the magnitude
        # np.abs gets magnitude of complex FFT values
        magnitude = 2.0/n * np.abs(fft_data[pos_mask])
        
        return freqs, magnitude, fft_data

    def reduce_noise(self, audio_data, noise_threshold=0.1):
        """Reduce noise by applying a threshold in frequency domain
        
        Args:
            audio_data (numpy.ndarray): Input audio data
            noise_threshold (float): Threshold for noise reduction (0.0 to 1.0)
            
        Returns:
            numpy.ndarray: Noise-reduced audio data
        """
        # Convert to float and normalize
        audio_normalized = audio_data.astype(float) / np.iinfo(np.int16).max
        
        # Calculate FFT
        n = len(audio_normalized)
        fft_data = fft(audio_normalized)
        
        # Apply threshold to remove noise
        magnitude = np.abs(fft_data)
        max_magnitude = np.max(magnitude)
        mask = magnitude > (max_magnitude * noise_threshold)
        fft_filtered = fft_data * mask
        
        # Convert back to time domain
        clean_audio = np.real(ifft(fft_filtered))
        
        # Scale back to int16 range
        clean_audio = clean_audio * np.iinfo(np.int16).max
        return clean_audio.astype(np.int16)

    def apply_lowpass_filter(self, audio_data, cutoff_freq=1000):
        """Apply low-pass filter to remove high frequencies
        
        Args:
            audio_data (numpy.ndarray): Input audio data
            cutoff_freq (int): Cutoff frequency in Hz (default: 1000 Hz)
            
        Returns:
            numpy.ndarray: Filtered audio data
        """
        # Convert to float and keep original amplitude for later scaling
        max_amplitude = np.max(np.abs(audio_data))
        audio_normalized = audio_data.astype(float) / np.iinfo(np.int16).max
        
        # Calculate FFT
        n = len(audio_normalized)
        fft_data = fft(audio_normalized)
        freq = fftfreq(n, 1 / self.sample_rate)
        
        # Create low-pass filter mask
        # Keep frequencies below cutoff_freq, remove others
        mask = np.abs(freq) <= cutoff_freq
        fft_filtered = fft_data * mask  # Apply filter in frequency domain
        
        # Convert back to time domain using Inverse FFT
        filtered_audio = np.real(ifft(fft_filtered))
        
        # Scale back to original amplitude range with additional amplification
        amplification = 3.0  # Increase filtered audio volume
        filtered_audio = filtered_audio * max_amplitude * amplification
        
        # Ensure output is properly scaled and prevent clipping
        max_possible = np.iinfo(np.int16).max
        if np.max(np.abs(filtered_audio)) > max_possible:
            filtered_audio = filtered_audio * (max_possible / np.max(np.abs(filtered_audio)))
        
        print(f"Filtered audio max amplitude: {np.max(np.abs(filtered_audio))}")
        
        return filtered_audio.astype(np.int16)

    def plot_results(self, time_data, freq_data, magnitude, filtered_data=None):
        """Plot audio data in time and frequency domains
        
        Args:
            time_data (numpy.ndarray): Original audio data
            freq_data (numpy.ndarray): Frequency values
            magnitude (numpy.ndarray): Magnitude of frequency components
            filtered_data (numpy.ndarray, optional): Filtered audio data
        """
        # Set up figure size based on number of plots needed
        if filtered_data is not None:
            plt.figure(figsize=(12, 8))
            n_plots = 3
        else:
            plt.figure(figsize=(12, 6))
            n_plots = 2
        
        # Plot 1: Original Time Domain Signal
        plt.subplot(n_plots, 1, 1)
        time_points = np.linspace(0, len(time_data)/self.sample_rate, len(time_data))
        plt.plot(time_points, time_data)
        plt.title('Original Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot 2: Frequency Domain (Spectrum)
        plt.subplot(n_plots, 1, 2)
        plt.plot(freq_data, magnitude)
        plt.title('Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 5000)  # Limit view to 0-5kHz for better visualization
        
        # Plot 3: Filtered Time Domain Signal (if available)
        if filtered_data is not None:
            plt.subplot(n_plots, 1, 3)
            plt.plot(time_points, filtered_data)
            plt.title('Filtered Time Domain')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
        
        plt.tight_layout()  # Adjust spacing between plots
        plt.show()

    def save_audio(self, audio_data, filename):
        """Save audio data to WAV file
        
        Args:
            audio_data (numpy.ndarray): Audio data to save
            filename (str): Output filename
        """
        # Convert to float for processing
        audio_data = audio_data.astype(float)
        
        # Ensure maximum amplitude without clipping
        max_possible = np.iinfo(np.int16).max
        if np.max(np.abs(audio_data)) > max_possible:
            audio_data = audio_data * (max_possible / np.max(np.abs(audio_data)))
        
        # Convert back to int16
        audio_data = audio_data.astype(np.int16)
        
        # Print debug information
        print(f"Saving audio with max amplitude: {np.max(np.abs(audio_data))}")
        
        # Write WAV file
        wavfile.write(filename, self.sample_rate, audio_data)
        print(f"Audio file saved successfully to {filename}")
        print(f"File size: {os.path.getsize(filename)} bytes")

def main():
    """Main function to run the audio processing program"""
    # Create instance of AudioTransform
    transformer = AudioTransform()
    
    # Get user input for audio source
    choice = input("Enter 1 to record audio, 2 to load from file: ")
    
    # Record or load audio based on user choice
    if choice == '1':
        audio_data = transformer.record_audio()
    elif choice == '2':
        filename = input("Enter the path to your WAV file: ")
        audio_data = transformer.load_audio(filename)
    else:
        print("Invalid choice")
        return

    # Process the audio:
    # 1. Apply FFT for frequency analysis
    freqs, magnitude, fft_data = transformer.apply_fft(audio_data)
    
    # 2. Apply low-pass filter
    filtered_audio = transformer.apply_lowpass_filter(audio_data, cutoff_freq=2000)  # Increased cutoff frequency
    
    # 3. Display visualizations
    transformer.plot_results(audio_data, freqs, magnitude, filtered_audio)
    
    # 4. Save audio files if requested
    if input("Save audio files? (y/n): ").lower() == 'y':
        timestamp = int(time.time())
        original_filename = f"original_audio_{timestamp}.wav"
        filtered_filename = f"filtered_audio_{timestamp}.wav"
        
        # Save both original and filtered versions
        transformer.save_audio(audio_data, original_filename)
        transformer.save_audio(filtered_audio, filtered_filename)
        print("\nYou can now compare:")
        print(f"1. Original audio: {original_filename}")
        print(f"2. Filtered audio: {filtered_filename}")

# Entry point of the program
if __name__ == "__main__":
    main() 