# Audio Signal Processing and Transformation

This Python project demonstrates digital signal processing concepts through audio manipulation using Fourier Transform. It provides tools for recording, analyzing, filtering, and visualizing audio signals in both time and frequency domains.

## 🎯 Features

- **Audio Input**
  - Record audio from microphone (5 seconds duration)
  - Load existing WAV files
  
- **Signal Processing**
  - Fast Fourier Transform (FFT) for frequency analysis
  - Low-pass filtering for noise reduction
  - Signal normalization
  
- **Visualization**
  - Time domain waveform display
  - Frequency spectrum analysis
  - Filtered signal comparison
  
- **Output Options**
  - Save original audio
  - Save filtered audio
  - Compare before/after results

## 🧮 Mathematical Concepts

The project implements several key digital signal processing concepts:

1. **Fourier Transform (FFT)**
   ```
   X(k) = Σ[n=0 to N-1] x(n) * e^(-j2πkn/N)
   ```
   Converts time-domain signal to frequency domain

2. **Frequency Calculation**
   ```
   f = k * fs/N
   ```
   Where:
   - f: frequency in Hz
   - fs: sampling rate (44100 Hz)
   - N: number of samples
   - k: frequency bin index

3. **Low-Pass Filter**
   ```
   H(f) = 1, if |f| ≤ fc
   H(f) = 0, if |f| > fc
   ```
   Where fc is the cutoff frequency

## 🛠️ Requirements

- Python 3.7 or higher
- Required packages:
  ```
  numpy      # For numerical computations
  scipy      # For signal processing
  matplotlib # For visualization
  sounddevice# For audio recording
  ```

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Create and activate virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. Run the main script:
   ```bash
   python audio_transform.py
   ```

2. Choose your input method:
   - Option 1: Record new audio (5 seconds)
   - Option 2: Load existing WAV file

3. View the results:
   - Time domain plot (amplitude vs. time)
   - Frequency spectrum (magnitude vs. frequency)
   - Filtered signal comparison

4. Optionally save the results:
   - Original audio as WAV
   - Filtered audio as WAV

## 📊 Example Output

The program generates three visualization plots:

1. **Time Domain Plot**
   - Shows how the audio signal amplitude changes over time
   - X-axis: Time (seconds)
   - Y-axis: Amplitude

2. **Frequency Domain Plot**
   - Shows the frequency components present in the signal
   - X-axis: Frequency (Hz)
   - Y-axis: Magnitude

3. **Filtered Signal Plot**
   - Shows the audio signal after applying the low-pass filter
   - X-axis: Time (seconds)
   - Y-axis: Amplitude

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📧 Contact

For questions and feedback, please open an issue in the GitHub repository. 