import resampy
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

def load_audio_task(fname, sample_rate, channels, dtype="float32"):
    if dtype not in ['float64', 'float32', 'int32', 'int16']:
        raise ValueError(f"dtype not supported: {dtype}")

    wav_data, sr = sf.read(fname, dtype=dtype)
    # For integer type PCM input, convert to [-1.0, +1.0]
    if dtype == 'int16':
        wav_data = wav_data / 32768.0
    elif dtype == 'int32':
        wav_data = wav_data / float(2**31)

    # Convert to mono
    assert channels in [1, 2], "channels must be 1 or 2"
    if len(wav_data.shape) > channels:
        wav_data = np.mean(wav_data, axis=1)

    if sr != sample_rate:
        wav_data = resampy.resample(wav_data, sr, sample_rate)

    return wav_data

def add_noise(data, stddev):
  """Adds Gaussian noise to the samples.
  Args:
    data: 1d Numpy array containing floating point samples. Not necessarily
      normalized.
    stddev: The standard deviation of the added noise.
  Returns:
     1d Numpy array containing the provided floating point samples with added
     Gaussian noise.
  Raises:
    ValueError: When data is not a 1d numpy array.
  """
  if len(data.shape) != 1:
    raise ValueError("expected 1d numpy array.")
  max_value = np.amax(np.abs(data))
  num_samples = data.shape[0]
  gauss = np.random.normal(0, stddev, (num_samples)) * max_value
  return data + gauss

def gen_sine_wave(freq=600,
                  length_seconds=6,
                  sample_rate=SAMPLE_RATE,
                  param=None):
  """Creates sine wave of the specified frequency, sample_rate and length."""
  t = np.linspace(0, length_seconds, int(length_seconds * sample_rate))
  samples = np.sin(2 * np.pi * t * freq)
  if param:
      samples = add_noise(samples, param)
  return np.asarray(2**15 * samples, dtype=np.int16)
