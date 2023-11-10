from absl import app
from absl import flags
from absl import logging
from functools import partial
from functools import wraps
import numpy as np
import queue
import sounddevice as sd
from time import time as now
import whisper
import speech_recognition as sr
from datetime import datetime, timedelta
from time import sleep

audioQueue = np.array([],dtype=np.float32)
lastSilence = np.array([],dtype=np.float32)
lastProcessTime = 0
silenceCnt = 0

FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'base.en',
                    'The version of the OpenAI Whisper model to use.')
flags.DEFINE_string('input_device', 'plughw:2,0',
                    'The input device used to record audio.')
flags.DEFINE_integer('sample_rate', 16000,
                     'The sample rate of the recorded audio.')
flags.DEFINE_integer('phrase_timeout', 3,
                     'How much empty space between recording before inserting a break.')
flags.DEFINE_integer('record_timeout', 2,
                     'How ral time the recording is in seconds.')
flags.DEFINE_integer('energy_threshold', 1000, 
                     'Energy level for mic to detect speeach.')

# A decorator to log the timing of performance-critical functions.
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = now()
        result = func(*args, **kwargs)
        stop = now()
        logging.debug(f'{func.__name__} took {stop-start:.3f}s')
        return result
    return wrapper

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte,dtype=np.int16), dtype='float32')

@timed
def transcribe(model, audio):
    # Run the Whisper model to transcribe the audio chunk.
    # print('audio=', len(audio))
    # dt = np.frombuffer(audio, dtype=np.int16) #float32)
    #dt_s16 = np.frombuffer(audio, dtype=np.int16, count=len(bytes)//2, offset=0)
    #dt_f = data_s16 * 0.5**15
    dt_f = byte_to_float(audio)
    # print(dt_f)
    result = whisper.transcribe(model=model, audio=dt_f)

    # Use the transcribed text.
    text = result['text'].strip()
    # logging.info(text)
    # print(text)
    return text


@timed
def stream_callback(_, audio:sr.AudioData, audio_queue: queue.Queue) -> None:
    data = audio.get_raw_data()
    audio_queue.put(data)
    
@timed
def hasVoice(audio):
    amin = min(audio)
    amax = max(audio)
    diff = abs(amin) + abs(amax)
    return (diff > 0.1)

# @timed
# def process_audio(audio_queue, model):
#     global silenceCnt 
#     global audioQueue
#     global lastProcessTime
#     global lastSilence
#     # Block until the next chunk of audio is available on the queue.
#     audio = audio_queue.get()

#     # check if this audio has data that is not silence
#     if hasVoice(audio):
#         if silenceCnt >= 1:
#             audioQueue = np.concatenate((audioQueue, lastSilence))
#         audioQueue = np.concatenate((audioQueue, audio))
#         silenceCnt = 0
#     else:
#         silenceCnt += 1
#         lastSilence = audio
#     lastProcessTime += 1

#     # check if it's been 10s or if I got consecutive 2 sec of silence 
#     if lastProcessTime >= 10 or (silenceCnt >= 1 and len(audioQueue) > 0):
#         # print(f'Processing at time :{lastProcessTime} silence:{silenceCnt}')
#         lastProcessTime = 0
#         silenceCnt = 0
#         # Transcribe the latest audio chunk.
#         transcribe(model=model, audio=audioQueue)
#         audioQueue = np.array([],dtype=np.float32)

#     # Transcribe the latest audio chunk.
#     # transcribe(model=model, audio=audio)


def main(argv):
    # Load the Whisper model into memory, downloading first if necessary.

    # The first run of the model is slow (buffer init), so run it once empty.
#    logging.info('Warming model up...')
#    block_size = FLAGS.chunk_seconds * FLAGS.sample_rate
#    whisper.transcribe(model=model,
#                       audio=np.zeros(block_size, dtype=np.float32))

    # Stream audio chunks into a queue and process them from there. The
    # callback is running on a separate thread.
    logging.info('Starting stream...')
    audio_queue = queue.Queue()


    recorder = sr.Recognizer()
    recorder.energy_threshold = FLAGS.energy_threshold
    recorder.dynamic_energy_threshold = False


    mic_name = FLAGS.input_device 
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")
        return
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=FLAGS.sample_rate, device_index=index)
                break

    record_timeout = FLAGS.record_timeout
    phrase_timeout = FLAGS.phrase_timeout


    with source:
        recorder.adjust_for_ambient_noise(source)

        
    logging.info(f'Loading model "{FLAGS.model_name}"...')
    model = whisper.load_model(name=FLAGS.model_name)
    logging.info('Warming model up...')
    block_size = 2 * FLAGS.sample_rate
    whisper.transcribe(model=model,
                       audio=np.zeros(block_size, dtype=np.float32))


    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    callback = partial(stream_callback, audio_queue=audio_queue)
    recorder.listen_in_background(source, callback, phrase_time_limit=record_timeout)
    phrase_time = None
    transcription = ['']
    last_sample = bytes()
    print("Starting...")
    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not audio_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not audio_queue.empty():
                    data = audio_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                # audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                # wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                # with open(temp_file, 'w+b') as f:
                #    f.write(wav_data.read())
                # print("Calling Transcribe with datasize ", len(last_sample))
                text = transcribe(model, last_sample)
                # Read the transcription.
                #result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                #text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                
                # Clear the console to reprint the updated transcription.
                # os.system('cls' if os.name=='nt' else 'clear')
                TM="[" + datetime.now().strftime('%H:%M:%S>') + "] "
                for line in transcription:
                    print(TM + line)
                # Flush stdout.
                print('', end='', flush=True)
                transcription = ['']

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break                

if __name__ == '__main__':
    app.run(main)
