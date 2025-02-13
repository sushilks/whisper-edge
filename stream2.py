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
import torch
import audioop

audioQueue = np.array([],dtype=np.float32)
lastSilence = np.array([],dtype=np.float32)
lastProcessTime = 0
silenceCnt = 0
pdev = torch.device("cuda:0")
FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'base.en',
                    'The version of the OpenAI Whisper model to use.')
flags.DEFINE_string('language', 'en',
                    'The language to use or empty to auto-detect.')
flags.DEFINE_string('input_device', 'plughw:2,0',
                    'The input device used to record audio.')
flags.DEFINE_integer('sample_rate', 16000,
                     'The sample rate of the recorded audio.')
flags.DEFINE_integer('num_channels', 1,
                     'The number of channels of the recorded audio.')
flags.DEFINE_integer('channel_index', 0,
                     'The index of the channel to use for transcription.')
flags.DEFINE_integer('chunk_seconds', 10,
                     'The length in seconds of each recorded chunk of audio.')
flags.DEFINE_string('latency', 'low', 'The latency of the recording stream.')


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


@timed
def transcribe(model, audio):
    # Run the Whisper model to transcribe the audio chunk.
    audio = torch.from_numpy(audio).to(pdev)
    result = whisper.transcribe(model=model, audio=audio)

    # Use the transcribed text.
    text = result['text'].strip()
    # logging.info(text)
    print(text)


@timed
def stream_callback(indata, frames, time, status, audio_queue):
    if status:
        logging.warning(f'Stream callback status: {status}')

    # Add this chunk of audio to the queue.
    audio = indata[:, FLAGS.channel_index].copy()
    # data = audioop.ratecv(audio, 1, 1, FLAGS.sample_rate, 16000, None )
    #data = np.array([], dtype=np.float32)
    #crate = FLAGS.sample_rate / 16000
    #i = 0
    #j = 0.0
    #while j < len(audio):
    #    data[i] = audio[int(j)]
    #    i += 1
    #    j += crate
    data = audio[np.arange(0, len(audio), 3)]
    l = len(data)
    if (l & 1 == 1):
        r = data[:-1]
    else:
        r = data
    audio_queue.put(r)
    
@timed
def hasVoice(audio):
    amin = min(audio)
    amax = max(audio)
    diff = abs(amin) + abs(amax)
    return (diff > 0.1)

@timed
def process_audio(audio_queue, model):
    global silenceCnt 
    global audioQueue
    global lastProcessTime
    global lastSilence
    # Block until the next chunk of audio is available on the queue.
    audio = audio_queue.get()

    # check if this audio has data that is not silence
    if hasVoice(audio):
        if silenceCnt >= 1:
            audioQueue = np.concatenate((audioQueue, lastSilence))
        audioQueue = np.concatenate((audioQueue, audio))
        silenceCnt = 0
    else:
        silenceCnt += 1
        lastSilence = audio
    lastProcessTime += 1

    # check if it's been 10s or if I got consecutive 2 sec of silence 
    if lastProcessTime >= 10 or (silenceCnt >= 1 and len(audioQueue) > 0):
        # print(f'Processing at time :{lastProcessTime} silence:{silenceCnt}')
        lastProcessTime = 0
        silenceCnt = 0
        # Transcribe the latest audio chunk.
        transcribe(model=model, audio=audioQueue)
        audioQueue = np.array([],dtype=np.float32)

    # Transcribe the latest audio chunk.
    # transcribe(model=model, audio=audio)


def main(argv):
    # Load the Whisper model into memory, downloading first if necessary.
    logging.info(f'Loading model "{FLAGS.model_name}"...')
    model = whisper.load_model(name=FLAGS.model_name)

    # The first run of the model is slow (buffer init), so run it once empty.
    logging.info('Warming model up...')
    block_size = FLAGS.chunk_seconds * FLAGS.sample_rate
    whisper.transcribe(model=model,
                       audio=torch.tensor(np.zeros(block_size, dtype=np.float32)).to(pdev))

    # Stream audio chunks into a queue and process them from there. The
    # callback is running on a separate thread.
    logging.info('Starting stream...')
    audio_queue = queue.Queue()
    callback = partial(stream_callback, audio_queue=audio_queue)
    with sd.InputStream(samplerate=FLAGS.sample_rate,
                        blocksize=block_size,
                        device=FLAGS.input_device,
                        channels=FLAGS.num_channels,
                        dtype=np.float32,
                        latency=FLAGS.latency,
                        callback=callback):
        while True:
            # Process chunks of audio from the queue.
            process_audio(audio_queue, model)


if __name__ == '__main__':
    app.run(main)
