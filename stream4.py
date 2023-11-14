from absl import app
from absl import flags
from absl import logging
from functools import partial
from functools import wraps
import numpy as np
import queue
import sounddevice as sd
from time import time as now
import time
import whisper
import torch
import audioop
import sys
from voiceDetect import FindVoiceSegments
from datetime import datetime, timedelta

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
flags.DEFINE_string('input_device', 'hw:0,0',
                    'The input device used to record audio.')
flags.DEFINE_integer('sample_rate', 48000,
                     'The sample rate of the recorded audio.')
flags.DEFINE_integer('channel_index', 0,
                     'The index of the channel to use for transcription.')
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
def stream_callback(indata, frames, time, status, fvs):
    #if status:
    #    logging.warning(f'Stream callback status: {status}')
    #    print("##", status)

    # Add this chunk of audio to the queue.
    audio = indata[:, FLAGS.channel_index].copy()
    fvs.processFloatData(audio)    



def main(argv):
    # Load the Whisper model into memory, downloading first if necessary.
    logging.info(f'Loading model "{FLAGS.model_name}"...')
    model = whisper.load_model(name=FLAGS.model_name)

    fvs = FindVoiceSegments(FLAGS.sample_rate, False)

    # The first run of the model is slow (buffer init), so run it once empty.
    logging.info('Warming model up...')
    block_size = int(30 * FLAGS.sample_rate / 1000)
    whisper.transcribe(model=model,
                       audio=torch.tensor(np.zeros(block_size, dtype=np.float32)).to(pdev))

    # Stream audio chunks into a queue and process them from there. The
    # callback is running on a separate thread.
    logging.info('Starting stream...')
    callback = partial(stream_callback, fvs=fvs)
    with sd.InputStream(samplerate=FLAGS.sample_rate,
                        blocksize=block_size,
                        device=FLAGS.input_device,
                        channels=1,
                        dtype=np.float32,
                        latency=FLAGS.latency,
                        callback=callback):
        while True:
            # Process chunks of audio from the queue.
            if fvs.pendingQSize() == 0:
                time.sleep(0.25)
            else:
                while(fvs.pendingQSize() != 0):
                    TM="[" + datetime.now().strftime('%H:%M:%S>') + "][" + str(fvs.pendingQSize()) +"]"
                    text=fvs.transcribe(model)
                    print(TM + text)
                    sys.stdout.flush()

if __name__ == '__main__':
    app.run(main)
