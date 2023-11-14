# Run the VAD on 10 ms of silence. The result should be False.
import webrtcvad
import sounddevice as sd
import queue
from functools import partial
from absl import flags, app
import numpy as np
import time
from time import sleep
from collections import deque
from voiceDetect import FindVoiceSegments

vad = webrtcvad.Vad(1)

FLAGS = flags.FLAGS
flags.DEFINE_string('input_device', 'hw:0,0',
                    'The input device used to record audio.')
flags.DEFINE_integer('sample_rate', 48000,
                     'The sample rate of the recorded audio.')
gc = 0



def stream_callback(indata, frames, time, status, audio_queue, fvs):
    global gc
    audio = indata[:, 0].copy()
    #baudio = float_to_byte(audio)
    fvs.processFloatData(audio)
#    audio_queue.put(baudio)
#    if(vad.is_speech(baudio, FLAGS.sample_rate)):
#        print(gc, "Got voice data:", len(audio))
#        gc += 1

        

def main(argv):
    mic_name = 'hw:0,0'
    audio_queue = queue.Queue()
    record_timeout=1

    fvs = FindVoiceSegments(FLAGS.sample_rate, True)

    callback = partial(stream_callback, audio_queue=audio_queue, fvs=fvs)

    block_size = int(30 * FLAGS.sample_rate / 1000)
    with sd.InputStream(samplerate=FLAGS.sample_rate,
                            blocksize=block_size,
                            device=FLAGS.input_device,
                            channels=1, 
                            dtype=np.float32,
                            latency='low',
                            callback=callback):
        cnt = 0
        while cnt < 100:
            sleep(0.25)
            cnt+=1

#            print("Audio Queue Size", audio_queue.qsize())



if __name__ == '__main__':
    app.run(main)
