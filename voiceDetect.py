
import webrtcvad
import queue
import time
import torch
from collections import deque
import numpy as np
import whisper

def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return  float2pcm(sig, dtype='int16').tobytes()

def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte,dtype=np.int16), dtype='float32')

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


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

# if not in recording 
#    start to record when voice is detected.
#    buffer up to 1 sec leading to the voice. 
# If Recording 
#    stop recording if no voice is detected for a second
#    also break the sentence if time is > 5 sec on a pause of 100ms 
#    if time is > 10 Sec break anyways. 

class FindVoiceSegments(object):
    def __init__(self, sample_rate, debug=False):
        self.debug = debug
        self.sampleDuration = 30 # ms
        self.last1sec = deque(maxlen=int(1000/self.sampleDuration))
        self.vad = webrtcvad.Vad(0)
        self.audio_queue = queue.Queue()
        self.sample_rate = sample_rate
        self.recording = False
        self.recordingStartTime = int(time.time())
        self.noVoiceCnt = 0
        self.hasVoiceCnt = 0
        self.transcriptionQueue = queue.Queue()
        #self.pdev = torch.device("cuda:0")
        self.pdev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def hasVoice(self, data):
        d = float_to_byte(data)
        return self.vad.is_speech(d, self.sample_rate)

    def startRecording(self, nowMinus1Sec):
        self.recordingStartTime = int(time.time())
        self.recording = True
        #self.audio_queue.clear()
        if nowMinus1Sec:
            last1secSamples = int(1000/self.sampleDuration)
            dt = np.array([],dtype=np.float32)
            if (len(self.last1sec) > last1secSamples):
                for i in range(len(self.last1sec) - last1secSamples, len(self.last1sec)):
                    dt += self.last1sec[i]
            self.audio_queue.put(dt) # start with 

    def pauseDetected(self, duration):
        sampleCnt = int(duration/self.sampleDuration)
        return self.noVoiceCnt > sampleCnt

    def flushQueue(self):
        dt = np.array([],dtype=np.float32)
        while not self.audio_queue.empty():
            dt = np.concatenate((dt, self.audio_queue.get()))
        if self.debug:
            print(".")
#            print("Flushing data size:", len(dt))
        self.transcriptionQueue.put(dt)

    def transcribe(self, model):
        if self.transcriptionQueue.qsize() == 0:
            return None
        dt_f = self.transcriptionQueue.get()
        audio = torch.from_numpy(dt_f).to(self.pdev)
        result = whisper.transcribe(model=model, audio=audio)
        text = result['text'].strip()
        return text

    def pendingQSize(self):
        return self.transcriptionQueue.qsize()

    def processBytesData(self, data): 
        return self.processBytesData(byte_to_float(data))

    def processFloatData(self, dataIn, debug=False): 
        # data should be 30 ms interval data with 
        # data is in array of floating point np.float32
        data = dataIn
        if self.sample_rate != 16000 and self.sample_rate != 48000:
            raise Exception("Only sample rate of 16k and 48k supported")
        if self.sample_rate == 48000:
            data = dataIn[::3]
        self.last1sec.append(data)
        hasVoice = self.hasVoice(data)
        if(self.debug and hasVoice and False):
            print("Got voice data:", self.transcriptionQueue.qsize(), self.noVoiceCnt, self.hasVoiceCnt)
        if hasVoice:
            self.noVoiceCnt = 0
            self.hasVoiceCnt += 1
        else:
            self.noVoiceCnt += 1
            self.hasVoiceCnt = 0
        if self.recording:
            self.audio_queue.put(data)
            timeElapsed = int(time.time() - self.recordingStartTime)
            if timeElapsed > 10 or (timeElapsed > 2 and self.pauseDetected(100)):
                self.flushQueue()
                self.recording = False
                if hasVoice:
                    self.startRecording(False)
        elif self.hasVoiceCnt > 2:
            self.startRecording(True)
