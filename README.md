# Whisper Edge

Porting [OpenAI Whisper](https://github.com/openai/whisper) speech recognition to edge devices with hardware ML accelerators, enabling always-on live voice transcription. Current work includes [Jetson Nano](#jetson-nano) and [Coral Edge TPU](#coral-edge-tpu).

## Jetson Nano

### Shopping cart

| Part | Price (2023) |
| :- | -: |
| [NVIDIA Jetson Nano Developer Kit (4G)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) | [$149.00](https://www.amazon.com/NVIDIA-Jetson-Nano-Developer-945-13450-0000-100/dp/B084DSDDLT/) |
| [ChanGeek CGS-M1 USB Microphone](https://www.amazon.com/gp/product/B08M37224H/ref=ppx_yo_dt_b_asin_title_o03_s00) | [$16.99](https://www.amazon.com/gp/product/B08M37224H/ref=ppx_yo_dt_b_asin_title_o03_s00) |
| [Noctua NF-A4x10 FLX Fan](https://noctua.at/en/nf-a4x10-flx) (or similar, recommended) | [$13.95](https://www.amazon.com/gp/product/B009NQLT0M/ref=ppx_yo_dt_b_asin_title_o00_s00) |
| [D-Link DWA-181](https://www.dlink.com/en/products/dwa-181-ac1300-mu-mimo-wi-fi-nano-usb-adapter) (or similar, optional) | [$21.94](https://www.amazon.com/D-Link-Wireless-Internet-Supported-DWA-181-US/dp/B07YYL3RYJ/) |

### Model

The [`base.en` version](https://github.com/openai/whisper#available-models-and-languages) of Whisper seems to work best for the Jetson Nano:
 - `base` is the largest model size that fits into the 4GB of memory without modification.
 - Inference performance with `base` is ~10x real-time in isolation and ~1x real-time while recording concurrently.
 - Using the english-only `.en` version further improves WER ([<5% on LibriSpeech test-clean](https://cdn.openai.com/papers/whisper.pdf)).

### Hack

Dilemma:
 - Whisper and some of its dependencies require Python 3.8.
 - The latest supported version of [JetPack](https://developer.nvidia.com/embedded/jetpack) for Jetson Nano is [4.6.3](https://developer.nvidia.com/jetpack-sdk-463), which is on Python 3.6.
 - [No easy way](https://github.com/maxbbraun/whisper-edge/issues/2) to update Python to 3.8 without losing CUDA support for PyTorch.

Workaround:
 - Fork [whisper](https://github.com/maxbbraun/whisper) and [tiktoken](https://github.com/maxbbraun/tiktoken), downgrading them to Python 3.6.

### Setup

First, follow the [Jetson Nano Developer Kit setup instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit). We will use [NVIDIA Docker containers](https://hub.docker.com/r/dustynv/jetson-inference/tags) to run inference.

```bash
ssh user@jetson-nano.local
git clone --recursive https://github.com/maxbbraun/jetson-inference
exit
```

```bash
git clone https://github.com/maxbbraun/whisper-edge.git
cd whisper-edge
scp Dockerfile.jetson-nano user@jetson-nano.local:~/jetson-inference/Dockerfile
scp stream.py user@jetson-nano.local:~/jetson-inference/
```

```bash
ssh user@jetson-nano.local
cd jetson-inference
DOCKER_IMAGE=dustynv/jetson-inference:r32.7.1  # JetPack 4.6.1
docker/build.sh $DOCKER_IMAGE
exit
```

### Run

```bash
ssh user@jetson-nano.local
cd jetson-inference
docker/run.sh --run "python stream.py"
```

```bash
I0317 00:42:23.979984 547488051216 stream.py:75] Loading model "base.en"...
100%|#######################################| 139M/139M [00:30<00:00, 4.71MiB/s]
I0317 00:43:14.232425 547488051216 stream.py:79] Warming model up...
I0317 00:43:55.164070 547488051216 stream.py:86] Starting stream...
I0317 00:44:19.775566 547488051216 stream.py:51]
I0317 00:44:22.046195 547488051216 stream.py:51] Open AI's mission is to ensure that artificial general intelligence
I0317 00:44:31.353919 547488051216 stream.py:51] benefits all of humanity.
I0317 00:44:49.219501 547488051216 stream.py:51]
```

## Coral Edge TPU

See the corresponding [issue](https://github.com/maxbbraun/whisper-edge/issues/1) about what supporting the [Google Coral Edge TPU](https://coral.ai/products/) may look like.
