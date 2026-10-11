# MNIST digit recognition with ExecuTorch under Zephyr

A second Zephyr sample next to `hello-executorch`, aimed at boards that are not Arm
Cortex-M or that have a camera. It embeds a 63 KB MNIST CNN (`TinyDigitNet`, fp32,
portable kernels, selective operator build), proves the export at boot by running an embedded
test image and comparing the logits with PyTorch's, and then, where the board has one,
classifies frames from the camera.

| Board | Core | Camera | Output |
|---|---|---|---|
| Seeed XIAO ESP32-S3 Sense (`xiao_esp32s3/esp32s3/procpu/sense`) | Xtensa LX7 | OV2640 through the Zephyr video API | console |
| Arducam Pico4ML (`rpi_pico` + overlay) | Cortex-M0+, no FPU | HM01B0 through PIO + DMA | console + ST7735 screen |
| any other Zephyr board with ~100 KB RAM | | none | self-test on the console |

The ExecuTorch Zephyr module links the portable kernels on every architecture; nothing here
depends on the Arm backend, CMSIS-NN or an NPU.

## Build and run

Set up the workspace as described in [`zephyr/README.md`](../../README.md), with the HAL for
your board in the project filter (`hal_espressif` and the ESP32-S3 toolchain for the XIAO,
`hal_rpi_pico` for the Pico4ML).

Self-test only, any board:

```
west build -p always -b <board> modules/lib/executorch/zephyr/samples/mnist-camera
```

XIAO ESP32-S3 Sense with the camera:

```
west build -p always -b xiao_esp32s3/esp32s3/procpu/sense modules/lib/executorch/zephyr/samples/mnist-camera
west flash
```

Arducam Pico4ML (camera, screen, USB console):

```
west build -p always -b rpi_pico modules/lib/executorch/zephyr/samples/mnist-camera
# copy build/zephyr/zephyr.uf2 to the RPI-RP2 drive (hold BOOTSEL while plugging in)
```

Once this firmware runs, opening its USB serial port at 1200 baud reboots the board into
BOOTSEL, so later flashes need no button.

## Expected output

```
ExecuTorch on rpi_pico/rp2040 (Zephyr 4.4.0)
model: 62888 bytes in flash, method pool 65536 B, temp pool 8192 B
method 'forward' loaded; input numel 784; pool used 59584 / 65536 bytes

self-test: embedded MNIST image, label 7
  logit[0] =   -3.1244
  ...
  logit[7] =    9.8817  <-- argmax
  ...
  predicted 7, expected 7, max |device - pytorch| = 3.30e-06, 766274 us
SELF-TEST PASS
inference: 766200 us average over 10 runs
camera: HM01B0 ready, 162x122, PIO sm 0, DMA ch 0

camera frame 0:
  ............................
  ...........++####+..........
  ...
  digit 7 (94%, 766012 us)
```

The ESP32-S3 at 240 MHz with its FPU does the same inference in a few milliseconds; the
Cortex-M0+ does fp32 in software, hence the 0.77 s. Every tenth frame the 28x28 crop the model
sees is printed as ASCII, which is the quickest way to check framing and exposure. Write a
digit with a thick pen on white paper and hold it about 10 cm from the lens.

## Regenerating the model

`models/mnist_tiny.pte` and `src/test_input.h` are committed so the sample builds without a
training step. To retrain or re-export:

```
pip install torchvision
python models/train_mnist.py            # ~1 min on CPU, writes both files
```

`models/tiny_digit_net.py` is the PyTorch definition; the firmware applies the same MNIST
normalisation constants.

## Files

| File | Purpose |
|---|---|
| `src/main.cpp` | runtime setup, self-test, camera loop, optional screen |
| `src/camera.h` | the two-call camera interface the board back ends implement |
| `src/camera_video.c` | Zephyr video API capture (RGB565 to 28x28 gray) |
| `src/camera_hm01b0.c` | HM01B0 1-bit serial capture with RP2040 PIO and DMA |
| `src/screen_st7735.c`, `src/font6x8.h` | ST7735 status screen through the display API |
| `boards/*.conf`, `boards/rpi_pico.overlay` | per-board camera, screen and console configuration |
| `scripts/pte_to_header.py` | embeds the `.pte` as a C array at build time |
