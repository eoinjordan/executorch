/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright (c) 2026 Eoin Jordan.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MNIST digit recognition with ExecuTorch under Zephyr on the Arducam
 * Pico4ML (RP2040, Cortex-M0+), and the embedded self-test on any board:
 *
 * 1. Load the .pte embedded in flash (model_pte.h, generated at build time)
 *    and run the MNIST test image baked into test_input.h; compare the
 *    logits with PyTorch's, so a fresh board proves the export -> embed ->
 *    execute path end to end (SELF-TEST PASS).
 * 2. Optionally keep classifying frames from the HM01B0 camera behind
 *    RP2040 PIO + DMA, with an optional ST7735 status screen.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/version.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "model_pte.h" /* generated: model_pte[], model_pte_len */
#include "test_input.h" /* generated: kTestInput[], kTestExpectedLogits[] */

#ifndef CONFIG_APP_CAMERA_NONE
#include "camera.h"
#endif
#ifdef CONFIG_APP_SCREEN_ST7735
#include "screen.h"
#endif
#ifdef CONFIG_APP_USB_BOOTSEL_ON_1200_BAUD
#include <zephyr/drivers/uart.h>

#include <pico/bootrom.h>
#endif

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

/* Same constants as models/tiny_digit_net.py */
static constexpr float kMean = 0.1307f;
static constexpr float kStd = 0.3081f;
static constexpr size_t kNumClasses = 10;
static constexpr int kSide = 28;

#ifndef APP_METHOD_ALLOCATOR_POOL_SIZE
#define APP_METHOD_ALLOCATOR_POOL_SIZE (64 * 1024)
#endif
#ifndef APP_TEMP_ALLOCATOR_POOL_SIZE
#define APP_TEMP_ALLOCATOR_POOL_SIZE (8 * 1024)
#endif

alignas(16) static uint8_t method_pool[APP_METHOD_ALLOCATOR_POOL_SIZE];
alignas(16) static uint8_t temp_pool[APP_TEMP_ALLOCATOR_POOL_SIZE];

namespace {

struct Runner {
  BufferDataLoader loader{model_pte, model_pte_len};
  MemoryAllocator method_allocator{APP_METHOD_ALLOCATOR_POOL_SIZE, method_pool};
  MemoryAllocator temp_allocator{APP_TEMP_ALLOCATOR_POOL_SIZE, temp_pool};
  std::vector<Span<uint8_t>> planned_spans;
  Program* program = nullptr;
  Method* method = nullptr;
  float* input = nullptr;
  size_t input_numel = 0;
  Tensor* input_tensor = nullptr;

  Error init();
  Error run(const float* pixels, float* logits, uint32_t* us);
};

Error Runner::init() {
  Result<Program> program_res = Program::load(&loader);
  if (!program_res.ok()) {
    ET_LOG(Error, "Program::load failed: 0x%x", (unsigned)program_res.error());
    return program_res.error();
  }
  program = new Program(std::move(program_res.get()));

  const char* method_name = program->get_method_name(0).get();
  Result<MethodMeta> meta = program->method_meta(method_name);
  if (!meta.ok()) {
    return meta.error();
  }

  for (size_t id = 0; id < meta->num_memory_planned_buffers(); ++id) {
    size_t size =
        static_cast<size_t>(meta->memory_planned_buffer_size(id).get());
    uint8_t* buf = static_cast<uint8_t*>(method_allocator.allocate(size, 16));
    if (buf == nullptr) {
      ET_LOG(
          Error,
          "planned buffer %zu (%zu B) does not fit the method pool",
          id,
          size);
      return Error::MemoryAllocationFailed;
    }
    planned_spans.push_back({buf, size});
  }

  static HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});
  static MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  Result<Method> method_res =
      program->load_method(method_name, &memory_manager);
  if (!method_res.ok()) {
    ET_LOG(
        Error,
        "load_method(%s) failed: 0x%x",
        method_name,
        (unsigned)method_res.error());
    return method_res.error();
  }
  method = new Method(std::move(method_res.get()));

  Result<TensorInfo> in_meta = meta->input_tensor_meta(0);
  if (!in_meta.ok()) {
    return in_meta.error();
  }
  input_numel = in_meta->nbytes() / sizeof(float);
  input = static_cast<float*>(method_allocator.allocate(in_meta->nbytes(), 16));
  if (input == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  static TensorImpl impl(
      in_meta->scalar_type(),
      in_meta->sizes().size(),
      const_cast<TensorImpl::SizesType*>(in_meta->sizes().data()),
      input,
      const_cast<TensorImpl::DimOrderType*>(in_meta->dim_order().data()));
  static Tensor tensor(&impl);
  input_tensor = &tensor;

  printk(
      "method '%s' loaded; input numel %zu; pool used %zu / %zu bytes\n",
      method_name,
      input_numel,
      method_allocator.used_size(),
      method_allocator.size());
  return Error::Ok;
}

Error Runner::run(const float* pixels, float* logits, uint32_t* us) {
  std::memcpy(input, pixels, input_numel * sizeof(float));
  Error err = method->set_input(EValue(*input_tensor), 0);
  if (err != Error::Ok) {
    ET_LOG(Error, "set_input failed: 0x%x", (unsigned)err);
    return err;
  }
  uint32_t t0 = k_cycle_get_32();
  err = method->execute();
  uint32_t t1 = k_cycle_get_32();
  if (err != Error::Ok) {
    ET_LOG(Error, "execute failed: 0x%x", (unsigned)err);
    return err;
  }
  *us = k_cyc_to_us_floor32(t1 - t0);

  const EValue& out = method->get_output(0);
  if (!out.isTensor()) {
    return Error::InvalidState;
  }
  const float* data = out.toTensor().const_data_ptr<float>();
  for (size_t i = 0; i < kNumClasses; ++i) {
    logits[i] = data[i];
  }
  return Error::Ok;
}

size_t argmax(const float* v, size_t n) {
  size_t best = 0;
  for (size_t i = 1; i < n; ++i) {
    if (v[i] > v[best]) {
      best = i;
    }
  }
  return best;
}

int confidence_pct(const float* logits) {
  size_t best = argmax(logits, kNumClasses);
  float sum = 0.f;
  for (size_t i = 0; i < kNumClasses; ++i) {
    sum += expf(logits[i] - logits[best]);
  }
  return (int)lroundf(100.f / sum);
}

void print_logits(const float* logits) {
  size_t best = argmax(logits, kNumClasses);
  for (size_t i = 0; i < kNumClasses; ++i) {
    printk(
        "  logit[%zu] = %9.4f%s\n",
        i,
        static_cast<double>(logits[i]),
        i == best ? "  <-- argmax" : "");
  }
}

void print_preview(const uint8_t* img) {
  for (int y = 0; y < kSide; ++y) {
    char line[kSide + 1];
    for (int x = 0; x < kSide; ++x) {
      uint8_t v = img[y * kSide + x];
      line[x] = v > 160 ? '#' : (v > 90 ? '+' : '.');
    }
    line[kSide] = '\0';
    printk("  %s\n", line);
  }
}

/* Undo the MNIST normalisation so the embedded test image can be shown. */
void input_to_gray(const float* in, uint8_t* gray) {
  for (int i = 0; i < kSide * kSide; ++i) {
    float v = (in[i] * kStd + kMean) * 255.f;
    gray[i] = (uint8_t)(v < 0.f ? 0.f : v > 255.f ? 255.f : v);
  }
}

#ifdef CONFIG_APP_SCREEN_ST7735
void show(
    const uint8_t* gray,
    const char* status,
    uint16_t status_color,
    int digit,
    int pct) {
  char line[16];
  screen_clear(COLOR_BLACK);
  screen_text(2, 2, "ExecuTorch", COLOR_WHITE, COLOR_BLACK);
  screen_text(2, 12, status, status_color, COLOR_BLACK);
  if (gray != nullptr) {
    screen_gray(12, 24, gray, kSide, kSide, 2);
  }
  if (digit >= 0) {
    snprintf(line, sizeof(line), "digit %d", digit);
    screen_text(2, 86, line, COLOR_YELLOW, COLOR_BLACK);
    snprintf(line, sizeof(line), "%3d%%", pct);
    screen_text(52, 86, line, COLOR_WHITE, COLOR_BLACK);
    screen_bar(2, 97, 76, 5, pct, pct >= 80 ? COLOR_GREEN : COLOR_YELLOW);
  }
  screen_text(2, 150, "zephyr", COLOR_GREY, COLOR_BLACK);
  screen_flush();
}
#else
void show(const uint8_t*, const char*, uint16_t, int, int) {}
#define COLOR_GREEN 0
#define COLOR_RED 0
#define COLOR_YELLOW 0
#define COLOR_WHITE 0
#endif

void maybe_reboot_to_bootsel() {
#ifdef CONFIG_APP_USB_BOOTSEL_ON_1200_BAUD
  /* Same convention as the pico-sdk stdio: opening the USB serial port at
   * 1200 baud reboots into the UF2 bootloader, so scripts can reflash
   * without touching the BOOTSEL button. */
  const struct device* const console = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
  uint32_t baud = 0;
  if (uart_line_ctrl_get(console, UART_LINE_CTRL_BAUD_RATE, &baud) == 0 &&
      baud == 1200) {
    printk("1200 baud seen: rebooting into BOOTSEL\n");
    k_msleep(100);
    reset_usb_boot(0, 0);
  }
#endif
}

} // namespace

int main(void) {
  executorch::runtime::runtime_init();

  /* Boards with a USB console need a moment before the first line. */
  k_msleep(CONFIG_APP_BOOT_DELAY_MS);

  printk(
      "\nExecuTorch on %s (Zephyr %s)\n",
      CONFIG_BOARD_TARGET,
      KERNEL_VERSION_STRING);
  printk(
      "model: %u bytes in flash, method pool %u B, temp pool %u B\n",
      (unsigned)model_pte_len,
      (unsigned)APP_METHOD_ALLOCATOR_POOL_SIZE,
      (unsigned)APP_TEMP_ALLOCATOR_POOL_SIZE);

#ifdef CONFIG_APP_SCREEN_ST7735
  if (screen_init() == 0) {
    show(nullptr, "loading...", COLOR_YELLOW, -1, 0);
  }
#endif

  static Runner runner;
  if (runner.init() != Error::Ok) {
    printk("FAIL: runtime init\n");
    show(nullptr, "RUNTIME FAIL", COLOR_RED, -1, 0);
    return 1;
  }

  /* --- 1. Self-test on the embedded MNIST image ----------------------- */
  float logits[kNumClasses];
  uint32_t us = 0;

  if (runner.run(kTestInput, logits, &us) != Error::Ok) {
    printk("FAIL: inference\n");
    return 1;
  }
  size_t pred = argmax(logits, kNumClasses);
  float max_diff = 0.f;
  for (size_t i = 0; i < kNumClasses; ++i) {
    max_diff = fmaxf(max_diff, fabsf(logits[i] - kTestExpectedLogits[i]));
  }
  printk("\nself-test: embedded MNIST image, label %d\n", TEST_INPUT_LABEL);
  print_logits(logits);
  printk(
      "  predicted %zu, expected %d, max |device - pytorch| = %.2e, %u us\n",
      pred,
      TEST_INPUT_LABEL,
      static_cast<double>(max_diff),
      us);
  bool ok = pred == TEST_INPUT_LABEL && max_diff < 1e-3f;
  printk(ok ? "SELF-TEST PASS\n" : "SELF-TEST FAIL\n");

  static uint8_t gray[kSide * kSide];
  input_to_gray(kTestInput, gray);
  show(
      gray,
      ok ? "SELF-TEST PASS" : "SELF-TEST FAIL",
      ok ? COLOR_GREEN : COLOR_RED,
      (int)pred,
      confidence_pct(logits));

  /* Warm timing: a few more runs, report the average. */
  uint32_t total = 0;
  const int reps = 10;
  for (int i = 0; i < reps; ++i) {
    runner.run(kTestInput, logits, &us);
    total += us;
  }
  printk("inference: %u us average over %d runs\n", total / reps, reps);

  /* --- 2. Optional live camera loop ----------------------------------- */
#ifndef CONFIG_APP_CAMERA_NONE
  if (app_camera_init() != 0) {
    printk("camera init failed; stopping after self-test\n");
  } else {
    static float frame[kSide * kSide];
    static uint8_t preview[kSide * kSide];
    uint32_t seq = 0;
    for (;;) {
      maybe_reboot_to_bootsel();
      if (app_camera_capture(frame, preview, kMean, kStd) == 0 &&
          runner.run(frame, logits, &us) == Error::Ok) {
        int digit = (int)argmax(logits, kNumClasses);
        int pct = confidence_pct(logits);
        if (seq % CONFIG_APP_PREVIEW_EVERY == 0) {
          printk("\ncamera frame %u:\n", seq);
          print_preview(preview);
        }
        printk("  digit %d (%d%%, %u us)\n", digit, pct, us);
        show(preview, "camera", COLOR_WHITE, digit, pct);
        seq++;
      }
      k_msleep(CONFIG_APP_CAMERA_INTERVAL_MS);
    }
  }
#endif
  for (;;) {
    maybe_reboot_to_bootsel();
    k_msleep(200);
  }
  return 0;
}
