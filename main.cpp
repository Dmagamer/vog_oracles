#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

namespace {

constexpr bool ATHEON = false;

// 0 off, 1 prints all sounds above decibel cutoff, 2 prints those making it through filters
constexpr int DEBUG = 0;

// Ignore notes below this volume.
constexpr double DECIBEL_CUTOFF = 20.0;
// Must be this close to the note.
constexpr double DIFF_CUTOFF = 0.04;
// B flat special condition.
constexpr double B_FLAT_MIN = 59.05;
constexpr double B_FLAT_MAX = 59.17;

constexpr std::size_t QUEUE_LENGTH = 8;
constexpr std::size_t NOTES_IN_QUEUE_REQUIRED = 5;

// Constants for note calculations.
constexpr int NOTE_MIN = 60;
constexpr int NOTE_MAX = 69;
constexpr double FSAMP = 22050.0;
constexpr std::size_t FRAME_SIZE = 2096;
constexpr std::size_t FRAMES_PER_FFT = 16;
constexpr std::size_t SAMPLES_PER_FFT = FRAME_SIZE * FRAMES_PER_FFT;
constexpr double FREQ_STEP = FSAMP / static_cast<double>(SAMPLES_PER_FFT);
constexpr double PI = 3.14159265358979323846;

// Clockwise note order with C as 1. Edit this vector to tweak mapping.
const std::vector<std::string> NOTE_CLOCKWISE_ORDER = {"C", "D", "F#", "A", "Bb", "G", "E"};

std::unordered_map<std::string, int> build_note_to_number() {
    std::unordered_map<std::string, int> mapping;
    for (std::size_t i = 0; i < NOTE_CLOCKWISE_ORDER.size(); ++i) {
        mapping[NOTE_CLOCKWISE_ORDER[i]] = static_cast<int>(i + 1);
    }
    return mapping;
}

const std::unordered_map<std::string, int> NOTE_TO_NUMBER = build_note_to_number();

inline double freq_to_number(double f) {
    return 69.0 + 12.0 * std::log2(f / 440.0);
}

inline double number_to_freq(double n) {
    return 440.0 * std::pow(2.0, ((n - 69.0) / 12.0));
}

inline const std::vector<std::string>& note_names() {
    static const std::vector<std::string> atheon = {"C", "N", "D", "N", "E", "N", "F#", "G", "N", "A", "N", "N"};
    static const std::vector<std::string> templar = {"C", "N", "D", "N", "E", "N", "F#", "G", "N", "A", "Bb", "N"};
    return ATHEON ? atheon : templar;
}

inline std::string note_name(int n) {
    const auto& names = note_names();
    int index = n % 12;
    if (index < 0) {
        index += 12;
    }
    return names[static_cast<std::size_t>(index)];
}

inline double note_to_fftbin(double n) {
    return number_to_freq(n) / FREQ_STEP;
}

struct DeviceChoice {
    ma_device_id id;
    std::string name;
};

DeviceChoice choose_input_device() {
    ma_context context;
    if (ma_context_init(nullptr, 0, nullptr, &context) != MA_SUCCESS) {
        throw std::runtime_error("Failed to initialize audio context");
    }

    ma_device_info* pPlaybackInfos;
    ma_uint32 playbackCount;
    ma_device_info* pCaptureInfos;
    ma_uint32 captureCount;

    if (ma_context_get_devices(&context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) != MA_SUCCESS) {
        ma_context_uninit(&context);
        throw std::runtime_error("Failed to list audio devices");
    }

    if (captureCount == 0) {
        ma_context_uninit(&context);
        throw std::runtime_error("No input devices available");
    }

    for (ma_uint32 i = 0; i < captureCount; ++i) {
        std::cout << "Input Device id " << i << " - " << pCaptureInfos[i].name << '\n';
    }

    std::cout << "Please choose a device ID: ";
    int selected = -1;
    std::cin >> selected;

    if (selected < 0 || static_cast<ma_uint32>(selected) >= captureCount) {
        ma_context_uninit(&context);
        throw std::runtime_error("Invalid device ID selected");
    }

    DeviceChoice choice{pCaptureInfos[selected].id, pCaptureInfos[selected].name};
    ma_context_uninit(&context);
    return choice;
}

// Ring buffer shared between the capture callback and the main thread.
struct CaptureContext {
    ma_pcm_rb rb{};
};

static void capture_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    (void)pOutput;
    auto* ctx = static_cast<CaptureContext*>(pDevice->pUserData);
    if (pInput == nullptr) {
        return;
    }

    const auto* src = static_cast<const int16_t*>(pInput);
    ma_uint32 remaining = frameCount;
    while (remaining > 0) {
        ma_uint32 frames = remaining;
        void* pBuffer = nullptr;
        if (ma_pcm_rb_acquire_write(&ctx->rb, &frames, &pBuffer) != MA_SUCCESS || frames == 0) {
            break;
        }
        std::memcpy(pBuffer, src, frames * sizeof(int16_t));
        ma_pcm_rb_commit_write(&ctx->rb, frames);
        src += frames;
        remaining -= frames;
    }
}

double goertzel_magnitude(const std::vector<double>& samples, std::size_t k) {
    const double omega = 2.0 * PI * static_cast<double>(k) / static_cast<double>(samples.size());
    const double coeff = 2.0 * std::cos(omega);

    double s_prev = 0.0;
    double s_prev2 = 0.0;
    for (double sample : samples) {
        const double s = sample + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }

    const double power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2;
    return std::sqrt(std::max(power, 0.0));
}

std::string note_to_output(const std::string& note) {
    const auto it = NOTE_TO_NUMBER.find(note);
    if (it == NOTE_TO_NUMBER.end()) {
        return "?";
    }
    return std::to_string(it->second);
}

}  // namespace

int main() {
    try {
        const DeviceChoice device = choose_input_device();

        CaptureContext capCtx;
        // Ring buffer: 16× frame size gives comfortable headroom.
        constexpr ma_uint32 RB_FRAMES = static_cast<ma_uint32>(FRAME_SIZE * 16);
        if (ma_pcm_rb_init(ma_format_s16, 1, RB_FRAMES, nullptr, nullptr, &capCtx.rb) != MA_SUCCESS) {
            throw std::runtime_error("Failed to initialize ring buffer");
        }

        ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
        cfg.capture.pDeviceID  = const_cast<ma_device_id*>(&device.id);
        cfg.capture.format     = ma_format_s16;
        cfg.capture.channels   = 1;
        cfg.sampleRate         = static_cast<ma_uint32>(FSAMP);
        cfg.dataCallback       = capture_callback;
        cfg.pUserData          = &capCtx;

        ma_device maDevice;
        if (ma_device_init(nullptr, &cfg, &maDevice) != MA_SUCCESS) {
            ma_pcm_rb_uninit(&capCtx.rb);
            throw std::runtime_error("Failed to initialize audio device");
        }

        if (ma_device_start(&maDevice) != MA_SUCCESS) {
            ma_device_uninit(&maDevice);
            ma_pcm_rb_uninit(&capCtx.rb);
            throw std::runtime_error("Failed to start audio device");
        }

        const std::size_t imin = std::max<std::size_t>(0, static_cast<std::size_t>(std::floor(note_to_fftbin(NOTE_MIN - 1))));
        const std::size_t imax = std::min<std::size_t>(SAMPLES_PER_FFT, static_cast<std::size_t>(std::ceil(note_to_fftbin(NOTE_MAX + 1))));

        std::vector<float> buf(SAMPLES_PER_FFT, 0.0f);
        std::vector<int16_t> frame(FRAME_SIZE, 0);
        std::vector<double> window(SAMPLES_PER_FFT, 0.0);
        std::vector<double> windowed(SAMPLES_PER_FFT, 0.0);

        for (std::size_t i = 0; i < SAMPLES_PER_FFT; ++i) {
            window[i] = 0.5 * (1.0 - std::cos((2.0 * PI * static_cast<double>(i)) / static_cast<double>(SAMPLES_PER_FFT)));
        }

        const double window_sum = std::accumulate(window.begin(), window.end(), 0.0);

        std::string last_printed_note;
        int samples_without_char = 0;
        std::deque<std::string> sequence(QUEUE_LENGTH, "");
        std::size_t num_frames = 0;

        while (true) {
            ++samples_without_char;
            if (samples_without_char == 35 || samples_without_char == 100) {
                last_printed_note.clear();
                std::cout << '\n';
            }

            // Block until FRAME_SIZE int16 frames have been read from the ring buffer.
            ma_uint32 framesRead = 0;
            while (framesRead < static_cast<ma_uint32>(FRAME_SIZE)) {
                ma_uint32 frames = static_cast<ma_uint32>(FRAME_SIZE) - framesRead;
                void* pBuffer = nullptr;
                if (ma_pcm_rb_acquire_read(&capCtx.rb, &frames, &pBuffer) == MA_SUCCESS && frames > 0) {
                    std::memcpy(frame.data() + framesRead, pBuffer, frames * sizeof(int16_t));
                    ma_pcm_rb_commit_read(&capCtx.rb, frames);
                    framesRead += frames;
                } else {
                    ma_sleep(1);
                }
            }

            std::move(buf.begin() + FRAME_SIZE, buf.end(), buf.begin());
            for (std::size_t i = 0; i < FRAME_SIZE; ++i) {
                buf[SAMPLES_PER_FFT - FRAME_SIZE + i] = static_cast<float>(frame[i]);
            }

            for (std::size_t i = 0; i < SAMPLES_PER_FFT; ++i) {
                windowed[i] = static_cast<double>(buf[i]) * window[i];
            }

            double decibel_sum = 0.0;
            double max_mag = -std::numeric_limits<double>::infinity();
            std::size_t max_bin = imin;

            for (std::size_t bin = imin; bin < imax; ++bin) {
                const double magnitude = (goertzel_magnitude(windowed, bin) * 2.0) / window_sum;
                const double normalized = std::max(magnitude / 32768.0, 1e-12);
                const double db = 20.0 * std::log10(normalized) + 120.0;
                decibel_sum += db;
                if (magnitude > max_mag) {
                    max_mag = magnitude;
                    max_bin = bin;
                }
            }

            const double decibel = decibel_sum / static_cast<double>(imax - imin);
            if (decibel < DECIBEL_CUTOFF) {
                continue;
            }

            const double freq = static_cast<double>(max_bin) * FREQ_STEP;
            const double n = freq_to_number(freq);
            const int n0 = static_cast<int>(std::round(n));
            std::string note = note_name(n0);
            const double diff = n - static_cast<double>(n0);
            ++num_frames;

            if (num_frames >= FRAMES_PER_FFT) {
                if (DEBUG == 1) {
                    std::cout << "note: " << n << " " << note << " " << diff << " " << decibel << '\n';
                }

                if ((std::abs(diff) < DIFF_CUTOFF && note != "N") || (n > B_FLAT_MIN && n < B_FLAT_MAX)) {
                    if (note == "N") {
                        if (ATHEON) {
                            continue;
                        }
                        note = "Bb";
                    }

                    if (DEBUG == 2) {
                        std::cout << "note: " << note << " " << diff << " " << decibel << '\n';
                    }

                    sequence.push_back(note);
                    if (sequence.size() > QUEUE_LENGTH) {
                        sequence.pop_front();
                    }

                    const std::size_t same_count = static_cast<std::size_t>(std::count(sequence.begin(), sequence.end(), note));
                    if (same_count == NOTES_IN_QUEUE_REQUIRED && last_printed_note != note) {
                        const std::string out = note_to_output(note);
                        if (DEBUG != 0) {
                            std::cout << "IF NOT DEBUG " << out << '\n';
                        } else {
                            std::cout << out;
                            std::cout.flush();
                        }
                        last_printed_note = note;
                        samples_without_char = 0;
                    }
                }
            }
        }

        ma_device_stop(&maDevice);
        ma_device_uninit(&maDevice);
        ma_pcm_rb_uninit(&capCtx.rb);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
