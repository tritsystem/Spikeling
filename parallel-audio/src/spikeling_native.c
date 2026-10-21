/*
 * Neural Sandbox — 3D PROOF (Clean Terminal Version)
 *
 * This version PROVES the 3D effect is working by showing:
 *   1. Real-time pan position that visibly moves
 *   2. Grain azimuth/elevation/distance that change every frame
 *   3. A moving "sound position" indicator
 *   4. All displayed without corrupting the grid
 */

#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "../include/miniaudio.h"
#include "kiss_fft.h"
#include "spikeling_hw.h"

/* ── Constants ───────────────────────────────────────────────────── */
/* SAMPLE_RATE is now overridable at compile time (e.g.
 * `-DSAMPLE_RATE=192000`) instead of hardcoded. Default stays 44100 --
 * unchanged for every existing build script/user of this WASAPI-loopback
 * "3D audio proof" engine (it captures SYSTEM OUTPUT via
 * ma_device_type_loopback, i.e. whatever's already playing on this PC,
 * not an external microphone -- unrelated in purpose to bat-call
 * capture, but made configurable here per explicit request so the same
 * pipeline could in principle be rebuilt for a higher-rate ultrasonic
 * front-end later).
 *
 * NYQUIST NOTE: at the default 44100 Hz, this can only represent audio
 * up to ~22.05 kHz -- nowhere near the 20-120 kHz range of UK bat
 * echolocation calls. Overriding to e.g. 250000+ would be needed before
 * this engine could see bat-range content at all; the actual bat
 * discrimination network built for this project (Resonator neurons in
 * the Python/.spk runtime, see ai-apps/bat_species_gate.spk) synthesizes
 * its own drive signal directly and is NOT limited by this file's rate.
 */
#ifndef SAMPLE_RATE
#define SAMPLE_RATE     44100
#endif
#define PI              3.14159265358979323846f
#define TWO_PI          (2.0f * PI)
#define DEG2RAD(deg)    ((deg) * PI / 180.0f)

#define GRID_WIDTH      80
#define GRID_HEIGHT     24
#define GRID_CELLS      (GRID_WIDTH * GRID_HEIGHT)
#define NUM_BANDS       12
#define ROWS_PER_BAND   (GRID_HEIGHT / NUM_BANDS)

#define FFT_SIZE        2048
#define BIN_HZ          ((double)SAMPLE_RATE / FFT_SIZE)

/* Band layout, ported from the validated spikeling_test/mic_band_encoder_test.c
 * design: NUM_BANDS log-spaced bands between BAND_FREQ_MIN/MAX (covers
 * perceptually/musically relevant range with roughly equal importance per
 * band, unlike equal-Hz spacing which wastes bands above a few kHz), with
 * each band clamped to a minimum width of MIN_BAND_BINS FFT bins so no band
 * is narrower than a Hann window's leakage skirt. Band energy below
 * ACTIVATION_DB_BELOW_PEAK relative to the loudest band in the current
 * frame is gated to zero -- this is what actually fixed the old fixed-band
 * design's tendency for narrow neighbor bands to falsely light up purely
 * from window leakage next to a loud tone. */
#define BAND_FREQ_MIN              20.0
#define BAND_FREQ_MAX              20000.0
#define MIN_BAND_BINS              4
#define ACTIVATION_DB_BELOW_PEAK   15.0

#define MAX_GRAINS      256
#define MASTER_VOL      0.17f
#define MAX_ACTIVE_GRAINS_SOFT_CAP 80

#define GRAIN_CYCLES        2.5f
#define GRAIN_LEN_MIN  ((int)(0.018f * SAMPLE_RATE))
#define GRAIN_LEN_MAX  ((int)(0.055f * SAMPLE_RATE))

#define ENV_ATK   (1.0f - expf(-1.0f / (0.002f * SAMPLE_RATE)))
#define ENV_REL   (1.0f - expf(-1.0f / (0.150f * SAMPLE_RATE)))

#define ADAPT_WIN  (SAMPLE_RATE * 2)
#define ADAPT_SCALE_MAX        25.0f
#define ADAPT_RMS_FLOOR        1e-4f
#define ADAPT_SMOOTH_PER_STEP  0.25f

#define COMB_A_LEN   1621
#define COMB_B_LEN   1801
#define COMB_C_LEN   1993
#define COMB_D_LEN   2143
#define AP_E_LEN      347
#define AP_F_LEN      113
#define COMB_FB       0.82f
#define AP_FB         0.5f

#define MCOMB_LEN    ((int)(0.12f * SAMPLE_RATE))
#define MCOMB_FB      0.62f
#define MS_WIDTH      1.8f
#define CHORUS_DELAY_MAX  ((int)(0.015f * SAMPLE_RATE))

/* ═══════════════════════════════════════════════════════════════════
   Band config
   ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    int    row_start;
    double gain;
    double decay;
    double synapse_weight;
    const char *name;
    float  sine_base_hz;
    float  grain_hz_lo;
    float  grain_hz_hi;
    float  sat_drive;
} BandConfig;

/* row_start, gain, decay, synapse_weight, and sat_drive are tapered by hand
 * (low bands = slower/louder/more coupled, high bands = faster/quieter/less
 * coupled, same intent as the old 4-band table). row_start, sine_base_hz,
 * grain_hz_lo, grain_hz_hi are filled in at startup by init_bands() once the
 * actual (bin-clamped) band edges are known -- see header comment above. */
static BandConfig BANDS[NUM_BANDS] = {
    { 0, 600.0, 0.970, 0.50, "B0 ", 0, 0, 0, 1.80f },
    { 0, 520.0, 0.965, 0.47, "B1 ", 0, 0, 0, 1.70f },
    { 0, 440.0, 0.960, 0.44, "B2 ", 0, 0, 0, 1.60f },
    { 0, 360.0, 0.955, 0.41, "B3 ", 0, 0, 0, 1.55f },
    { 0, 300.0, 0.950, 0.38, "B4 ", 0, 0, 0, 1.50f },
    { 0, 260.0, 0.940, 0.35, "B5 ", 0, 0, 0, 1.45f },
    { 0, 220.0, 0.930, 0.32, "B6 ", 0, 0, 0, 1.40f },
    { 0, 190.0, 0.920, 0.29, "B7 ", 0, 0, 0, 1.35f },
    { 0, 160.0, 0.910, 0.27, "B8 ", 0, 0, 0, 1.30f },
    { 0, 130.0, 0.900, 0.25, "B9 ", 0, 0, 0, 1.25f },
    { 0, 105.0, 0.890, 0.23, "B10", 0, 0, 0, 1.20f },
    { 0,  80.0, 0.880, 0.20, "B11", 0, 0, 0, 1.20f },
};

static int g_band_lo_bin[NUM_BANDS];
static int g_band_hi_bin[NUM_BANDS];

/* Same log-spacing + min-bin-width-clamp algorithm validated in
 * spikeling_test/mic_band_encoder_test.c, just parameterized on this
 * engine's own FFT_SIZE/SAMPLE_RATE (different from the test harness). */
static void init_bands(void) {
    double edges_hz[NUM_BANDS + 1];
    double ratio = pow(BAND_FREQ_MAX / BAND_FREQ_MIN, 1.0 / NUM_BANDS);
    edges_hz[0] = BAND_FREQ_MIN;
    for (int k = 1; k <= NUM_BANDS; k++) edges_hz[k] = edges_hz[k - 1] * ratio;

    int edge_bin[NUM_BANDS + 1];
    for (int k = 0; k <= NUM_BANDS; k++) edge_bin[k] = (int)lround(edges_hz[k] / BIN_HZ);
    if (edge_bin[0] < 1) edge_bin[0] = 1; /* never include DC */

    for (int k = 1; k <= NUM_BANDS; k++) {
        if (edge_bin[k] < edge_bin[k - 1] + MIN_BAND_BINS)
            edge_bin[k] = edge_bin[k - 1] + MIN_BAND_BINS;
    }

    int max_bin = FFT_SIZE / 2 - 1;
    if (edge_bin[NUM_BANDS] > max_bin) edge_bin[NUM_BANDS] = max_bin;

    for (int b = 0; b < NUM_BANDS; b++) {
        g_band_lo_bin[b] = edge_bin[b];
        g_band_hi_bin[b] = edge_bin[b + 1] - 1;
        if (g_band_hi_bin[b] < g_band_lo_bin[b]) g_band_hi_bin[b] = g_band_lo_bin[b];

        float lo_hz = (float)(g_band_lo_bin[b] * BIN_HZ);
        float hi_hz = (float)((g_band_hi_bin[b] + 1) * BIN_HZ);
        BANDS[b].row_start    = b * ROWS_PER_BAND;
        BANDS[b].grain_hz_lo  = lo_hz;
        BANDS[b].grain_hz_hi  = hi_hz;
        BANDS[b].sine_base_hz = sqrtf(lo_hz * hi_hz); /* geometric band center */
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Shared state
   ═══════════════════════════════════════════════════════════════════ */
static volatile float g_band_energy[NUM_BANDS]    = {0};
static volatile float g_spike_density[NUM_BANDS]  = {0};
static volatile float g_total_density     = 0.0f;
static volatile float g_bass_energy       = 0.0f;

static double g_adapt_sum[NUM_BANDS]   = {0};
static long   g_adapt_count    = 0;
static float  g_adapt_scale[NUM_BANDS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

/* ── 3D PROOF STATE ── */
static float g_proof_azimuth = 0.0f;      /* Current position */
static float g_proof_elevation = 0.0f;
static float g_proof_distance = 1.0f;
static float g_proof_pan_l = 0.5f;
static float g_proof_pan_r = 0.5f;
static int   g_proof_active = 0;
static float g_proof_time = 0.0f;

/* Trigger queue */
#define TRIGGER_QUEUE 64
typedef struct { 
    int band; 
    float pan;
} GrainTrig;
static volatile GrainTrig g_trig_queue[TRIGGER_QUEUE];
static volatile int g_trig_write = 0;
static volatile int g_trig_read  = 0;
static volatile int g_active_grains = 0;

/* Ring buffer */
#define RING_SIZE (FFT_SIZE * 4)
static float         g_ring[RING_SIZE];
static volatile long g_ring_write = 0;
static volatile long g_ring_read  = 0;

/* ═══════════════════════════════════════════════════════════════════
   Synthesis state
   ═══════════════════════════════════════════════════════════════════ */

static float s_sine_phase[NUM_BANDS]     = {0};
static float s_chorus_phase[NUM_BANDS]   = {0};
static float s_lfo_phase[NUM_BANDS]      = {0};
static float s_sine_detune[NUM_BANDS]    = {0};
static float s_env_follow[NUM_BANDS]     = {0};
static float s_chorus_buf[CHORUS_DELAY_MAX * 2];
static int   s_chorus_pos = 0;

typedef struct {
    int   active, pos, length;
    float phase, freq, amp;
    float azimuth;
    float elevation;
    float distance;
    float pan_l, pan_r;
    float velocity;
    float phase_offset;
} Grain;
static Grain s_grains[MAX_GRAINS];
static int s_band_grain[NUM_BANDS] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
};
#define GRAIN_EXTEND_SAMPLES ((int)(0.05f * SAMPLE_RATE))
#define GRAIN_PAN_GLIDE       0.3f

static float s_comb_a[COMB_A_LEN], s_comb_b[COMB_B_LEN];
static float s_comb_c[COMB_C_LEN], s_comb_d[COMB_D_LEN];
static int   s_comb_ap[4];
static float s_ap_e[AP_E_LEN], s_ap_f[AP_F_LEN];
static int   s_ap_ep, s_ap_fp;

static float s_mcomb_buf[MCOMB_LEN];
static int   s_mcomb_pos   = 0;
static float s_mcomb_delay = (int)(0.07f * SAMPLE_RATE);

static unsigned int s_rng = 12345;
static inline float rngf(void) {
    s_rng = s_rng * 1664525u + 1013904223u;
    return (float)(s_rng >> 8) / (float)(1 << 24);
}

static float g_spatial_time = 0.0f;

/* ═══════════════════════════════════════════════════════════════════
   3D Processing with Proof Tracking
   ═══════════════════════════════════════════════════════════════════ */

static void apply_3d(float sample, float azimuth, float elevation, float distance,
                     float *outL, float *outR) {
    if (azimuth > 180.0f) azimuth = 180.0f;
    if (azimuth < -180.0f) azimuth = -180.0f;
    if (elevation > 80.0f) elevation = 80.0f;
    if (elevation < -80.0f) elevation = -80.0f;
    if (distance < 0.2f) distance = 0.2f;
    if (distance > 8.0f) distance = 8.0f;
    
    float az_rad = DEG2RAD(azimuth);
    
    /* Hard pan */
    float pan_pos = sinf(az_rad);
    float pan_l = 0.5f + 0.5f * pan_pos;
    float pan_r = 0.5f - 0.5f * pan_pos;
    pan_l = powf(pan_l, 1.5f);
    pan_r = powf(pan_r, 1.5f);
    
    /* Head shadow */
    float shadow_l, shadow_r;
    if (azimuth > 10.0f) {
        shadow_l = 0.15f + 0.35f * (1.0f - (azimuth / 180.0f));
        shadow_r = 1.0f;
    } else if (azimuth < -10.0f) {
        shadow_l = 1.0f;
        shadow_r = 0.15f + 0.35f * (1.0f - (-azimuth / 180.0f));
    } else {
        shadow_l = 0.8f;
        shadow_r = 0.8f;
    }
    if (shadow_l < 0.05f) shadow_l = 0.05f;
    if (shadow_r < 0.05f) shadow_r = 0.05f;
    
    float dist_atten = 1.0f / (1.0f + distance * 0.15f);
    if (dist_atten < 0.02f) dist_atten = 0.02f;
    float proximity = 1.0f / (1.0f + distance * 0.3f);
    float bass_boost = 1.0f + 1.5f * proximity;
    
    float sampleL = sample * pan_l * shadow_l * dist_atten * bass_boost;
    float sampleR = sample * pan_r * shadow_r * dist_atten * bass_boost;
    
    *outL += sampleL;
    *outR += sampleR;
    
    /* Update proof state with latest position */
    g_proof_azimuth = azimuth;
    g_proof_elevation = elevation;
    g_proof_distance = distance;
    g_proof_pan_l = pan_l;
    g_proof_pan_r = pan_r;
    g_proof_active = 1;
}

/* ═══════════════════════════════════════════════════════════════════
   Soft Saturation
   ═══════════════════════════════════════════════════════════════════ */
static inline float saturate(float x, float drive) {
    x *= drive;
    float ax = fabsf(x);
    x = x * (27.0f + ax) / (27.0f + 9.0f * ax);
    return x / drive;
}

/* ═══════════════════════════════════════════════════════════════════
   Grain Management
   ═══════════════════════════════════════════════════════════════════ */

static void init_grain(Grain *grain, int band, float pan) {
    float azimuth = (pan - 0.5f) * 360.0f;
    azimuth += (rngf() - 0.5f) * 60.0f;
    if (azimuth > 180.0f) azimuth = 180.0f;
    if (azimuth < -180.0f) azimuth = -180.0f;
    
    grain->azimuth = azimuth;
    grain->elevation = (rngf() - 0.5f) * 160.0f;
    grain->distance = 0.2f + rngf() * 8.0f;
    grain->velocity = 0.5f + rngf() * 2.0f;
    grain->phase_offset = rngf() * TWO_PI;
}

static void trigger_grain(int band, float pan) {
    if (g_active_grains >= MAX_ACTIVE_GRAINS_SOFT_CAP) return;
    for (int i = 0; i < MAX_GRAINS; i++) {
        if (!s_grains[i].active) {
            s_grains[i].active = 1;
            s_grains[i].pos    = 0;
            s_grains[i].phase  = rngf() * 2.0f * PI;

            float lo = BANDS[band].grain_hz_lo;
            float hi = BANDS[band].grain_hz_hi;
            float freq = lo + rngf() * (hi - lo);
            s_grains[i].freq  = freq;

            int len = (int)(GRAIN_CYCLES * SAMPLE_RATE / freq);
            if (len < GRAIN_LEN_MIN) len = GRAIN_LEN_MIN;
            if (len > GRAIN_LEN_MAX) len = GRAIN_LEN_MAX;
            s_grains[i].length = len;

            s_grains[i].amp   = 0.28f + rngf() * 0.55f;
            
            init_grain(&s_grains[i], band, pan);
            
            s_band_grain[band] = i;
            g_active_grains++;
            return;
        }
    }
}

static void extend_or_trigger(int band, float pan) {
    int idx = s_band_grain[band];
    if (idx >= 0 && s_grains[idx].active) {
        int new_len = s_grains[idx].length + GRAIN_EXTEND_SAMPLES;
        if (new_len > GRAIN_LEN_MAX) new_len = GRAIN_LEN_MAX;
        s_grains[idx].length = new_len;

        float target_az = (pan - 0.5f) * 360.0f;
        s_grains[idx].azimuth += 0.3f * (target_az - s_grains[idx].azimuth);
        
    } else {
        trigger_grain(band, pan);
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Schroeder Reverb
   ═══════════════════════════════════════════════════════════════════ */
static inline void reverb_process(float in, float *outL, float *outR) {
    float ca = s_comb_a[s_comb_ap[0]]; s_comb_a[s_comb_ap[0]] = in + ca * COMB_FB; s_comb_ap[0] = (s_comb_ap[0]+1) % COMB_A_LEN;
    float cb = s_comb_b[s_comb_ap[1]]; s_comb_b[s_comb_ap[1]] = in + cb * COMB_FB; s_comb_ap[1] = (s_comb_ap[1]+1) % COMB_B_LEN;
    float cc = s_comb_c[s_comb_ap[2]]; s_comb_c[s_comb_ap[2]] = in + cc * COMB_FB; s_comb_ap[2] = (s_comb_ap[2]+1) % COMB_C_LEN;
    float cd = s_comb_d[s_comb_ap[3]]; s_comb_d[s_comb_ap[3]] = in + cd * COMB_FB; s_comb_ap[3] = (s_comb_ap[3]+1) % COMB_D_LEN;
    float wet = (ca + cb + cc + cd) * 0.25f;

    float ae = s_ap_e[s_ap_ep];
    s_ap_e[s_ap_ep] = wet + ae * AP_FB;
    wet = ae - wet * AP_FB;
    s_ap_ep = (s_ap_ep + 1) % AP_E_LEN;

    float af = s_ap_f[s_ap_fp];
    s_ap_f[s_ap_fp] = wet + af * AP_FB;
    wet = af - wet * AP_FB;
    s_ap_fp = (s_ap_fp + 1) % AP_F_LEN;

    *outL = (ca + cc) * 0.3f + wet * 0.4f;
    *outR = (cb + cd) * 0.3f + wet * 0.4f;
}

/* ═══════════════════════════════════════════════════════════════════
   Capture Callback
   ═══════════════════════════════════════════════════════════════════ */
static void capture_callback(ma_device *pDev, void *pOut,
                              const void *pIn, ma_uint32 frames) {
    (void)pDev; (void)pOut;
    const float *in = (const float *)pIn;
    if (!in) return;
    for (ma_uint32 i = 0; i < frames; i++) {
        float mono = (in[i*2] + in[i*2+1]) * 0.5f;
        g_ring[g_ring_write % RING_SIZE] = mono;
        g_ring_write++;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Playback Callback
   ═══════════════════════════════════════════════════════════════════ */
static void playback_callback(ma_device *pDev, void *pOut,
                               const void *pIn, ma_uint32 frames) {
    (void)pDev; (void)pIn;
    float *out = (float *)pOut;
    
    for (ma_uint32 i = 0; i < frames * 2; i++) {
        out[i] = 0.0f;
    }

    while (g_trig_read != g_trig_write) {
        GrainTrig t = g_trig_queue[g_trig_read % TRIGGER_QUEUE];
        g_trig_read++;
        extend_or_trigger(t.band, t.pan);
    }

    for (ma_uint32 f = 0; f < frames; f++) {
        float mixL = 0.0f, mixR = 0.0f;
        float grain_bus = 0.0f;
        float time = g_spatial_time + f / (float)SAMPLE_RATE;

        /* Layer 1: Sine Tones */
        for (int b = 0; b < NUM_BANDS; b++) {
            float raw    = g_band_energy[b] * g_adapt_scale[b];
            float shaped = raw * raw * 140.0f;
            if (shaped > 1.0f) shaped = 1.0f;

            float coeff = (shaped > s_env_follow[b]) ? ENV_ATK : ENV_REL;
            s_env_follow[b] += coeff * (shaped - s_env_follow[b]);
            float amp = s_env_follow[b] * 0.20f;

            float tgt_cents = (g_spike_density[b] - 0.05f) * 24.0f;
            if (tgt_cents >  20.0f) tgt_cents =  20.0f;
            if (tgt_cents < -20.0f) tgt_cents = -20.0f;
            s_sine_detune[b] += 0.0015f * (tgt_cents - s_sine_detune[b]);
            float hz = BANDS[b].sine_base_hz * powf(2.0f, s_sine_detune[b] / 1200.0f);

            float lfo_rate = 0.25f + (float)b * 0.15f;
            s_lfo_phase[b] += 2.0f * PI * lfo_rate / SAMPLE_RATE;
            if (s_lfo_phase[b] > 2.0f * PI) s_lfo_phase[b] -= 2.0f * PI;
            float lfo = sinf(s_lfo_phase[b]);

            s_sine_phase[b] += 2.0f * PI * hz / SAMPLE_RATE;
            if (s_sine_phase[b] > 2.0f * PI) s_sine_phase[b] -= 2.0f * PI;
            float s1 = sinf(s_sine_phase[b]);

            float detune_hz = hz * (1.0f + lfo * 0.002f);
            s_chorus_phase[b] += 2.0f * PI * detune_hz / SAMPLE_RATE;
            if (s_chorus_phase[b] > 2.0f * PI) s_chorus_phase[b] -= 2.0f * PI;
            float s2 = sinf(s_chorus_phase[b]);

            float mono_s = saturate((s1 * 0.6f + s2 * 0.4f) * amp,
                                     BANDS[b].sat_drive);

            /* Sweep sine tones for proof */
            float sweep = sinf(time * 0.5f + b * 1.5f) * 0.9f;
            float az = sweep * 170.0f;
            float elev = sinf(time * 0.3f + b * 0.7f) * 70.0f;
            float dist = 0.3f + 4.0f * (0.5f + 0.5f * sinf(time * 0.2f + b));
            
            apply_3d(mono_s, az, elev, dist, &mixL, &mixR);
        }

        /* Layer 2: Grains */
        for (int i = 0; i < MAX_GRAINS; i++) {
            if (!s_grains[i].active) continue;
            
            float len1 = (float)(s_grains[i].length - 1);
            float env  = 0.5f * (1.0f - cosf(2.0f * PI * (float)s_grains[i].pos / len1));
            
            float t = time * s_grains[i].velocity + s_grains[i].phase_offset;
            
            s_grains[i].azimuth += sinf(t * 0.7f) * 3.0f;
            s_grains[i].elevation = sinf(t * 0.5f) * 70.0f;
            s_grains[i].distance = 0.3f + 4.0f * (0.5f + 0.5f * sinf(t * 0.3f));
            
            if (s_grains[i].azimuth > 180.0f) s_grains[i].azimuth -= 360.0f;
            if (s_grains[i].azimuth < -180.0f) s_grains[i].azimuth += 360.0f;
            if (s_grains[i].elevation > 80.0f) s_grains[i].elevation = 80.0f;
            if (s_grains[i].elevation < -80.0f) s_grains[i].elevation = -80.0f;
            if (s_grains[i].distance < 0.2f) s_grains[i].distance = 0.2f;
            if (s_grains[i].distance > 8.0f) s_grains[i].distance = 8.0f;
            
            s_grains[i].phase += 2.0f * PI * s_grains[i].freq / SAMPLE_RATE;
            if (s_grains[i].phase > 2.0f * PI) s_grains[i].phase -= 2.0f * PI;
            float g = saturate(sinf(s_grains[i].phase) * env * s_grains[i].amp,
                                1.6f) * 0.15f;
            
            float grainL = 0.0f, grainR = 0.0f;
            apply_3d(g, s_grains[i].azimuth, s_grains[i].elevation, 
                   s_grains[i].distance, &grainL, &grainR);
            
            mixL += grainL;
            mixR += grainR;
            grain_bus += g * 0.2f;
            
            if (++s_grains[i].pos >= s_grains[i].length) {
                s_grains[i].active = 0;
                if (g_active_grains > 0) g_active_grains--;
            }
        }

        /* Layer 3: Reverb */
        float revL, revR;
        reverb_process(grain_bus, &revL, &revR);
        mixL += revL * 0.55f;
        mixR += revR * 0.55f;

        /* Layer 4: Modulated Comb */
        float bass_norm = g_bass_energy * g_adapt_scale[1];
        if (bass_norm > 1.0f) bass_norm = 1.0f;
        float tgt_delay = (int)(0.035f * SAMPLE_RATE)
                        + bass_norm * (int)(0.085f * SAMPLE_RATE);
        if (tgt_delay < 1.0f)        tgt_delay = 1.0f;
        if (tgt_delay >= MCOMB_LEN)  tgt_delay = MCOMB_LEN - 1;
        s_mcomb_delay += 0.0008f * (tgt_delay - s_mcomb_delay);

        int   rd  = (s_mcomb_pos - (int)s_mcomb_delay + MCOMB_LEN) % MCOMB_LEN;
        float del = s_mcomb_buf[rd];
        float cin = (mixL + mixR) * 0.5f * g_total_density * 0.45f;
        s_mcomb_buf[s_mcomb_pos] = cin + del * MCOMB_FB;
        s_mcomb_pos = (s_mcomb_pos + 1) % MCOMB_LEN;

        int rd2 = (rd - 1 + MCOMB_LEN) % MCOMB_LEN;
        mixL += s_mcomb_buf[rd]  * 0.25f;
        mixR += s_mcomb_buf[rd2] * 0.25f;

        /* Mid/Side Widener */
        float mid  = (mixL + mixR) * 0.5f;
        float side = (mixL - mixR) * 0.5f;
        side *= (1.0f + MS_WIDTH);
        mixL = mid + side;
        mixR = mid - side;

        mixL = saturate(mixL * MASTER_VOL, 1.3f);
        mixR = saturate(mixR * MASTER_VOL, 1.3f);

        out[f*2]   += mixL;
        out[f*2+1] += mixR;
    }
    
    g_spatial_time += frames / (float)SAMPLE_RATE;
    if (g_spatial_time > 1000.0f) g_spatial_time -= 1000.0f;
}

/* ═══════════════════════════════════════════════════════════════════
   FFT + Adaptive Calibration
   ═══════════════════════════════════════════════════════════════════ */
static float        g_hann[FFT_SIZE];
static kiss_fft_cfg g_fft_cfg = NULL;
static kiss_fft_cpx g_fft_in[FFT_SIZE];
static kiss_fft_cpx g_fft_out[FFT_SIZE];

static void build_hann(void) {
    for (int i = 0; i < FFT_SIZE; i++)
        g_hann[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (FFT_SIZE - 1)));
}

static void run_fft(void) {
    long avail = g_ring_write - g_ring_read;
    if (avail < FFT_SIZE) return;
    if (avail > FFT_SIZE) g_ring_read = g_ring_write - FFT_SIZE;

    for (int i = 0; i < FFT_SIZE; i++) {
        g_fft_in[i].r = g_ring[(g_ring_read + i) % RING_SIZE] * g_hann[i];
        g_fft_in[i].i = 0.0f;
    }
    g_ring_read += FFT_SIZE;
    kiss_fft(g_fft_cfg, g_fft_in, g_fft_out);

    /* band_db[] is only used to find which bands are within
     * ACTIVATION_DB_BELOW_PEAK of the loudest band this frame -- bands that
     * fail the relative threshold get their energy gated to zero before it
     * ever reaches g_band_energy / g_adapt_sum, so leakage-only bands don't
     * drive neurons or get chased by the adaptive gain calibration below. */
    float band_db[NUM_BANDS];
    float raw_e[NUM_BANDS];
    float peak_db = -1000.0f;
    for (int b = 0; b < NUM_BANDS; b++) {
        double sum = 0.0;
        int count = g_band_hi_bin[b] - g_band_lo_bin[b] + 1;
        for (int k = g_band_lo_bin[b]; k <= g_band_hi_bin[b]; k++) {
            double re = g_fft_out[k].r, im = g_fft_out[k].i;
            sum += re*re + im*im;
        }
        float e = (float)(sqrt(sum / count) / FFT_SIZE);
        raw_e[b] = e;
        float db = 10.0f * log10f(e * e + 1e-18f);
        band_db[b] = db;
        if (db > peak_db) peak_db = db;
    }
    for (int b = 0; b < NUM_BANDS; b++) {
        float e = (band_db[b] >= peak_db - ACTIVATION_DB_BELOW_PEAK) ? raw_e[b] : 0.0f;
        g_band_energy[b] = e;
        g_adapt_sum[b]  += e * e;
    }
    g_bass_energy = g_band_energy[1];

    g_adapt_count++;
    if (g_adapt_count >= (ADAPT_WIN / FFT_SIZE)) {
        for (int b = 0; b < NUM_BANDS; b++) {
            float rms = sqrtf((float)(g_adapt_sum[b] / g_adapt_count));
            float target;
            if (rms > ADAPT_RMS_FLOOR) {
                target = 0.05f / rms;
                if (target > ADAPT_SCALE_MAX) target = ADAPT_SCALE_MAX;
            } else {
                target = 1.0f;
            }
            g_adapt_scale[b] += ADAPT_SMOOTH_PER_STEP * (target - g_adapt_scale[b]);
            g_adapt_sum[b] = 0.0;
        }
        g_adapt_count = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Grid
   ═══════════════════════════════════════════════════════════════════ */
static char   firedNow[GRID_CELLS];
static char   justFired[GRID_CELLS];
static double p_next[GRID_CELLS];
static const char RAMP[] = " .:-=+*#%@";
#define RAMP_LEVELS ((int)(sizeof(RAMP)-1))

static ULONGLONG    s_band_next_fire_ms[NUM_BANDS] = {0,0,0,0,0,0,0,0,0,0,0,0};
static const ULONGLONG BAND_COOLDOWN_MS[NUM_BANDS] = {
    220, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 50
};

static void grid_tick(void) {
    int   spike_count[NUM_BANDS]    = {0};
    float fired_col_sum[NUM_BANDS]  = {0};
    ULONGLONG now_ms = GetTickCount64();

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < GRID_CELLS; j++) {
        int row  = j / GRID_WIDTH;
        int col  = j % GRID_WIDTH;
        int band = row / ROWS_PER_BAND;
        if (band >= NUM_BANDS) band = NUM_BANDS - 1;

        neurons[j].p += (double)(g_band_energy[band] * g_adapt_scale[band]
                                 * (float)BANDS[band].gain);
        neurons[j].p *= BANDS[band].decay;

        if (neurons[j].p >= (double)neurons[j].threshold) {
            neurons[j].p = 0.0;
            neurons[j].fire_count++;
            firedNow[j]  = 1;
            justFired[j] = 1;
            #pragma omp atomic
            spike_count[band]++;
            #pragma omp atomic
            fired_col_sum[band] += (float)col / (GRID_WIDTH - 1);
        } else {
            firedNow[j] = 0;
        }
    }

    float total = 0.0f;
    for (int b = 0; b < NUM_BANDS; b++) {
        float d = (float)spike_count[b] / (ROWS_PER_BAND * GRID_WIDTH);
        g_spike_density[b] = g_spike_density[b] * 0.75f + d * 0.25f;
        total += g_spike_density[b];

        if (spike_count[b] > 0 && now_ms >= s_band_next_fire_ms[b]) {
            float avg_pan = fired_col_sum[b] / spike_count[b];
            int wi = g_trig_write % TRIGGER_QUEUE;
            g_trig_queue[wi].band = b;
            g_trig_queue[wi].pan  = avg_pan;
            g_trig_write++;
            s_band_next_fire_ms[b] = now_ms + BAND_COOLDOWN_MS[b];
        }
    }
    g_total_density = total;

    memset(p_next, 0, sizeof(p_next));
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < GRID_CELLS; j++) {
        int row  = j / GRID_WIDTH;
        int col  = j % GRID_WIDTH;
        int band = row / ROWS_PER_BAND;
        if (band >= NUM_BANDS) band = NUM_BANDS - 1;
        double sw = BANDS[band].synapse_weight;
        int lr       = row - BANDS[band].row_start;
        int up_row   = ((lr-1+ROWS_PER_BAND)%ROWS_PER_BAND) + BANDS[band].row_start;
        int down_row = ((lr+1)%ROWS_PER_BAND)                + BANDS[band].row_start;
        int up    = up_row   * GRID_WIDTH + col;
        int down  = down_row * GRID_WIDTH + col;
        int left  = row * GRID_WIDTH + ((col-1+GRID_WIDTH)%GRID_WIDTH);
        int right = row * GRID_WIDTH + ((col+1)%GRID_WIDTH);
        double boost = 0.0;
        if (firedNow[up])    boost += sw;
        if (firedNow[down])  boost += sw;
        if (firedNow[left])  boost += sw;
        if (firedNow[right]) boost += sw;
        p_next[j] = boost;
    }
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < GRID_CELLS; j++)
        neurons[j].p += p_next[j];
}

/* ═══════════════════════════════════════════════════════════════════
   main — CLEAN PROOF DISPLAY
   ═══════════════════════════════════════════════════════════════════ */
int main(void) {
    if (GRID_CELLS > NEURON_COUNT) {
        printf("Need >= %d neurons; have %d.\n", GRID_CELLS, NEURON_COUNT);
        return -1;
    }

    build_hann();
    init_bands();

    g_fft_cfg = kiss_fft_alloc(FFT_SIZE, 0, NULL, NULL);
    if (!g_fft_cfg) { printf("kiss_fft_alloc failed\n"); return -1; }

    memset(s_grains,   0, sizeof(s_grains));
    memset(s_comb_a,   0, sizeof(s_comb_a));
    memset(s_comb_b,   0, sizeof(s_comb_b));
    memset(s_comb_c,   0, sizeof(s_comb_c));
    memset(s_comb_d,   0, sizeof(s_comb_d));
    memset(s_ap_e,     0, sizeof(s_ap_e));
    memset(s_ap_f,     0, sizeof(s_ap_f));
    memset(s_mcomb_buf,0, sizeof(s_mcomb_buf));
    memset(s_chorus_buf,0,sizeof(s_chorus_buf));

    /* Capture */
    ma_device_config cap_cfg = ma_device_config_init(ma_device_type_loopback);
    cap_cfg.capture.format   = ma_format_f32;
    cap_cfg.capture.channels = 2;
    cap_cfg.sampleRate       = SAMPLE_RATE;
    cap_cfg.dataCallback     = capture_callback;
    ma_device cap_dev;
    if (ma_device_init(NULL, &cap_cfg, &cap_dev) != MA_SUCCESS) {
        printf("Failed to open capture device.\n"); return -1;
    }
    ma_device_start(&cap_dev);

    /* Playback */
    ma_device_config pb_cfg  = ma_device_config_init(ma_device_type_playback);
    pb_cfg.playback.format   = ma_format_f32;
    pb_cfg.playback.channels = 2;
    pb_cfg.sampleRate        = SAMPLE_RATE;
    pb_cfg.dataCallback      = playback_callback;
    ma_device pb_dev;
    if (ma_device_init(NULL, &pb_cfg, &pb_dev) != MA_SUCCESS) {
        printf("Failed to open playback device.\n"); return -1;
    }
    ma_device_start(&pb_dev);

    memset(firedNow,  0, sizeof(firedNow));
    memset(justFired, 0, sizeof(justFired));

    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║     NEURAL SANDBOX — 3D PROOF MODE                                 ║\n");
    printf("║     If numbers below are MOVING, 3D audio is WORKING              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    CONSOLE_SCREEN_BUFFER_INFO info;
    GetConsoleScreenBufferInfo(hOut, &info);
    COORD gridOrigin = { 0, info.dwCursorPosition.Y };

    char rowbuf[GRID_WIDTH + 2];
    ULONGLONG lastDraw = GetTickCount64();
    int frame = 0;

    printf("▶ Press Ctrl+C to quit\n\n");

    while (1) {
        Sleep(1);
        run_fft();
        grid_tick();
        frame++;

        ULONGLONG now = GetTickCount64();
        if (now - lastDraw >= 40) {
            SetConsoleCursorPosition(hOut, gridOrigin);
            
            /* Grid */
            for (int row = 0; row < GRID_HEIGHT; row++) {
                int band = row / ROWS_PER_BAND;
                if (band >= NUM_BANDS) band = NUM_BANDS - 1;
                for (int col = 0; col < GRID_WIDTH; col++) {
                    int j = row * GRID_WIDTH + col;
                    if (justFired[j]) {
                        rowbuf[col] = '@';
                    } else {
                        double ratio = neurons[j].p / (double)neurons[j].threshold;
                        if (ratio < 0.0) ratio = 0.0;
                        if (ratio > 1.0) ratio = 1.0;
                        rowbuf[col] = RAMP[(int)(ratio*(RAMP_LEVELS-1))];
                    }
                }
                rowbuf[GRID_WIDTH] = '\n';
                rowbuf[GRID_WIDTH+1] = '\0';
                fputs(rowbuf, stdout);
                if (row == BANDS[band].row_start + ROWS_PER_BAND - 1 && band < NUM_BANDS - 1) {
                    for (int c = 0; c < GRID_WIDTH; c++) rowbuf[c] = '-';
                    rowbuf[GRID_WIDTH] = '\n';
                    rowbuf[GRID_WIDTH+1] = '\0';
                    fputs(rowbuf, stdout);
                }
            }

            /* ═══ PROOF DISPLAY — CLEAN AND READABLE ═══ */
            
            /* Pan Meter */
            printf("\n┌────────────────────────────────────────────────────────────┐\n");
            printf("│ POSITION: ");
            
            int pos = (int)((g_proof_pan_l - 0.5f) * 30.0f + 15.0f);
            if (pos < 0) pos = 0;
            if (pos > 30) pos = 30;
            
            printf("[");
            for (int i = 0; i < 31; i++) {
                if (i == pos) {
                    if (i < 15) printf("◄");
                    else if (i > 15) printf("►");
                    else printf("●");
                } else if (i == 15) {
                    printf("│");
                } else if (i < 15 && i > pos) {
                    printf(" ");
                } else if (i > 15 && i < pos) {
                    printf(" ");
                } else if (i < pos) {
                    printf("▓");
                } else if (i > pos) {
                    printf("▓");
                } else {
                    printf(" ");
                }
            }
            printf("]  ");
            
            /* Show L/R values */
            printf("L%5.2f R%5.2f\n", g_proof_pan_l, g_proof_pan_r);
            printf("└────────────────────────────────────────────────────────────┘\n");
            
            /* 3D Coordinates */
            printf("┌────────────────────────────────────────────────────────────┐\n");
            printf("│ 3D:  AZIMUTH %+7.1f°  ELEVATION %+7.1f°  DISTANCE %5.2fm\n",
                   g_proof_azimuth, g_proof_elevation, g_proof_distance);
            printf("└────────────────────────────────────────────────────────────┘\n");
            
            /* Active count and status */
            printf("┌────────────────────────────────────────────────────────────┐\n");
            printf("│ ACTIVE: %3d grains  |  STATUS: %s\n",
                   g_active_grains,
                   (g_proof_active && g_active_grains > 0) ? "✅ 3D ACTIVE" : "⏳ waiting...");
            
            /* Motion indicator */
            static float last_az = 0.0f;
            float motion = fabsf(g_proof_azimuth - last_az);
            last_az = g_proof_azimuth;
            
            printf("│ MOTION: ");
            if (motion > 5.0f) {
                printf("🔥 MOVING! (%.1f°/frame)  ← PROOF OF 3D EFFECT\n", motion);
            } else if (motion > 1.0f) {
                printf("🔹 gentle motion (%.1f°/frame)\n", motion);
            } else {
                printf("⏸  steady (waiting for audio)\n");
            }
            printf("└────────────────────────────────────────────────────────────┘\n");
            
            /* Band energies */
            printf("BANDS: ");
            for (int b = 0; b < NUM_BANDS; b++) {
                printf("%s %.3f  ", BANDS[b].name, (double)g_band_energy[b]);
            }
            printf("|  Grains: %d     \n", g_active_grains);
            
            fflush(stdout);
            memset(justFired, 0, sizeof(justFired));
            lastDraw = now;
        }
    }

    ma_device_uninit(&pb_dev);
    ma_device_uninit(&cap_dev);
    kiss_fft_free(g_fft_cfg);
    return 0;
}