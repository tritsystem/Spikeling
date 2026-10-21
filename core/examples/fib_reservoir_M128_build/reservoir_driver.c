/* Stage 2 driver, device idea #17: M=128 Fibonacci-coupled Resonator chain
 * on the REAL generated C backend (spikeling_hw.c/.h, from fib_reservoir_M128.spk,
 * unmodified codegen output).
 *
 * WHY THIS FILE EXISTS AND ISN'T PART OF THE CODEGEN ITSELF: compiler.py's
 * CCodeGenerator explicitly does not route synapses to Resonators ("Resonators
 * don't currently participate in synapses" -- read directly in compiler.py,
 * not assumed). So the Fibonacci nearest-neighbor coupling (the K@x term in
 * fib_noise_combined_check_M128.py's run()) is applied HERE, externally, one
 * scalar `drive` per resonator per substep, exactly mirroring what the Python
 * reference does outside ResonatorState.step() -- each individual resonator's
 * own dynamics (spikeling_resonator_step) are the real, unmodified, generated
 * function, called unchanged.
 *
 * CROSS-BACKEND TARGET: fib_reservoir_data.h carries the exact seed=0, M=128
 * clean-case NMSE this should reproduce: 0.4464 (FIB_PY_NMSE). This is a
 * conformance check, not a fresh claim -- same discipline as the existing
 * 4-bar cross-backend LIF/Izhikevich/AdEx conformance harness.
 */
#include "spikeling_hw.h"
#include "fib_reservoir_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double Xfeat[FIB_T - FIB_BURN][2 * FIB_M];

int main(void) {
    if (RESONATOR_COUNT != FIB_M) {
        fprintf(stderr, "FATAL: RESONATOR_COUNT=%d != FIB_M=%d\n", RESONATOR_COUNT, FIB_M);
        return 1;
    }

    static double x_snapshot[FIB_M];

    for (int t = 0; t < FIB_T; t++) {
        for (int h = 0; h < FIB_HOLD; h++) {
            for (int i = 0; i < FIB_M; i++) x_snapshot[i] = (double)resonators[i].x;

            for (int i = 0; i < FIB_M; i++) {
                double coupling_term = 0.0;
                if (i > 0)             coupling_term += fib_bonds[i - 1] * x_snapshot[i - 1];
                if (i < FIB_M - 1)     coupling_term += fib_bonds[i]     * x_snapshot[i + 1];
                double drive = (double)fib_win[i] * (double)fib_u[t] + coupling_term;
                spikeling_resonator_step((uint32_t)i, (float)drive, FIB_DT, (float)t);
            }
        }
        if (t >= FIB_BURN) {
            int row = t - FIB_BURN;
            for (int i = 0; i < FIB_M; i++) {
                double xv = (double)resonators[i].x;
                Xfeat[row][i]         = xv;
                Xfeat[row][FIB_M + i] = xv * xv;
            }
        }
    }

    /* Test-split NMSE, same slicing as the Python script: rows [FIB_TR, len) */
    int n_burned = FIB_T - FIB_BURN;
    int n_test = n_burned - FIB_TR;
    double sse = 0.0, y_mean = 0.0;
    for (int r = FIB_TR; r < n_burned; r++) y_mean += fib_y_burned[r];
    y_mean /= n_test;

    double y_var = 0.0;
    for (int r = FIB_TR; r < n_burned; r++) y_var += (fib_y_burned[r] - y_mean) * (fib_y_burned[r] - y_mean);
    y_var /= n_test;

    for (int r = FIB_TR; r < n_burned; r++) {
        double pred = 0.0;
        for (int c = 0; c < 2 * FIB_M; c++) pred += Xfeat[r][c] * fib_decoder_weights[c];
        double err = pred - fib_y_burned[r];
        sse += err * err;
    }
    double nmse = (sse / n_test) / (y_var + 1e-12);

    printf("C reservoir_driver: M=%d, %d test rows\n", FIB_M, n_test);
    printf("  C-backend NMSE   = %.6f\n", nmse);
    printf("  Python NMSE      = %.6f  (FIB_PY_NMSE, same seed/config)\n", FIB_PY_NMSE);
    printf("  relative diff    = %.4f%%\n", 100.0 * fabs(nmse - FIB_PY_NMSE) / FIB_PY_NMSE);
    return 0;
}
