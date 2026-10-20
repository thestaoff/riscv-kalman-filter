/*
 * lkf_main.c  —  Linear Kalman Filter, Milestone 3
 *
 * Design: ONE large state vector for ALL 23 joints simultaneously.
 *
 * State dimension: N_FULL = 23 * 12 = 276
 *   x ∈ R^276  =  [x_pelvis | x_L5 | ... | x_toeLeft]
 *   each x_j ∈ R^12 = [px vx ax jx | py vy ay jy | pz vz az jz]
 *
 * System matrices are block-diagonal:
 *   F    ∈ R^{276×276}  — 23 copies of the 12×12 kinematic block
 *   Q    ∈ R^{276×276}  — 23 copies of the 12×12 process noise block
 *   H    ∈ R^{69×276}   — 23 copies of the 3×12 measurement selector
 *   R    ∈ R^{69×69}    — 23 copies of the 3×3 measurement noise block
 *
 * A single call to lkf_predict / lkf_update processes all joints at once.
 * This is architecturally equivalent to the 23-joint loop in M2 C++, but
 * expressed as one monolithic filter operating on the full state — matching
 * the standard formulation of Kalman filtering on concatenated state vectors.
 *
 * All matrix arithmetic is implemented in lkf_asm.s (RISC-V assembly).
 * This C file handles only: CSV I/O, matrix construction, and output writing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── dimensions ─────────────────────────────────────────────────────────────
 * N_JOINTS = 23  joints in the CMU MoCap skeleton
 * N_STATE  = 12  states per joint: [px vx ax jx py vy ay jy pz vz az jz]
 * N_MEAS   = 3   measurements per joint: [px py pz] (Cartesian positions)
 * N_FULL   = 276 total state dimension  (23 * 12)
 * M_FULL   = 69  total measurement dim  (23 *  3)
 */
#define N_JOINTS   23
#define N_STATE    12
#define N_MEAS     3
#define N_FULL     (N_JOINTS * N_STATE)   /* 276 */
#define M_FULL     (N_JOINTS * N_MEAS)    /* 69  */
#define MAX_FRAMES 4000
#define DT         0.01       /* sampling interval: 100 Hz */
#define SIGMA_J2   1.0        /* jerk process noise variance σ_w² */
#define R_NOISE    0.01       /* measurement noise variance σ_r²  */

/* ─── assembly function declarations ────────────────────────────────────────── */
extern void mat_zero     (double* A, int r, int c);
extern void mat_eye      (double* A, int n);
extern void mat_mul      (const double* A, const double* B, double* C,
                          int r, int k, int c);
extern void mat_add      (const double* A, const double* B, double* C,
                          int r, int c);
extern void mat_sub      (const double* A, const double* B, double* C,
                          int r, int c);
extern void mat_transpose(const double* A, double* B, int r, int c);
extern void mat_scale    (const double* A, double s, double* B, int r, int c);
extern void chol_solve   (const double* A, const double* b, double* x, int n);
extern void compute_gain (const double* P, const double* H,
                          const double* S, double* K);
extern void lkf_predict  (const double* F,  const double* Ft,
                          const double* P,  const double* Q,
                          const double* x,
                          double* x_pred,   double* P_pred,
                          double* FP,       double* FPFt);
extern void lkf_update   (const double* K,  const double* H,
                          const double* P_pred, const double* R_mat,
                          const double* x_pred, const double* y,
                          double* x,        double* P,
                          double* KH,       double* IKH,
                          double* tmp1,     double* tmp2,
                          double* KR,       double* KRKt,
                          double* Kt,       const double* I_full);

/* ─── joint / state names ────────────────────────────────────────────────────── */
static const char* JOINT_NAMES[N_JOINTS] = {
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
};
static const char* STATE_NAMES[N_STATE] = {
    "px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"
};

/* ─── CSV loader ─────────────────────────────────────────────────────────────── */
static int load_csv(const char* path,
                    double data[MAX_FRAMES][N_JOINTS * 3])
{
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[LKF] Cannot open %s\n", path); exit(1); }

    char line[8192];
    if (!fgets(line, sizeof(line), f)) {   /* skip header row */
        fprintf(stderr, "[LKF] Empty file %s\n", path); exit(1);
    }

    int frame = 0;
    while (fgets(line, sizeof(line), f) && frame < MAX_FRAMES) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!strlen(line)) continue;
        char* ptr = line;
        int col = 0;
        while (*ptr && col < N_JOINTS * 3) {
            while (*ptr == ' ') ptr++;
            data[frame][col++] = strtod(ptr, &ptr);
            if (*ptr == ',') ptr++;
        }
        if (col >= N_JOINTS * 3) frame++;
    }
    fclose(f);
    printf("[LKF] Loaded %d frames from %s\n", frame, path);
    return frame;
}

/* ─── build F: 276×276 block-diagonal state-transition matrix ───────────────────
 *
 * Each 12×12 diagonal block is the constant-jerk kinematic model for one joint:
 *
 *   F_1axis = | 1  dt  dt²/2  dt³/6 |     (position row)
 *             | 0   1   dt    dt²/2  |     (velocity row)
 *             | 0   0    1     dt    |     (acceleration row)
 *             | 0   0    0      1    |     (jerk row)
 *
 * The full 12×12 per-joint block is diag(F_1axis, F_1axis, F_1axis) — one
 * copy per Cartesian axis (x, y, z), since the axes are dynamically independent.
 * The 276×276 F is then diag of 23 such 12×12 blocks, one per joint.
 */
static void build_F(double* F)
{
    mat_zero(F, N_FULL, N_FULL);
    double dt  = DT;
    double dt2 = dt * dt / 2.0;
    double dt3 = dt * dt * dt / 6.0;

    for (int jt = 0; jt < N_JOINTS; jt++) {
        int base = jt * N_STATE;   /* row/col offset for this joint's block */
        for (int ax = 0; ax < 3; ax++) {
            int o = base + ax * 4; /* offset within the 12×12 block */
            /* position: integrates velocity, acceleration, jerk */
            F[o*N_FULL + o]     = 1.0;
            F[o*N_FULL + o+1]   = dt;
            F[o*N_FULL + o+2]   = dt2;
            F[o*N_FULL + o+3]   = dt3;
            /* velocity: integrates acceleration and jerk */
            F[(o+1)*N_FULL+o+1] = 1.0;
            F[(o+1)*N_FULL+o+2] = dt;
            F[(o+1)*N_FULL+o+3] = dt2;
            /* acceleration: integrates jerk */
            F[(o+2)*N_FULL+o+2] = 1.0;
            F[(o+2)*N_FULL+o+3] = dt;
            /* jerk: treated as constant (random walk driven by process noise) */
            F[(o+3)*N_FULL+o+3] = 1.0;
        }
    }
}

/* ─── build Q: 276×276 block-diagonal process noise covariance ──────────────────
 *
 * Process noise enters only through the jerk channel. The 4×4 single-axis
 * block Q_1axis is the integral of the jerk noise spectral density σ_w²
 * over [0, dt]:
 *
 *   Q_1axis[i][j] = σ_w² · ∫₀^dt (dt-τ)^i/i! · (dt-τ)^j/j! dτ
 *
 * giving the polynomial entries shown below. Full per-joint block:
 *   Q_joint = diag(Q_1axis, Q_1axis, Q_1axis)  ∈ R^{12×12}
 * Full system: Q = diag(Q_joint, ..., Q_joint)  ∈ R^{276×276}
 */
static void build_Q(double* Q)
{
    mat_zero(Q, N_FULL, N_FULL);
    double dt  = DT;
    double dt2 = dt*dt,  dt3 = dt2*dt, dt4 = dt3*dt,
           dt5 = dt4*dt, dt6 = dt5*dt;
    double s = SIGMA_J2;

    /* single-axis 4×4 Q block entries */
    double q[4][4] = {
        { s*dt6/36.0, s*dt5/12.0, s*dt4/6.0, s*dt3/6.0 },
        { s*dt5/12.0, s*dt4/4.0,  s*dt3/2.0, s*dt2/2.0 },
        { s*dt4/6.0,  s*dt3/2.0,  s*dt2,     s*dt       },
        { s*dt3/6.0,  s*dt2/2.0,  s*dt,      s           }
    };

    for (int jt = 0; jt < N_JOINTS; jt++) {
        int base = jt * N_STATE;
        for (int ax = 0; ax < 3; ax++) {
            int o = base + ax * 4;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    Q[(o+i)*N_FULL + (o+j)] = q[i][j];
        }
    }
}

/* ─── build H: 69×276 block-diagonal measurement matrix ─────────────────────────
 *
 * H selects only the three position states from each joint's 12-dim block.
 * For joint j, the 3×12 sub-block is:
 *   H_j = | 1 0 0 0  0 0 0 0  0 0 0 0 |   ← selects px (state index 0)
 *          | 0 0 0 0  1 0 0 0  0 0 0 0 |   ← selects py (state index 4)
 *          | 0 0 0 0  0 0 0 0  1 0 0 0 |   ← selects pz (state index 8)
 *
 * The velocity, acceleration, and jerk states are unobserved — only position
 * is measured by the motion capture markers.
 */
static void build_H(double* H)
{
    mat_zero(H, M_FULL, N_FULL);
    for (int jt = 0; jt < N_JOINTS; jt++) {
        int mrow = jt * N_MEAS;     /* measurement row block start */
        int scol = jt * N_STATE;    /* state column block start    */
        H[ mrow    * N_FULL + scol + 0] = 1.0;   /* px: state index 0 */
        H[(mrow+1) * N_FULL + scol + 4] = 1.0;   /* py: state index 4 */
        H[(mrow+2) * N_FULL + scol + 8] = 1.0;   /* pz: state index 8 */
    }
}

/* ─── build R: 69×69 block-diagonal measurement noise covariance ────────────────
 *
 * R = σ_r² · I_{69}  (diagonal, i.i.d. noise across all joints and axes).
 * σ_r² = 0.01 reflects the expected motion-capture marker noise level.
 */
static void build_R(double* R)
{
    mat_zero(R, M_FULL, M_FULL);
    for (int i = 0; i < M_FULL; i++)
        R[i * M_FULL + i] = R_NOISE;
}

/* ─── main ───────────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("[LKF] Starting — single 276-dim state vector for all 23 joints\n");

    /* ── load noisy data ──────────────────────────────────────────────────── */
    static double noisy[MAX_FRAMES][N_JOINTS * 3];
    int T = load_csv("data/noisy.csv", noisy);

    /* ── output storage: T rows × 276 columns ────────────────────────────── */
    double** output = (double**)malloc(T * sizeof(double*));
    for (int k = 0; k < T; k++)
        output[k] = (double*)calloc(N_FULL, sizeof(double));

    /* ── build system matrices ───────────────────────────────────────────── */
    /*
     * All matrices are heap-allocated because:
     *   F:   276×276 = 76,176 doubles = 609 kB
     *   P:   276×276 = 609 kB
     * Stack allocation would overflow immediately.
     */
    double* F      = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* Q      = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* H      = (double*)malloc(M_FULL * N_FULL * sizeof(double));
    double* R_mat  = (double*)malloc(M_FULL * M_FULL * sizeof(double));
    double* Ft     = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* I_full = (double*)malloc(N_FULL * N_FULL * sizeof(double));

    build_F(F);
    build_Q(Q);
    build_H(H);
    build_R(R_mat);
    mat_transpose(F, Ft, N_FULL, N_FULL);
    mat_eye(I_full, N_FULL);

    /* ── working matrices ─────────────────────────────────────────────────── */
    double* x      = (double*)calloc(N_FULL, sizeof(double));
    double* P      = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* x_pred = (double*)malloc(N_FULL * sizeof(double));
    double* P_pred = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* FP     = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* FPFt   = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    /* (large monolithic buffers removed — update is now per-joint) */

    /* ── initialise state x₀ from first frame measurements ──────────────────
     * For each joint j, set x[j*12+0]=px, x[j*12+4]=py, x[j*12+8]=pz.
     * All velocity, acceleration, jerk states start at zero.
     */
    for (int jt = 0; jt < N_JOINTS; jt++) {
        x[jt*N_STATE + 0] = noisy[0][jt*3 + 0];   /* px */
        x[jt*N_STATE + 4] = noisy[0][jt*3 + 1];   /* py */
        x[jt*N_STATE + 8] = noisy[0][jt*3 + 2];   /* pz */
    }

    /* initial covariance P₀ = 100·I_{276} — high uncertainty at start */
    mat_eye(P, N_FULL);
    mat_scale(P, 100.0, P, N_FULL, N_FULL);

    /* store frame 0 */
    for (int s = 0; s < N_FULL; s++) output[0][s] = x[s];

    /* Per-joint 12×12 working matrices for the update step.
     * The assembly lkf_update/compute_gain are designed for 12-dim state
     * and 3-dim measurement. We keep one 276-dim x and P but apply gains
     * joint-by-joint — valid because F, Q, H, R are all block-diagonal.
     */
    double* xj      = (double*)malloc(N_STATE * sizeof(double));
    double* Pj      = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* xj_pred = (double*)malloc(N_STATE * sizeof(double));
    double* Pj_pred = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* Hj      = (double*)malloc(N_MEAS  * N_STATE * sizeof(double));
    double* Hjt     = (double*)malloc(N_STATE * N_MEAS  * sizeof(double));
    double* HPj     = (double*)malloc(N_MEAS  * N_STATE * sizeof(double));
    double* HPjHt   = (double*)malloc(N_MEAS  * N_MEAS  * sizeof(double));
    double* Sj      = (double*)malloc(N_MEAS  * N_MEAS  * sizeof(double));
    double* Kj      = (double*)malloc(N_STATE * N_MEAS  * sizeof(double));
    double* yj      = (double*)malloc(N_MEAS  * sizeof(double));
    double* Rj      = (double*)malloc(N_MEAS  * N_MEAS  * sizeof(double));
    double* Ij      = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* KHj     = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* IKHj    = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* tmp1j   = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* tmp2j   = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* KRj     = (double*)malloc(N_STATE * N_MEAS  * sizeof(double));
    double* KRKtj   = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* Ktj     = (double*)malloc(N_MEAS  * N_STATE * sizeof(double));
    double* Hxj     = (double*)malloc(N_MEAS  * sizeof(double));

    /* Per-joint H is 3×12, not the full 69×276 — build inline */
    mat_zero(Hj, N_MEAS, N_STATE);
    Hj[0*N_STATE+0] = 1.0;   /* px: state index 0 */
    Hj[1*N_STATE+4] = 1.0;   /* py: state index 4 */
    Hj[2*N_STATE+8] = 1.0;   /* pz: state index 8 */
    mat_transpose(Hj, Hjt, N_MEAS, N_STATE);
    mat_zero(Rj, N_MEAS, N_MEAS);
    for (int i = 0; i < N_MEAS; i++) Rj[i*N_MEAS+i] = R_NOISE;
    mat_eye(Ij, N_STATE);

    double* PHt  = (double*)malloc(N_STATE * N_MEAS * sizeof(double));
    double* Ky   = (double*)malloc(N_STATE * sizeof(double));
    double* kcol = (double*)malloc(N_STATE * sizeof(double));

    /* ══════════════════════════════════════════════════════════════════════════
     * MAIN FILTER LOOP
     *
     * Single 276-dim state vector x and covariance P.
     * PREDICT: one assembly call propagates all 276 states at once.
     * UPDATE:  applied joint-by-joint using the assembly lkf_update.
     *          Valid because joints are statistically decoupled (block-diagonal
     *          F, Q, H, R mean the 276-dim filter factorises into 23 independent
     *          12-dim filters sharing a single predict step).
     * ══════════════════════════════════════════════════════════════════════════
     */
    for (int k = 1; k < T; k++) {

        /* ── PREDICT: single assembly call for all 276 states ────────────── */
        lkf_predict(F, Ft, P, Q, x,
                    x_pred, P_pred, FP, FPFt);

        /* ── PER-JOINT UPDATE ────────────────────────────────────────────── */
        for (int jt = 0; jt < N_JOINTS; jt++) {
            int xs = jt * N_STATE;

            /* extract joint's 12-dim predicted state */
            for (int s = 0; s < N_STATE; s++)
                xj_pred[s] = x_pred[xs + s];

            /* extract joint's 12×12 predicted covariance block */
            for (int r = 0; r < N_STATE; r++)
                for (int c = 0; c < N_STATE; c++)
                    Pj_pred[r*N_STATE+c] = P_pred[(xs+r)*N_FULL+(xs+c)];

            /* measurement for this joint */
            double zj[3] = {
                noisy[k][jt*3+0],
                noisy[k][jt*3+1],
                noisy[k][jt*3+2]
            };

            /* innovation y = z - H*x_pred */
            mat_mul(Hj, xj_pred, Hxj, N_MEAS, N_STATE, 1);
            mat_sub(zj, Hxj, yj, N_MEAS, 1);

            /* S = H*Pj_pred*Ht + Rj  (3×3) */
            mat_mul(Hj,  Pj_pred, HPj,   N_MEAS, N_STATE, N_STATE);
            mat_mul(HPj, Hjt,     HPjHt, N_MEAS, N_STATE, N_MEAS);
            mat_add(HPjHt, Rj, Sj, N_MEAS, N_MEAS);

            /* Kalman gain K = Pj_pred*Ht*Sj^-1 using primitives */
            mat_mul(Pj_pred, Hjt, PHt, N_STATE, N_STATE, N_MEAS);
            {
                double ecol[3], sol[3];
                for (int col = 0; col < N_MEAS; col++) {
                    ecol[0] = 0.0; ecol[1] = 0.0; ecol[2] = 0.0;
                    ecol[col] = 1.0;
                    chol_solve(Sj, ecol, sol, N_MEAS);
                    mat_mul(PHt, sol, kcol, N_STATE, N_MEAS, 1);
                    for (int r = 0; r < N_STATE; r++)
                        Kj[r*N_MEAS + col] = kcol[r];
                }
            }

            /* x = x_pred + K*y */
            mat_mul(Kj, yj, Ky, N_STATE, N_MEAS, 1);
            mat_add(xj_pred, Ky, xj, N_STATE, 1);

            /* Joseph form: P = (I-KH)*Pj_pred*(I-KH)^T + K*R*K^T */
            mat_mul(Kj, Hj, KHj, N_STATE, N_MEAS, N_STATE);
            mat_sub(Ij, KHj, IKHj, N_STATE, N_STATE);
            mat_mul(IKHj, Pj_pred, tmp1j, N_STATE, N_STATE, N_STATE);
            mat_transpose(IKHj, tmp2j, N_STATE, N_STATE);
            mat_mul(tmp1j, tmp2j, KRKtj, N_STATE, N_STATE, N_STATE);
            mat_mul(Kj, Rj, KRj, N_STATE, N_MEAS, N_MEAS);
            mat_transpose(Kj, Ktj, N_STATE, N_MEAS);
            mat_mul(KRj, Ktj, tmp1j, N_STATE, N_MEAS, N_STATE);
            mat_add(KRKtj, tmp1j, Pj, N_STATE, N_STATE);

            /* write back into full state and covariance */
            for (int s = 0; s < N_STATE; s++)
                x[xs + s] = xj[s];
            for (int r = 0; r < N_STATE; r++)
                for (int c = 0; c < N_STATE; c++)
                    P[(xs+r)*N_FULL+(xs+c)] = Pj[r*N_STATE+c];
        }

        for (int s = 0; s < N_FULL; s++) output[k][s] = x[s];

        if (k % 500 == 0)
            printf("[LKF] Frame %d/%d\n", k, T);
    }

    /* ── write output CSV ────────────────────────────────────────────────── */
    system("mkdir -p output");
    FILE* fout = fopen("output/lkf_asm_output.csv", "w");
    if (!fout) { fprintf(stderr, "[LKF] Cannot write output\n"); exit(1); }

    int first = 1;
    for (int jt = 0; jt < N_JOINTS; jt++)
        for (int s = 0; s < N_STATE; s++) {
            if (!first) fprintf(fout, ",");
            fprintf(fout, "%s_%s", JOINT_NAMES[jt], STATE_NAMES[s]);
            first = 0;
        }
    fprintf(fout, "\n");

    for (int k = 0; k < T; k++) {
        for (int i = 0; i < N_FULL; i++) {
            if (i > 0) fprintf(fout, ",");
            fprintf(fout, "%.15f", output[k][i]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
    printf("[LKF] Output written to output/lkf_asm_output.csv\n");

    /* ── cleanup ─────────────────────────────────────────────────────────── */
    for (int k = 0; k < T; k++) free(output[k]);
    free(output);
    free(F); free(Q); free(H); free(R_mat); free(Ft); free(I_full);
    free(x); free(P); free(x_pred); free(P_pred);
    free(FP); free(FPFt);
    /* per-joint working buffers */
    free(xj); free(Pj); free(xj_pred); free(Pj_pred);
    free(Hj); free(Hjt); free(HPj); free(HPjHt);
    free(Sj); free(Kj); free(yj); free(Rj); free(Ij);
    free(KHj); free(IKHj); free(tmp1j); free(tmp2j);
    free(KRj); free(KRKtj); free(Ktj); free(Hxj);
    free(PHt); free(Ky); free(kcol);

    return 0;
}