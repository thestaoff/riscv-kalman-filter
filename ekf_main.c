/*
 * ekf_main.c  —  Extended Kalman Filter, Milestone 3
 *
 * Design: ONE large state vector for ALL 23 joints simultaneously.
 *
 * State dimension: N_FULL = 23 * 12 = 276
 *   x ∈ R^276  =  [x_pelvis | x_L5 | ... | x_toeLeft]
 *   each x_j ∈ R^12 = [px vx ax jx | py vy ay jy | pz vz az jz]
 *
 * The EKF uses a nonlinear spherical measurement function h: R^276 → R^69
 * applied independently per-joint (h is block-separable). The Jacobian
 * J = dh/dx ∈ R^{69×276} is block-diagonal, computed per-timestep since
 * the spherical mapping is position-dependent.
 *
 * System matrices:
 *   F    ∈ R^{276×276}  — 23 copies of the 12×12 kinematic block (same as LKF)
 *   Q    ∈ R^{276×276}  — 23 copies of the 12×12 process noise block
 *   R    ∈ R^{69×69}    — 23 copies of the 3×3 spherical measurement noise
 *   J_k  ∈ R^{69×276}   — block-diagonal Jacobian, updated each timestep
 *
 * All matrix arithmetic and the arctan2 approximation are in ekf_asm.s.
 * This C file handles: CSV I/O, matrix construction, angle wrapping, output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── dimensions ─────────────────────────────────────────────────────────────── */
#define N_JOINTS   23
#define N_STATE    12
#define N_MEAS     3
#define N_FULL     (N_JOINTS * N_STATE)   /* 276 */
#define M_FULL     (N_JOINTS * N_MEAS)    /* 69  */
#define MAX_FRAMES 4000
#define DT         0.01
#define SIGMA_J2   1.0
#define R_RANGE    0.05        /* range measurement noise variance      */
#define R_ANGLE    0.005       /* azimuth / elevation noise variance    */
#define EPS        1e-9        /* singularity guard for spherical coords */

/* ─── assembly declarations ──────────────────────────────────────────────────── */
extern void mat_zero      (double* A, int r, int c);
extern void mat_eye       (double* A, int n);
extern void mat_mul       (const double* A, const double* B, double* C,
                           int r, int k, int c);
extern void mat_add       (const double* A, const double* B, double* C,
                           int r, int c);
extern void mat_sub       (const double* A, const double* B, double* C,
                           int r, int c);
extern void mat_transpose (const double* A, double* B, int r, int c);
extern void mat_scale     (const double* A, double s, double* B, int r, int c);
extern void chol_solve    (const double* A, const double* b, double* x, int n);
extern void compute_gain  (const double* P, const double* J,
                           const double* S, double* K);
extern void h_spherical   (const double* x, double* z_sph);
extern void compute_jacobian(const double* x_pred, double* J_single);
extern void ekf_predict   (const double* F,  const double* Ft,
                           const double* P,  const double* Q,
                           const double* x,
                           double* x_pred,   double* P_pred,
                           double* FP,       double* FPFt);
extern void ekf_update    (const double* K,  const double* J,
                           const double* P_pred, const double* R_mat,
                           const double* x_pred, const double* y,
                           double* x,        double* P,
                           double* KJ,       double* IKJ,
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
    if (!f) { fprintf(stderr, "[EKF] Cannot open %s\n", path); exit(1); }

    char line[8192];
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "[EKF] Empty file %s\n", path); exit(1);
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
    printf("[EKF] Loaded %d frames from %s\n", frame, path);
    return frame;
}

/* ─── angle wrapping to (−π, π] ─────────────────────────────────────────────── */
static double wrap_angle(double a)
{
    const double PI = 3.14159265358979323846;
    while (a >  PI) a -= 2.0 * PI;
    while (a < -PI) a += 2.0 * PI;
    return a;
}

/* ─── build F: 276×276 block-diagonal (identical to LKF) ────────────────────── */
static void build_F(double* F)
{
    mat_zero(F, N_FULL, N_FULL);
    double dt  = DT;
    double dt2 = dt * dt / 2.0;
    double dt3 = dt * dt * dt / 6.0;

    for (int jt = 0; jt < N_JOINTS; jt++) {
        int base = jt * N_STATE;
        for (int ax = 0; ax < 3; ax++) {
            int o = base + ax * 4;
            F[o*N_FULL + o]     = 1.0;
            F[o*N_FULL + o+1]   = dt;
            F[o*N_FULL + o+2]   = dt2;
            F[o*N_FULL + o+3]   = dt3;
            F[(o+1)*N_FULL+o+1] = 1.0;
            F[(o+1)*N_FULL+o+2] = dt;
            F[(o+1)*N_FULL+o+3] = dt2;
            F[(o+2)*N_FULL+o+2] = 1.0;
            F[(o+2)*N_FULL+o+3] = dt;
            F[(o+3)*N_FULL+o+3] = 1.0;
        }
    }
}

/* ─── build Q: 276×276 block-diagonal process noise (identical to LKF) ──────── */
static void build_Q(double* Q)
{
    mat_zero(Q, N_FULL, N_FULL);
    double dt  = DT;
    double dt2 = dt*dt,  dt3 = dt2*dt, dt4 = dt3*dt,
           dt5 = dt4*dt, dt6 = dt5*dt;
    double s = SIGMA_J2;

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

/* ─── build R: 69×69 block-diagonal spherical measurement noise ──────────────────
 *
 * For the EKF, measurements are in spherical coordinates (r, θ, φ).
 * Each joint's 3×3 noise block is diagonal with different variances:
 *   R_j = diag(R_RANGE, R_ANGLE, R_ANGLE)
 * Range noise is higher than angle noise because range depends on all three
 * Cartesian coordinates, accumulating more uncertainty.
 */
static void build_R(double* R)
{
    mat_zero(R, M_FULL, M_FULL);
    for (int jt = 0; jt < N_JOINTS; jt++) {
        int o = jt * N_MEAS;
        R[(o+0)*M_FULL + (o+0)] = R_RANGE;   /* range variance      */
        R[(o+1)*M_FULL + (o+1)] = R_ANGLE;   /* azimuth variance    */
        R[(o+2)*M_FULL + (o+2)] = R_ANGLE;   /* elevation variance  */
    }
}

/* ─── build full Jacobian J: 69×276 block-diagonal ──────────────────────────────
 *
 * The assembly compute_jacobian computes a single 3×12 Jacobian for one joint.
 * Here we call it 23 times and pack the results into the full 69×276 J matrix.
 *
 * J is block-diagonal: only the 3×12 sub-block at rows [jt*3, jt*3+3) and
 * columns [jt*12, jt*12+12) is nonzero for joint jt. All other entries are 0
 * because h for joint jt depends only on that joint's state.
 */
static void build_jacobian_full(const double* x_pred, double* J_full)
{
    mat_zero(J_full, M_FULL, N_FULL);

    double J_single[N_MEAS * N_STATE];   /* 3×12 per-joint Jacobian */

    for (int jt = 0; jt < N_JOINTS; jt++) {
        /* pointer to this joint's 12-dim predicted state */
        const double* xj = x_pred + jt * N_STATE;

        /* compute 3×12 Jacobian for joint jt in assembly */
        compute_jacobian(xj, J_single);

        /* copy into the correct block of the 69×276 full Jacobian */
        int row0 = jt * N_MEAS;    /* first row of this joint's block */
        int col0 = jt * N_STATE;   /* first col of this joint's block */
        for (int r = 0; r < N_MEAS; r++)
            for (int c = 0; c < N_STATE; c++)
                J_full[(row0 + r) * N_FULL + (col0 + c)] = J_single[r * N_STATE + c];
    }
}

/* ─── convert full 276-state predicted x to 69-dim spherical measurement ─────────
 *
 * The assembly h_spherical converts a single joint's 12-dim state to [r, θ, φ].
 * We call it 23 times and pack results into the 69-dim z_pred vector.
 */
static void h_full(const double* x_pred, double* z_pred)
{
    double z_single[N_MEAS];
    for (int jt = 0; jt < N_JOINTS; jt++) {
        h_spherical(x_pred + jt * N_STATE, z_single);
        z_pred[jt*3 + 0] = z_single[0];   /* range     */
        z_pred[jt*3 + 1] = z_single[1];   /* azimuth   */
        z_pred[jt*3 + 2] = z_single[2];   /* elevation */
    }
}

/* ─── convert noisy Cartesian measurements to 69-dim spherical ───────────────── */
static void cart_to_spherical_full(const double* noisy_row, double* z_meas)
{
    double tmp_state[N_STATE];
    double z_single[N_MEAS];
    for (int jt = 0; jt < N_JOINTS; jt++) {
        memset(tmp_state, 0, sizeof(tmp_state));
        tmp_state[0] = noisy_row[jt*3 + 0];   /* px into position slot */
        tmp_state[4] = noisy_row[jt*3 + 1];   /* py */
        tmp_state[8] = noisy_row[jt*3 + 2];   /* pz */
        h_spherical(tmp_state, z_single);
        z_meas[jt*3 + 0] = z_single[0];
        z_meas[jt*3 + 1] = z_single[1];
        z_meas[jt*3 + 2] = z_single[2];
    }
}

/* ─── main ───────────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("[EKF] Starting — single 276-dim state vector for all 23 joints\n");

    /* ── load noisy data ──────────────────────────────────────────────────── */
    static double noisy[MAX_FRAMES][N_JOINTS * 3];
    int T = load_csv("data/noisy.csv", noisy);

    /* ── output storage ───────────────────────────────────────────────────── */
    double** output = (double**)malloc(T * sizeof(double*));
    for (int k = 0; k < T; k++)
        output[k] = (double*)calloc(N_FULL, sizeof(double));

    /* ── system matrices (heap-allocated: 276×276 = 609 kB each) ─────────── */
    double* F      = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* Q      = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* R_mat  = (double*)malloc(M_FULL * M_FULL * sizeof(double));
    double* Ft     = (double*)malloc(N_FULL * N_FULL * sizeof(double));
    double* I_full = (double*)malloc(N_FULL * N_FULL * sizeof(double));

    build_F(F);
    build_Q(Q);
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

    /* ── initialise state x₀ ─────────────────────────────────────────────── */
    for (int jt = 0; jt < N_JOINTS; jt++) {
        x[jt*N_STATE + 0] = noisy[0][jt*3 + 0];
        x[jt*N_STATE + 4] = noisy[0][jt*3 + 1];
        x[jt*N_STATE + 8] = noisy[0][jt*3 + 2];
    }
    mat_eye(P, N_FULL);
    mat_scale(P, 100.0, P, N_FULL, N_FULL);
    for (int s = 0; s < N_FULL; s++) output[0][s] = x[s];

    /* Per-joint 12×12 working matrices for the update step.
     * The assembly ekf_update/compute_gain are designed for 12-dim state
     * and 3-dim measurement (3×3 S, 12×3 K). We keep one 276-dim state
     * vector x and P, but apply gains joint-by-joint — equivalent to the
     * monolithic filter because P and F are block-diagonal (joints decouple).
     */
    double* xj      = (double*)malloc(N_STATE * sizeof(double));
    double* Pj      = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* xj_pred = (double*)malloc(N_STATE * sizeof(double));
    double* Pj_pred = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* FPj     = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* FPjFt   = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* Jj      = (double*)malloc(N_MEAS  * N_STATE * sizeof(double));
    double* Sj      = (double*)malloc(N_MEAS  * N_MEAS  * sizeof(double));
    double* Kj      = (double*)malloc(N_STATE * N_MEAS  * sizeof(double));
    double* yj      = (double*)malloc(N_MEAS  * sizeof(double));
    double* Rj      = (double*)malloc(N_MEAS  * N_MEAS  * sizeof(double));
    double* Ij      = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* KJj     = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* IKJj    = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* tmp1j   = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* tmp2j   = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* KRj     = (double*)malloc(N_STATE * N_MEAS  * sizeof(double));
    double* KRKtj   = (double*)malloc(N_STATE * N_STATE * sizeof(double));
    double* Ktj     = (double*)malloc(N_MEAS  * N_STATE * sizeof(double));
    double* Jjt     = (double*)malloc(N_STATE * N_MEAS  * sizeof(double));
    double* JPj     = (double*)malloc(N_MEAS  * N_STATE * sizeof(double));
    double* JPjJt   = (double*)malloc(N_MEAS  * N_MEAS  * sizeof(double));
    double* z_predj = (double*)malloc(N_MEAS  * sizeof(double));

    /* Per-joint R is the 3×3 diagonal noise block */
    mat_zero(Rj, N_MEAS, N_MEAS);
    Rj[0*N_MEAS+0] = R_RANGE;
    Rj[1*N_MEAS+1] = R_ANGLE;
    Rj[2*N_MEAS+2] = R_ANGLE;
    mat_eye(Ij, N_STATE);

    double* PJt   = (double*)malloc(N_STATE * N_MEAS * sizeof(double));  /* 12×3 */
    double* Ky    = (double*)malloc(N_STATE * sizeof(double));
    double* kcol  = (double*)malloc(N_STATE * sizeof(double));

    /* ══════════════════════════════════════════════════════════════════════════
     * MAIN FILTER LOOP
     *
     * Single 276-dim state vector x and covariance P.
     * PREDICT: one assembly call propagates all 276 states.
     * UPDATE:  applied joint-by-joint (each joint's 12×12 P block updated
     *          independently — valid because F, Q, H, R are block-diagonal
     *          so joints are statistically decoupled).
     * ══════════════════════════════════════════════════════════════════════════
     */
    for (int k = 1; k < T; k++) {

        /* ── PREDICT: propagate all 276 states in one assembly call ──────── */
        ekf_predict(F, Ft, P, Q, x,
                    x_pred, P_pred, FP, FPFt);

        /* ── PER-JOINT UPDATE ────────────────────────────────────────────── */
        for (int jt = 0; jt < N_JOINTS; jt++) {
            int xs = jt * N_STATE;   /* state offset for this joint */

            /* extract this joint's 12-dim predicted state */
            for (int s = 0; s < N_STATE; s++)
                xj_pred[s] = x_pred[xs + s];

            /* extract this joint's 12×12 predicted covariance block */
            for (int r = 0; r < N_STATE; r++)
                for (int c = 0; c < N_STATE; c++)
                    Pj_pred[r*N_STATE+c] = P_pred[(xs+r)*N_FULL+(xs+c)];

            /* Jacobian dh/dx for this joint (assembly, 3×12) */
            compute_jacobian(xj_pred, Jj);

            /* h(x_pred) for this joint (assembly) */
            h_spherical(xj_pred, z_predj);

            /* noisy measurement → spherical for this joint */
            double tmp_state[N_STATE];
            memset(tmp_state, 0, sizeof(tmp_state));
            tmp_state[0] = noisy[k][jt*3+0];
            tmp_state[4] = noisy[k][jt*3+1];
            tmp_state[8] = noisy[k][jt*3+2];
            double z_measj[N_MEAS];
            h_spherical(tmp_state, z_measj);

            /* innovation with angle wrapping */
            yj[0] = z_measj[0] - z_predj[0];
            yj[1] = wrap_angle(z_measj[1] - z_predj[1]);
            yj[2] = wrap_angle(z_measj[2] - z_predj[2]);

            /* S = J*Pj_pred*Jt + Rj  (3×3) */
            mat_transpose(Jj, Jjt, N_MEAS, N_STATE);
            mat_mul(Jj, Pj_pred, JPj,   N_MEAS, N_STATE, N_STATE);
            mat_mul(JPj, Jjt,    JPjJt, N_MEAS, N_STATE, N_MEAS);
            mat_add(JPjJt, Rj, Sj, N_MEAS, N_MEAS);

            /* Kalman gain K = Pj_pred*Jt*Sj^-1 using primitives */
            mat_mul(Pj_pred, Jjt, PJt, N_STATE, N_STATE, N_MEAS);
            {
                double ecol[3], sol[3];
                for (int col = 0; col < N_MEAS; col++) {
                    ecol[0] = 0.0; ecol[1] = 0.0; ecol[2] = 0.0;
                    ecol[col] = 1.0;
                    chol_solve(Sj, ecol, sol, N_MEAS);
                    mat_mul(PJt, sol, kcol, N_STATE, N_MEAS, 1);
                    for (int r = 0; r < N_STATE; r++)
                        Kj[r*N_MEAS + col] = kcol[r];
                }
            }

            /* x = x_pred + K*y */
            mat_mul(Kj, yj, Ky, N_STATE, N_MEAS, 1);
            mat_add(xj_pred, Ky, xj, N_STATE, 1);

            /* Joseph form: P = (I-KJ)*Pj_pred*(I-KJ)^T + K*R*K^T */
            /* KJj = K*J  (12×3 * 3×12 = 12×12) */
            mat_mul(Kj, Jj, KJj, N_STATE, N_MEAS, N_STATE);
            /* IKJj = I - KJ */
            mat_sub(Ij, KJj, IKJj, N_STATE, N_STATE);
            /* tmp1j = IKJ * Pj_pred */
            mat_mul(IKJj, Pj_pred, tmp1j, N_STATE, N_STATE, N_STATE);
            /* IKJt = IKJ^T stored in tmp2j */
            mat_transpose(IKJj, tmp2j, N_STATE, N_STATE);
            /* KRKtj = IKJ*Pj_pred*IKJt (store in KRKtj temporarily) */
            mat_mul(tmp1j, tmp2j, KRKtj, N_STATE, N_STATE, N_STATE);
            /* KRj = K*R  (12×3 * 3×3 = 12×3) */
            mat_mul(Kj, Rj, KRj, N_STATE, N_MEAS, N_MEAS);
            /* Ktj = K^T  (3×12) */
            mat_transpose(Kj, Ktj, N_STATE, N_MEAS);
            /* tmp1j = KR*Kt  (12×3 * 3×12 = 12×12) */
            mat_mul(KRj, Ktj, tmp1j, N_STATE, N_MEAS, N_STATE);
            /* Pj = KRKtj + tmp1j */
            mat_add(KRKtj, tmp1j, Pj, N_STATE, N_STATE);

            /* write updated state back into full 276-dim x */
            for (int s = 0; s < N_STATE; s++)
                x[xs + s] = xj[s];

            /* write updated covariance block back into full 276×276 P */
            for (int r = 0; r < N_STATE; r++)
                for (int c = 0; c < N_STATE; c++)
                    P[(xs+r)*N_FULL+(xs+c)] = Pj[r*N_STATE+c];
        }

        for (int s = 0; s < N_FULL; s++) output[k][s] = x[s];

        if (k % 500 == 0)
            printf("[EKF] Frame %d/%d\n", k, T);
    }

    /* ── write output CSV ────────────────────────────────────────────────── */
    system("mkdir -p output");
    FILE* fout = fopen("output/ekf_asm_output.csv", "w");
    if (!fout) { fprintf(stderr, "[EKF] Cannot write output\n"); exit(1); }

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
    printf("[EKF] Output written to output/ekf_asm_output.csv\n");

    /* ── cleanup ─────────────────────────────────────────────────────────── */
    for (int k = 0; k < T; k++) free(output[k]);
    free(output);
    /* Note: remaining heap buffers freed by OS on exit */
    printf("[EKF] Done.\n");
    return 0;
}