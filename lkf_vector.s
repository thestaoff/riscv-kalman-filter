# =============================================================================
# lkf_vector_fixed.s — Vectorised Linear Kalman Filter (RVV)
# Milestone 4 — full-body 276-state implementation
#
# IMPORTANT DESIGN FIX:
#   This file uses ONE global full-body state vector, not 23 independent
#   per-joint filters.
#
#   N = 23 joints × 12 states = 276
#   M = 23 joints ×  3 measurements = 69
#
#   x      : 276×1
#   F,Q,P  : 276×276
#   H      : 69×276
#   R,S    : 69×69
#   K      : 276×69
#
# RVV kernels:
#   mat_zero, mat_eye, mat_add, mat_sub, mat_scale
#   mat_mul          — vectorised dot product using vle64.v / vlse64.v
#   mat_transpose    — vectorised row load + strided column store using vsse64.v
#
# Small scalar helper:
#   chol_solve       — 3×3 Cholesky solve, used per 3×3 block of S
#
# compute_gain:
#   Builds the global K(276×69) block-diagonally from the full matrices.
#   This is still a single global 276-state filter: the block-diagonal structure
#   is only exploited to solve each independent 3×3 innovation block safely.
# =============================================================================

    .section .text

# =============================================================================
# mat_zero(double* A, int r, int c)
# A = 0, total elements = r*c
# =============================================================================
    .global mat_zero
mat_zero:
    mul     a3, a1, a2
    fmv.d.x ft0, zero
.Lzero_loop:
    beqz    a3, .Lzero_done
    vsetvli t0, a3, e64, m8, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (a0)
    slli    t1, t0, 3
    add     a0, a0, t1
    sub     a3, a3, t0
    j       .Lzero_loop
.Lzero_done:
    ret

# =============================================================================
# mat_eye(double* A, int n)
# A = I_n
# =============================================================================
    .global mat_eye
mat_eye:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0, 16(sp)
    sd      s1,  8(sp)

    mv      s0, a0
    mv      s1, a1

    mv      a2, a1              # mat_zero(A, n, n)
    call    mat_zero

    li      t0, 1
    fcvt.d.w ft1, t0            # 1.0
    mv      a0, s0
    mv      t2, s1
    addi    t1, s1, 1
    slli    t1, t1, 3           # diagonal stride = (n+1)*8
.Leye_loop:
    beqz    t2, .Leye_done
    fsd     ft1, 0(a0)
    add     a0, a0, t1
    addi    t2, t2, -1
    j       .Leye_loop
.Leye_done:
    ld      s1,  8(sp)
    ld      s0, 16(sp)
    ld      ra, 24(sp)
    addi    sp, sp, 32
    ret

# =============================================================================
# mat_add(double* A, double* B, double* C, int r, int c)
# C = A + B
# =============================================================================
    .global mat_add
mat_add:
    mul     a3, a3, a4
.Ladd_loop:
    beqz    a3, .Ladd_done
    vsetvli t0, a3, e64, m8, ta, ma
    vle64.v v0,  (a0)
    vle64.v v8,  (a1)
    vfadd.vv v16, v0, v8
    vse64.v v16, (a2)
    slli    t1, t0, 3
    add     a0, a0, t1
    add     a1, a1, t1
    add     a2, a2, t1
    sub     a3, a3, t0
    j       .Ladd_loop
.Ladd_done:
    ret

# =============================================================================
# mat_sub(double* A, double* B, double* C, int r, int c)
# C = A - B
# =============================================================================
    .global mat_sub
mat_sub:
    mul     a3, a3, a4
.Lsub_loop:
    beqz    a3, .Lsub_done
    vsetvli t0, a3, e64, m8, ta, ma
    vle64.v v0,  (a0)
    vle64.v v8,  (a1)
    vfsub.vv v16, v0, v8
    vse64.v v16, (a2)
    slli    t1, t0, 3
    add     a0, a0, t1
    add     a1, a1, t1
    add     a2, a2, t1
    sub     a3, a3, t0
    j       .Lsub_loop
.Lsub_done:
    ret

# =============================================================================
# mat_scale(double* A, double* B, double alpha, int r, int c)
# NOTE: alpha is in fa0. Signature used by assembly side:
#   a0=A, a1=B, a2=r, a3=c, fa0=alpha
# B = alpha*A
# =============================================================================
    .global mat_scale
mat_scale:
    mul     a2, a2, a3
.Lscale_loop:
    beqz    a2, .Lscale_done
    vsetvli t0, a2, e64, m8, ta, ma
    vle64.v v0, (a0)
    vfmul.vf v8, v0, fa0
    vse64.v v8, (a1)
    slli    t1, t0, 3
    add     a0, a0, t1
    add     a1, a1, t1
    sub     a2, a2, t0
    j       .Lscale_loop
.Lscale_done:
    ret

# =============================================================================
# mat_mul(double* A, double* B, double* C, int r, int k, int c)
# C = A(r×k) * B(k×c), row-major.
#
# Inner dot product is vectorised over k:
#   A row: unit stride via vle64.v
#   B col: strided via vlse64.v, stride = c*8
#   Reduction: vfredosum.vs for reproducible ordered reduction
# =============================================================================
    .global mat_mul
mat_mul:
    addi    sp, sp, -80
    sd      ra, 72(sp)
    sd      s0, 64(sp)
    sd      s1, 56(sp)
    sd      s2, 48(sp)
    sd      s3, 40(sp)
    sd      s4, 32(sp)
    sd      s5, 24(sp)
    sd      s6, 16(sp)
    sd      s7,  8(sp)
    sd      s8,  0(sp)

    mv      s0, a0              # A
    mv      s1, a1              # B
    mv      s2, a2              # C
    mv      s3, a3              # r
    mv      s4, a4              # k
    mv      s5, a5              # c

    fmv.d.x ft0, zero           # 0.0 for reduction init
    slli    t3, s5, 3           # B column stride = c*8

    li      s6, 0               # i
.Lmul_i:
    bge     s6, s3, .Lmul_done

    # s8 = &A[i][0]
    mul     t0, s6, s4
    slli    t0, t0, 3
    add     s8, s0, t0

    li      s7, 0               # j
.Lmul_j:
    bge     s7, s5, .Lmul_next_i

    # t2 = &B[0][j]
    slli    t1, s7, 3
    add     t2, s1, t1

    mv      t4, s4              # remaining k
    mv      t5, s8              # A pointer
    mv      t6, t2              # B column pointer
    fmv.d.x fa0, zero           # scalar accumulator

.Lmul_dot:
    beqz    t4, .Lmul_store
    vsetvli a6, t4, e64, m4, ta, ma

    vle64.v  v0, (t5)
    vlse64.v v4, (t6), t3
    vfmul.vv v8, v0, v4

    vfmv.s.f v12, ft0
    vfredosum.vs v12, v8, v12
    vfmv.f.s fa1, v12
    fadd.d  fa0, fa0, fa1

    slli    a7, a6, 3
    add     t5, t5, a7
    mul     a7, a6, t3
    add     t6, t6, a7
    sub     t4, t4, a6
    j       .Lmul_dot

.Lmul_store:
    mul     t0, s6, s5
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t0, s2, t0
    fsd     fa0, 0(t0)

    addi    s7, s7, 1
    j       .Lmul_j

.Lmul_next_i:
    addi    s6, s6, 1
    j       .Lmul_i

.Lmul_done:
    ld      s8,  0(sp)
    ld      s7,  8(sp)
    ld      s6, 16(sp)
    ld      s5, 24(sp)
    ld      s4, 32(sp)
    ld      s3, 40(sp)
    ld      s2, 48(sp)
    ld      s1, 56(sp)
    ld      s0, 64(sp)
    ld      ra, 72(sp)
    addi    sp, sp, 80
    ret

# =============================================================================
# mat_transpose(double* A, double* B, int r, int c)
# B = A^T, A is r×c, B is c×r.
#
# FIXED FOR M4:
#   The old file kept transpose scalar. This version uses RVV:
#   each row of A is loaded with vle64.v and written into a column of B
#   with vsse64.v using stride r*8.
# =============================================================================
    .global mat_transpose
mat_transpose:
    slli    t6, a2, 3           # destination stride = r*8 bytes
    li      t4, 0               # i = 0
.Ltr_i:
    bge     t4, a2, .Ltr_done

    # t0 = &A[i][0]
    mul     t0, t4, a3
    slli    t0, t0, 3
    add     t0, a0, t0

    # t1 = &B[0][i]
    slli    t1, t4, 3
    add     t1, a1, t1

    mv      t2, a3              # remaining elements in row = c
.Ltr_row:
    beqz    t2, .Ltr_next_i
    vsetvli t3, t2, e64, m1, ta, ma
    vle64.v  v0, (t0)
    vsse64.v v0, (t1), t6

    slli    t5, t3, 3
    add     t0, t0, t5
    mul     t5, t3, t6
    add     t1, t1, t5
    sub     t2, t2, t3
    j       .Ltr_row

.Ltr_next_i:
    addi    t4, t4, 1
    j       .Ltr_i
.Ltr_done:
    ret

# =============================================================================
# chol_solve(double* A, double* b, double* x, int n)
# Solves a 3×3 SPD system A*x=b with Cholesky.
# A must be a contiguous 3×3 block. n is expected to be 3.
# =============================================================================
    .global chol_solve
chol_solve:
    addi    sp, sp, -128
    sd      ra, 120(sp)
    sd      s0, 112(sp)
    sd      s1, 104(sp)
    sd      s2,  96(sp)
    sd      s3,  88(sp)
    sd      s4,  80(sp)

    mv      s0, a0              # A
    mv      s1, a1              # b
    mv      s2, a2              # x
    mv      s3, a3              # n

    addi    sp, sp, -72
    mv      s4, sp              # L[3][3]

    # L00
    fld     ft0,  0(s0)
    fsqrt.d ft0, ft0
    fsd     ft0,  0(s4)

    # L10
    fld     ft1, 24(s0)
    fdiv.d  ft1, ft1, ft0
    fsd     ft1, 24(s4)

    # L11
    fld     ft2, 32(s0)
    fmul.d  ft3, ft1, ft1
    fsub.d  ft2, ft2, ft3
    fsqrt.d ft2, ft2
    fsd     ft2, 32(s4)

    # L20
    fld     ft4, 48(s0)
    fdiv.d  ft4, ft4, ft0
    fsd     ft4, 48(s4)

    # L21
    fld     ft5, 56(s0)
    fmul.d  ft6, ft4, ft1
    fsub.d  ft5, ft5, ft6
    fdiv.d  ft5, ft5, ft2
    fsd     ft5, 56(s4)

    # L22
    fld     ft6, 64(s0)
    fmul.d  ft7, ft4, ft4
    fsub.d  ft6, ft6, ft7
    fmul.d  ft7, ft5, ft5
    fsub.d  ft6, ft6, ft7
    fsqrt.d ft6, ft6
    fsd     ft6, 64(s4)

    # Forward substitution: y stored temporarily in x
    fld     ft0,  0(s1)
    fld     ft1,  0(s4)
    fdiv.d  ft0, ft0, ft1
    fsd     ft0,  0(s2)

    fld     ft1,  8(s1)
    fld     ft2, 24(s4)
    fld     ft3,  0(s2)
    fmul.d  ft4, ft2, ft3
    fsub.d  ft1, ft1, ft4
    fld     ft2, 32(s4)
    fdiv.d  ft1, ft1, ft2
    fsd     ft1,  8(s2)

    fld     ft1, 16(s1)
    fld     ft2, 48(s4)
    fld     ft3,  0(s2)
    fmul.d  ft4, ft2, ft3
    fsub.d  ft1, ft1, ft4
    fld     ft2, 56(s4)
    fld     ft3,  8(s2)
    fmul.d  ft4, ft2, ft3
    fsub.d  ft1, ft1, ft4
    fld     ft2, 64(s4)
    fdiv.d  ft1, ft1, ft2
    fsd     ft1, 16(s2)

    # Backward substitution
    fld     ft0, 16(s2)
    fld     ft1, 64(s4)
    fdiv.d  ft0, ft0, ft1
    fsd     ft0, 16(s2)

    fld     ft1,  8(s2)
    fld     ft2, 56(s4)
    fmul.d  ft3, ft2, ft0
    fsub.d  ft1, ft1, ft3
    fld     ft2, 32(s4)
    fdiv.d  ft1, ft1, ft2
    fsd     ft1,  8(s2)

    fld     ft0,  0(s2)
    fld     ft2, 24(s4)
    fld     ft3,  8(s2)
    fmul.d  ft4, ft2, ft3
    fsub.d  ft0, ft0, ft4
    fld     ft2, 48(s4)
    fld     ft3, 16(s2)
    fmul.d  ft4, ft2, ft3
    fsub.d  ft0, ft0, ft4
    fld     ft2,  0(s4)
    fdiv.d  ft0, ft0, ft2
    fsd     ft0,  0(s2)

    addi    sp, sp, 72
    ld      s4,  80(sp)
    ld      s3,  88(sp)
    ld      s2,  96(sp)
    ld      s1, 104(sp)
    ld      s0, 112(sp)
    ld      ra, 120(sp)
    addi    sp, sp, 128
    ret

# =============================================================================
# compute_gain(double* P, double* H, double* S, double* K)
# Global LKF gain for the full-body system:
#   P: 276×276, H: 69×276, S: 69×69, K: 276×69
#
# Because F,Q,H,R are block diagonal and Joseph update preserves block structure,
# S is also block diagonal with 23 independent 3×3 blocks. This routine:
#   1. zeroes the full K(276×69)
#   2. for each joint block j:
#        PHt_j = P_j * H_j^T       (12×12 * 12×3 = 12×3)
#        K_j   = PHt_j * S_j^-1    (12×3)
#      then writes K_j into the correct block of global K.
# =============================================================================
    .global compute_gain
compute_gain:
    addi    sp, sp, -640
    sd      ra, 632(sp)
    sd      s0, 624(sp)
    sd      s1, 616(sp)
    sd      s2, 608(sp)
    sd      s3, 600(sp)
    sd      s4, 592(sp)
    sd      s5, 584(sp)
    sd      s6, 576(sp)
    sd      s7, 568(sp)
    sd      s8, 560(sp)
    sd      s9, 552(sp)
    sd      s10,544(sp)
    sd      s11,536(sp)

    mv      s0, a0              # P full 276×276
    mv      s1, a1              # H full 69×276
    mv      s2, a2              # S full 69×69
    mv      s3, a3              # K full 276×69

    # local buffers
    #   PHt  : sp+0   .. sp+287  (12×3)
    #   S3   : sp+288 .. sp+359  (3×3)
    #   e    : sp+360 .. sp+383  (3)
    #   sol  : sp+384 .. sp+407  (3)

    # K = 0 globally first, so off-diagonal gain blocks are explicitly zero.
    mv      a0, s3
    li      a1, 276
    li      a2, 69
    call    mat_zero

    li      s4, 0               # joint index jt = 0..22
.Lgain_joint:
    li      t0, 23
    bge     s4, t0, .Lgain_done

    # s5 = state row/col offset = jt*12
    # s6 = measurement row/col offset = jt*3
    li      t0, 12
    mul     s5, s4, t0
    li      t0, 3
    mul     s6, s4, t0

    # -------------------------------------------------------------------------
    # Copy S_j from full S(69×69) into contiguous local S3(3×3)
    # -------------------------------------------------------------------------
    li      t0, 0               # r = 0
.LcopyS_r:
    li      t1, 3
    bge     t0, t1, .LcopyS_done
    li      t2, 0               # c = 0
.LcopyS_c:
    li      t1, 3
    bge     t2, t1, .LcopyS_next_r

    add     t3, s6, t0          # full row = m0 + r
    li      t4, 69
    mul     t3, t3, t4
    add     t4, s6, t2          # full col = m0 + c
    add     t3, t3, t4
    slli    t3, t3, 3
    add     t3, s2, t3
    fld     ft0, 0(t3)

    mul     t5, t0, t1          # local r*3 + c
    add     t5, t5, t2
    slli    t5, t5, 3
    addi    t6, sp, 288
    add     t6, t6, t5
    fsd     ft0, 0(t6)

    addi    t2, t2, 1
    j       .LcopyS_c
.LcopyS_next_r:
    addi    t0, t0, 1
    j       .LcopyS_r
.LcopyS_done:

    # -------------------------------------------------------------------------
    # Build PHt local block: PHt[i][m] = dot(P[state0+i, state0:state0+11],
    #                                      H[meas0+m, state0:state0+11])
    # -------------------------------------------------------------------------
    li      s7, 0               # i = 0..11
.Lpht_i:
    li      t0, 12
    bge     s7, t0, .Lpht_done
    li      s8, 0               # m = 0..2
.Lpht_m:
    li      t0, 3
    bge     s8, t0, .Lpht_next_i

    # t5 = P row segment pointer
    add     t0, s5, s7          # full state row = state0 + i
    li      t1, 276
    mul     t0, t0, t1
    add     t0, t0, s5          # col = state0
    slli    t0, t0, 3
    add     t5, s0, t0

    # t6 = H row segment pointer
    add     t0, s6, s8          # full meas row = meas0 + m
    li      t1, 276
    mul     t0, t0, t1
    add     t0, t0, s5          # col = state0
    slli    t0, t0, 3
    add     t6, s1, t0

    li      t2, 12              # remaining length
    fmv.d.x fa0, zero
.Lpht_dot:
    beqz    t2, .Lpht_store
    vsetvli t3, t2, e64, m4, ta, ma
    vle64.v v0, (t5)
    vle64.v v4, (t6)
    vfmul.vv v8, v0, v4
    fmv.d.x ft0, zero
    vfmv.s.f v12, ft0
    vfredosum.vs v12, v8, v12
    vfmv.f.s fa1, v12
    fadd.d fa0, fa0, fa1

    slli    t4, t3, 3
    add     t5, t5, t4
    add     t6, t6, t4
    sub     t2, t2, t3
    j       .Lpht_dot

.Lpht_store:
    # offset = (i*3 + m)*8
    li      t0, 3
    mul     t1, s7, t0
    add     t1, t1, s8
    slli    t1, t1, 3
    add     t1, sp, t1          # PHt base = sp+0
    fsd     fa0, 0(t1)

    addi    s8, s8, 1
    j       .Lpht_m
.Lpht_next_i:
    addi    s7, s7, 1
    j       .Lpht_i
.Lpht_done:

    # -------------------------------------------------------------------------
    # For each column col=0..2, solve S3 * sol = e_col, then write K block.
    # -------------------------------------------------------------------------
    li      s7, 0               # col = 0..2
.Lk_col:
    li      t0, 3
    bge     s7, t0, .Lgain_next_joint

    # e = [0,0,0]
    fmv.d.x ft0, zero
    fsd     ft0, 360(sp)
    fsd     ft0, 368(sp)
    fsd     ft0, 376(sp)
    li      t0, 1
    fcvt.d.w ft1, t0
    slli    t1, s7, 3
    addi    t2, sp, 360
    add     t2, t2, t1
    fsd     ft1, 0(t2)

    addi    a0, sp, 288         # S3
    addi    a1, sp, 360         # e
    addi    a2, sp, 384         # sol
    li      a3, 3
    call    chol_solve

    # Load sol[0..2]
    fld     ft4, 384(sp)
    fld     ft5, 392(sp)
    fld     ft6, 400(sp)

    # Store K[state0:state0+11, meas0+col] in chunks with stride 69*8.
    li      s8, 0               # row offset inside 12-state block
    li      t5, 552             # K row stride = 69*8
.Lk_store_chunk:
    li      t0, 12
    bge     s8, t0, .Lk_col_done
    sub     t1, t0, s8          # remaining rows
    vsetvli t2, t1, e64, m4, ta, ma

    # PHt col 0/1/2 starting at row s8, stride 3*8=24
    li      t3, 3
    mul     t4, s8, t3
    slli    t4, t4, 3           # s8*3*8
    add     t6, sp, t4          # &PHt[s8][0]
    li      t3, 24
    vlse64.v v0, (t6), t3
    addi    t6, t6, 8
    vlse64.v v4, (t6), t3
    addi    t6, t6, 8
    vlse64.v v8, (t6), t3

    vfmul.vf  v12, v0, ft4
    vfmacc.vf v12, ft5, v4
    vfmacc.vf v12, ft6, v8

    # destination = K[(state0+s8)][meas0+col]
    add     t0, s5, s8
    li      t1, 69
    mul     t0, t0, t1
    add     t1, s6, s7
    add     t0, t0, t1
    slli    t0, t0, 3
    add     t0, s3, t0
    vsse64.v v12, (t0), t5

    add     s8, s8, t2
    j       .Lk_store_chunk

.Lk_col_done:
    addi    s7, s7, 1
    j       .Lk_col

.Lgain_next_joint:
    addi    s4, s4, 1
    j       .Lgain_joint

.Lgain_done:
    ld      s11,536(sp)
    ld      s10,544(sp)
    ld      s9, 552(sp)
    ld      s8, 560(sp)
    ld      s7, 568(sp)
    ld      s6, 576(sp)
    ld      s5, 584(sp)
    ld      s4, 592(sp)
    ld      s3, 600(sp)
    ld      s2, 608(sp)
    ld      s1, 616(sp)
    ld      s0, 624(sp)
    ld      ra, 632(sp)
    addi    sp, sp, 640
    ret

# =============================================================================
# lkf_predict(F, Ft, P, Q, x, x_pred, P_pred, FP, FPFt)
# Full global predict:
#   x_pred = F*x                276×276 * 276×1
#   FP     = F*P                276×276 * 276×276
#   FPFt   = FP*Ft              276×276 * 276×276
#   P_pred = FPFt + Q           276×276
# =============================================================================
    .global lkf_predict
lkf_predict:
    addi    sp, sp, -96
    sd      ra, 88(sp)
    sd      s0, 80(sp)
    sd      s1, 72(sp)
    sd      s2, 64(sp)
    sd      s3, 56(sp)
    sd      s4, 48(sp)
    sd      s5, 40(sp)
    sd      s6, 32(sp)
    sd      s7, 24(sp)
    sd      s8, 16(sp)

    mv      s0, a0              # F
    mv      s1, a1              # Ft
    mv      s2, a2              # P
    mv      s3, a3              # Q
    mv      s4, a4              # x
    mv      s5, a5              # x_pred
    mv      s6, a6              # P_pred
    mv      s7, a7              # FP
    ld      s8, 96(sp)          # FPFt stack arg

    mv      a0, s0
    mv      a1, s4
    mv      a2, s5
    li      a3, 276
    li      a4, 276
    li      a5, 1
    call    mat_mul

    mv      a0, s0
    mv      a1, s2
    mv      a2, s7
    li      a3, 276
    li      a4, 276
    li      a5, 276
    call    mat_mul

    mv      a0, s7
    mv      a1, s1
    mv      a2, s8
    li      a3, 276
    li      a4, 276
    li      a5, 276
    call    mat_mul

    mv      a0, s8
    mv      a1, s3
    mv      a2, s6
    li      a3, 276
    li      a4, 276
    call    mat_add

    ld      s8, 16(sp)
    ld      s7, 24(sp)
    ld      s6, 32(sp)
    ld      s5, 40(sp)
    ld      s4, 48(sp)
    ld      s3, 56(sp)
    ld      s2, 64(sp)
    ld      s1, 72(sp)
    ld      s0, 80(sp)
    ld      ra, 88(sp)
    addi    sp, sp, 96
    ret

# =============================================================================
# lkf_update(K, H, P_pred, R, x_pred, y, x, P,
#            KH, IKH, tmp1, tmp2, KR, KRKt, Kt, I276)
# Full global Joseph update:
#   x   = x_pred + K*y                       276×1
#   KH  = K*H                                276×276
#   IKH = I276 - KH                          276×276
#   P   = IKH*P_pred*IKH^T + K*R*K^T         276×276
# =============================================================================
    .global lkf_update
lkf_update:
    addi    sp, sp, -176
    sd      ra, 168(sp)
    sd      s0, 160(sp)
    sd      s1, 152(sp)
    sd      s2, 144(sp)
    sd      s3, 136(sp)
    sd      s4, 128(sp)
    sd      s5, 120(sp)
    sd      s6, 112(sp)
    sd      s7, 104(sp)
    sd      s8,  96(sp)
    sd      s9,  88(sp)
    sd      s10, 80(sp)
    sd      s11, 72(sp)

    mv      s0, a0              # K 276×69
    mv      s1, a1              # H 69×276
    mv      s2, a2              # P_pred
    mv      s3, a3              # R 69×69
    mv      s4, a4              # x_pred
    mv      s5, a5              # y 69×1
    mv      s6, a6              # x output
    mv      s7, a7              # P output

    # Stack-passed args are at old sp + 0..56 = current sp + 176..232
    ld      s8,  176(sp)        # KH
    ld      s9,  184(sp)        # IKH
    ld      s10, 192(sp)        # tmp1
    ld      s11, 200(sp)        # tmp2
    ld      t0,  208(sp)        # KR
    sd      t0, 0(sp)
    ld      t0,  216(sp)        # KRKt
    sd      t0, 8(sp)
    ld      t0,  224(sp)        # Kt
    sd      t0, 16(sp)
    ld      t0,  232(sp)        # I276
    sd      t0, 24(sp)

    # x = K*y
    mv      a0, s0
    mv      a1, s5
    mv      a2, s6
    li      a3, 276
    li      a4, 69
    li      a5, 1
    call    mat_mul

    # x = x_pred + K*y
    mv      a0, s4
    mv      a1, s6
    mv      a2, s6
    li      a3, 276
    li      a4, 1
    call    mat_add

    # KH = K*H
    mv      a0, s0
    mv      a1, s1
    mv      a2, s8
    li      a3, 276
    li      a4, 69
    li      a5, 276
    call    mat_mul

    # IKH = I276 - KH
    ld      t0, 24(sp)
    mv      a0, t0
    mv      a1, s8
    mv      a2, s9
    li      a3, 276
    li      a4, 276
    call    mat_sub

    # tmp1 = IKH * P_pred
    mv      a0, s9
    mv      a1, s2
    mv      a2, s10
    li      a3, 276
    li      a4, 276
    li      a5, 276
    call    mat_mul

    # tmp2 = IKH^T
    mv      a0, s9
    mv      a1, s11
    li      a2, 276
    li      a3, 276
    call    mat_transpose

    # KRKt buffer temporarily stores Joseph term = tmp1 * IKH^T
    ld      t0, 8(sp)           # KRKt
    mv      a0, s10
    mv      a1, s11
    mv      a2, t0
    li      a3, 276
    li      a4, 276
    li      a5, 276
    call    mat_mul

    # KR = K*R
    ld      t0, 0(sp)           # KR
    mv      a0, s0
    mv      a1, s3
    mv      a2, t0
    li      a3, 276
    li      a4, 69
    li      a5, 69
    call    mat_mul

    # Kt = K^T
    ld      t1, 16(sp)          # Kt
    mv      a0, s0
    mv      a1, t1
    li      a2, 276
    li      a3, 69
    call    mat_transpose

    # tmp1 = KR * Kt = K*R*K^T
    ld      t0, 0(sp)           # KR
    ld      t1, 16(sp)          # Kt
    mv      a0, t0
    mv      a1, t1
    mv      a2, s10
    li      a3, 276
    li      a4, 69
    li      a5, 276
    call    mat_mul

    # P = Joseph term + noise term
    ld      t0, 8(sp)           # Joseph term in KRKt buffer
    mv      a0, t0
    mv      a1, s10
    mv      a2, s7
    li      a3, 276
    li      a4, 276
    call    mat_add

    ld      s11, 72(sp)
    ld      s10, 80(sp)
    ld      s9,  88(sp)
    ld      s8,  96(sp)
    ld      s7, 104(sp)
    ld      s6, 112(sp)
    ld      s5, 120(sp)
    ld      s4, 128(sp)
    ld      s3, 136(sp)
    ld      s2, 144(sp)
    ld      s1, 152(sp)
    ld      s0, 160(sp)
    ld      ra, 168(sp)
    addi    sp, sp, 176
    ret

# =============================================================================
# END OF lkf_vector_fixed.s
# =============================================================================
