# =============================================================================
#  lkf_asm.s  —  Linear Kalman Filter: RISC-V scalar assembly
#
#  Functions implemented:
#    1.  mat_zero        zero a matrix
#    2.  mat_eye         identity matrix
#    3.  mat_mul         matrix multiply  C = A*B
#    4.  mat_add         matrix add       C = A+B
#    5.  mat_sub         matrix subtract  C = A-B
#    6.  mat_transpose   transpose        B = Aᵀ
#    7.  mat_scale       scalar multiply  B = s*A
#    8.  chol_solve      Cholesky solve   A*x = b  (A is 3×3 SPD)
#    9.  compute_gain    Kalman gain      K = P*Hᵀ*S⁻¹
#   10.  lkf_predict     predict step     x_pred, P_pred
#   11.  lkf_update      update step      Joseph-form covariance
#
#  Calling convention: RISC-V LP64D
#    Integer args : a0–a7  (x10–x17)
#    FP args      : fa0–fa7 (f10–f17)
#    Saved regs   : s0–s11, fs0–fs11  — MUST be preserved
#    Temporaries  : t0–t6,  ft0–ft11  — caller-saved, free to clobber
#    Return addr  : ra (x1)           — MUST save if we call anything
#
#  All matrices are row-major doubles (8 bytes each).
#  Indexing: element [i][j] of an r×c matrix = base + (i*c + j)*8
# =============================================================================

    .section .text

# =============================================================================
# 1. mat_zero(double* A, int r, int c)
#    Sets all r*c elements of A to 0.0
#
#    Args:
#      a0 = double* A   (pointer to matrix data)
#      a1 = int r       (rows)
#      a2 = int c       (cols)
#
#    Register use:
#      a0  — base pointer, advances each iteration
#      a3  — total element count = r*c
#      ft0 — constant 0.0
# =============================================================================

    .global mat_zero
mat_zero:
    # no calls made, no saved regs needed — leaf function
    mul     a3, a1, a2          # a3 = r*c  (total elements)
    beqz    a3, mat_zero_done   # nothing to do if 0 elements
    fmv.d.x ft0, zero           # ft0 = 0.0  (move int 0 into FP reg)
mat_zero_loop:
    fsd     ft0, 0(a0)          # store 0.0 at A[i]
    addi    a0, a0, 8           # advance pointer by 8 bytes (one double)
    addi    a3, a3, -1          # decrement counter
    bnez    a3, mat_zero_loop   # loop until done
mat_zero_done:
    ret
# =============================================================================

# 2. mat_eye(double* A, int n)
#    Sets A to the n×n identity matrix.
#    First zeros the whole matrix, then sets diagonal to 1.0
#
#    Args:
#      a0 = double* A
#      a1 = int n
#
#    Register use:
#      a0  — base pointer
#      a1  — n (dimension)
#      a2  — total elements for zero pass
#      a3  — diagonal stride in bytes = (n+1)*8
#      a4  — loop counter for diagonal
#      ft0 — 0.0
#      ft1 — 1.0
# =============================================================================

    .global mat_eye
mat_eye:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0, 16(sp)
    sd      s1,  8(sp)

    mv      s0, a0              # save base pointer
    mv      s1, a1              # save n

    # zero entire matrix first (call mat_zero)
    mv      a2, a1              # c = n
    call    mat_zero            # mat_zero(A, n, n)

    # set diagonal: A[i*n+i] = 1.0
    # stride between diagonal elements = (n+1) doubles = (n+1)*8 bytes
    fmv.d.x ft1, zero           # ft1 = 0.0 first
    li      t0, 1
    fcvt.d.w ft1, t0            # ft1 = 1.0
    mv      a0, s0              # restore base
    mv      a4, s1              # loop counter = n
    addi    t1, s1, 1           # t1 = n+1
    slli    t1, t1, 3           # t1 = (n+1)*8  (diagonal stride in bytes)
mat_eye_diag:
    beqz    a4, mat_eye_done
    fsd     ft1, 0(a0)          # A[i*n+i] = 1.0
    add     a0, a0, t1          # advance to next diagonal element
    addi    a4, a4, -1
    j       mat_eye_diag
mat_eye_done:
    ld      ra, 24(sp)
    ld      s0, 16(sp)
    ld      s1,  8(sp)
    addi    sp, sp, 32
    ret


# =============================================================================
# 3. mat_mul(double* A, double* B, double* C, int r, int k, int c)
#    C = A * B   where A is r×k, B is k×c, C is r×c
#    Uses fmadd.d for fused multiply-accumulate as required by spec.
#
#    Args:
#      a0 = double* A
#      a1 = double* B
#      a2 = double* C
#      a3 = int r
#      a4 = int k
#      a5 = int c
#
#    Register use:
#      s0–s5  — saved copies of A, B, C, r, k, c
#      s6     — row loop counter i
#      s7     — col loop counter j
#      s8     — inner loop counter p
#      ft0    — accumulator s
#      ft1    — A[i*k+p]
#      ft2    — B[p*c+j]
#      t0–t3  — address calculations
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

    mv      s0, a0              # A base
    mv      s1, a1              # B base
    mv      s2, a2              # C base
    mv      s3, a3              # r
    mv      s4, a4              # k
    mv      s5, a5              # c

    li      s6, 0               # i = 0
mat_mul_i:
    bge     s6, s3, mat_mul_done    # i >= r → done

    li      s7, 0               # j = 0
mat_mul_j:
    bge     s7, s5, mat_mul_j_done  # j >= c → next i

    # accumulator = 0.0
    fmv.d.x ft0, zero

    li      s8, 0               # p = 0
mat_mul_p:
    bge     s8, s4, mat_mul_p_done  # p >= k → store result

    # A[i*k+p]: offset = (i*k + p)*8
    mul     t0, s6, s4          # t0 = i*k
    add     t0, t0, s8          # t0 = i*k + p
    slli    t0, t0, 3           # t0 = (i*k+p)*8
    add     t0, s0, t0          # t0 = &A[i*k+p]
    fld     ft1, 0(t0)          # ft1 = A[i*k+p]

    # B[p*c+j]: offset = (p*c + j)*8
    mul     t1, s8, s5          # t1 = p*c
    add     t1, t1, s7          # t1 = p*c + j
    slli    t1, t1, 3           # t1 = (p*c+j)*8
    add     t1, s1, t1          # t1 = &B[p*c+j]
    fld     ft2, 0(t1)          # ft2 = B[p*c+j]

    # s += A[i*k+p] * B[p*c+j]  using fmadd.d: ft0 = ft1*ft2 + ft0
    fmadd.d ft0, ft1, ft2, ft0

    addi    s8, s8, 1
    j       mat_mul_p

mat_mul_p_done:
    # C[i*c+j] = s
    mul     t2, s6, s5          # t2 = i*c
    add     t2, t2, s7          # t2 = i*c + j
    slli    t2, t2, 3           # t2 = (i*c+j)*8
    add     t2, s2, t2          # t2 = &C[i*c+j]
    fsd     ft0, 0(t2)          # C[i*c+j] = s

    addi    s7, s7, 1
    j       mat_mul_j

mat_mul_j_done:
    addi    s6, s6, 1
    j       mat_mul_i

mat_mul_done:
    ld      ra, 72(sp)
    ld      s0, 64(sp)
    ld      s1, 56(sp)
    ld      s2, 48(sp)
    ld      s3, 40(sp)
    ld      s4, 32(sp)
    ld      s5, 24(sp)
    ld      s6, 16(sp)
    ld      s7,  8(sp)
    ld      s8,  0(sp)
    addi    sp, sp, 80
    ret


# =============================================================================
# 4. mat_add(double* A, double* B, double* C, int r, int c)
#    C = A + B  element-wise
#
#    Args:
#      a0 = double* A
#      a1 = double* B
#      a2 = double* C
#      a3 = int r
#      a4 = int c
# =============================================================================
    .global mat_add
mat_add:
    mul     a3, a3, a4          # a3 = r*c (total elements)
    beqz    a3, mat_add_done
mat_add_loop:
    fld     ft0, 0(a0)          # ft0 = A[i]
    fld     ft1, 0(a1)          # ft1 = B[i]
    fadd.d  ft2, ft0, ft1       # ft2 = A[i] + B[i]
    fsd     ft2, 0(a2)          # C[i] = ft2
    addi    a0, a0, 8
    addi    a1, a1, 8
    addi    a2, a2, 8
    addi    a3, a3, -1
    bnez    a3, mat_add_loop
mat_add_done:
    ret


# =============================================================================
# 5. mat_sub(double* A, double* B, double* C, int r, int c)
#    C = A - B  element-wise
#
#    Args:
#      a0 = double* A
#      a1 = double* B
#      a2 = double* C
#      a3 = int r
#      a4 = int c
# =============================================================================
    .global mat_sub
mat_sub:
    mul     a3, a3, a4          # a3 = r*c
    beqz    a3, mat_sub_done
mat_sub_loop:
    fld     ft0, 0(a0)
    fld     ft1, 0(a1)
    fsub.d  ft2, ft0, ft1       # ft2 = A[i] - B[i]
    fsd     ft2, 0(a2)
    addi    a0, a0, 8
    addi    a1, a1, 8
    addi    a2, a2, 8
    addi    a3, a3, -1
    bnez    a3, mat_sub_loop
mat_sub_done:
    ret


# =============================================================================
# 6. mat_transpose(double* A, double* B, int r, int c)
#    B = Aᵀ   where A is r×c, B is c×r
#    B[j*r+i] = A[i*c+j]
#
#    Args:
#      a0 = double* A
#      a1 = double* B
#      a2 = int r
#      a3 = int c
#
#    Register use:
#      s0–s3  — A, B, r, c
#      s4, s5 — loop counters i, j
#      t0, t1 — address offsets
# =============================================================================
    .global mat_transpose
mat_transpose:
    addi    sp, sp, -56
    sd      ra, 48(sp)
    sd      s0, 40(sp)
    sd      s1, 32(sp)
    sd      s2, 24(sp)
    sd      s3, 16(sp)
    sd      s4,  8(sp)
    sd      s5,  0(sp)

    mv      s0, a0              # A
    mv      s1, a1              # B
    mv      s2, a2              # r
    mv      s3, a3              # c

    li      s4, 0               # i = 0
mat_T_i:
    bge     s4, s2, mat_T_done

    li      s5, 0               # j = 0
mat_T_j:
    bge     s5, s3, mat_T_j_done

    # src = A[i*c+j]
    mul     t0, s4, s3          # i*c
    add     t0, t0, s5          # i*c+j
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # ft0 = A[i*c+j]

    # dst = B[j*r+i]
    mul     t1, s5, s2          # j*r
    add     t1, t1, s4          # j*r+i
    slli    t1, t1, 3
    add     t1, s1, t1
    fsd     ft0, 0(t1)          # B[j*r+i] = A[i*c+j]

    addi    s5, s5, 1
    j       mat_T_j
mat_T_j_done:
    addi    s4, s4, 1
    j       mat_T_i
mat_T_done:
    ld      ra, 48(sp)
    ld      s0, 40(sp)
    ld      s1, 32(sp)
    ld      s2, 24(sp)
    ld      s3, 16(sp)
    ld      s4,  8(sp)
    ld      s5,  0(sp)
    addi    sp, sp, 56
    ret


# =============================================================================
# 7. mat_scale(double* A, double s, double* B, int r, int c)
#    B = s * A  element-wise
#
#    Args:
#      a0  = double* A
#      fa0 = double s   (scalar — first FP argument goes in fa0)
#      a1  = double* B
#      a2  = int r
#      a3  = int c
# =============================================================================
    .global mat_scale
mat_scale:
    mul     a2, a2, a3          # a2 = r*c
    beqz    a2, mat_scale_done
mat_scale_loop:
    fld     ft1, 0(a0)          # ft1 = A[i]
    fmul.d  ft2, ft1, fa0       # ft2 = s * A[i]
    fsd     ft2, 0(a1)          # B[i] = ft2
    addi    a0, a0, 8
    addi    a1, a1, 8
    addi    a2, a2, -1
    bnez    a2, mat_scale_loop
mat_scale_done:
    ret


# =============================================================================
# 8. chol_solve(double* A, double* b, double* x, int n)
#    Solves A*x = b where A is n×n symmetric positive definite.
#    Strategy: Cholesky decomposition A = L*Lᵀ, then forward/back sub.
#    In our filter n = N_MEAS = 3, so this is always a 3×3 solve.
#
#    Args:
#      a0 = double* A  (n×n SPD matrix, READ-ONLY)
#      a1 = double* b  (n×1 right-hand side)
#      a2 = double* x  (n×1 output solution)
#      a3 = int n
#
#    Stack layout (local arrays, n≤3 so bounded size):
#      We allocate L[n*n] and y[n] on the stack.
#      For n=3: L needs 9*8=72 bytes, y needs 3*8=24 bytes → 96 bytes.
#      We allocate 128 bytes for safety + saved regs.
#
#    Register use:
#      s0–s3  — A, b, x, n
#      s4     — pointer to L (on stack)
#      s5     — pointer to y (on stack)
#      s6–s8  — loop counters i, j, k
#      ft0–ft7 — FP temporaries
# =============================================================================
    .global chol_solve
chol_solve:
    addi    sp, sp, -192
    sd      ra,  184(sp)
    sd      s0,  176(sp)
    sd      s1,  168(sp)
    sd      s2,  160(sp)
    sd      s3,  152(sp)
    sd      s4,  144(sp)
    sd      s5,  136(sp)
    sd      s6,  128(sp)
    sd      s7,  120(sp)
    sd      s8,  112(sp)

    mv      s0, a0              # A
    mv      s1, a1              # b
    mv      s2, a2              # x
    mv      s3, a3              # n

    # L lives at sp+0  (n*n doubles = up to 9*8 = 72 bytes)
    # y lives at sp+72 (n doubles   = up to 3*8 = 24 bytes)
    addi    s4, sp, 0           # s4 = &L[0]
    addi    s5, sp, 72          # s5 = &y[0]

    # ── zero L ──────────────────────────────────────────────────────────────
    fmv.d.x ft0, zero
    mul     t0, s3, s3          # n*n
    mv      t1, s4
chol_zero_L:
    beqz    t0, chol_zero_L_done
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    j       chol_zero_L
chol_zero_L_done:

    # ── Cholesky decomposition: L[i][j] for i=0..n-1, j=0..i ───────────────
    li      s6, 0               # i = 0
chol_i:
    bge     s6, s3, chol_fwd    # done with decomp, go to forward sub

    li      s7, 0               # j = 0
chol_j:
    bgt     s7, s6, chol_j_done # j > i → next i

    # s = A[i*n+j]
    mul     t0, s6, s3          # i*n
    add     t0, t0, s7          # i*n+j
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # ft0 = A[i*n+j]  (s)

    # subtract sum: for k=0..j-1: s -= L[i*n+k] * L[j*n+k]
    li      s8, 0               # k = 0
chol_k:
    bge     s8, s7, chol_k_done # k >= j → done inner sum

    # L[i*n+k]
    mul     t0, s6, s3
    add     t0, t0, s8
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)          # ft1 = L[i*n+k]

    # L[j*n+k]
    mul     t1, s7, s3
    add     t1, t1, s8
    slli    t1, t1, 3
    add     t1, s4, t1
    fld     ft2, 0(t1)          # ft2 = L[j*n+k]

    # s -= L[i*n+k] * L[j*n+k]  using fmsub: ft0 = ft0 - ft1*ft2
    fmul.d  ft3, ft1, ft2
    fsub.d  ft0, ft0, ft3

    addi    s8, s8, 1
    j       chol_k
chol_k_done:

    # if i == j: L[i*i] = sqrt(s)  (clamp s to EPS if near zero)
    bne     s6, s7, chol_off_diag

    # clamp: if s < 1e-15, set s = 1e-15
    li      t0, 1
    fcvt.d.w ft4, t0
    li      t0, 100000000000000  # 1e14
    fcvt.d.l ft5, t0
    fdiv.d  ft5, ft4, ft5        # ft5 = 1e-15
    flt.d   t0, ft0, ft5
    beqz    t0, chol_no_clamp
    fmv.d   ft0, ft5             # clamp to 1e-15
chol_no_clamp:
    fsqrt.d ft0, ft0             # L[i*n+i] = sqrt(s)
    j       chol_store

chol_off_diag:
    # L[i*n+j] = s / L[j*n+j]
    mul     t1, s7, s3
    add     t1, t1, s7
    slli    t1, t1, 3
    add     t1, s4, t1
    fld     ft4, 0(t1)           # ft4 = L[j*n+j]
    fdiv.d  ft0, ft0, ft4        # ft0 = s / L[j*n+j]

chol_store:
    # store result at L[i*n+j]
    mul     t2, s6, s3
    add     t2, t2, s7
    slli    t2, t2, 3
    add     t2, s4, t2
    fsd     ft0, 0(t2)

    addi    s7, s7, 1
    j       chol_j
chol_j_done:
    addi    s6, s6, 1
    j       chol_i

    # ── Forward substitution: L*y = b ────────────────────────────────────
chol_fwd:
    li      s6, 0               # i = 0
chol_fwd_i:
    bge     s6, s3, chol_bwd

    # s = b[i]
    slli    t0, s6, 3
    add     t0, s1, t0
    fld     ft0, 0(t0)          # ft0 = b[i]

    # subtract: for k=0..i-1: s -= L[i*n+k]*y[k]
    li      s8, 0
chol_fwd_k:
    bge     s8, s6, chol_fwd_k_done

    mul     t0, s6, s3
    add     t0, t0, s8
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)           # L[i*n+k]

    slli    t1, s8, 3
    add     t1, s5, t1
    fld     ft2, 0(t1)           # y[k]

    fmul.d  ft3, ft1, ft2
    fsub.d  ft0, ft0, ft3

    addi    s8, s8, 1
    j       chol_fwd_k
chol_fwd_k_done:

    # y[i] = s / L[i*n+i]
    mul     t0, s6, s3
    add     t0, t0, s6
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)           # L[i*n+i]
    fdiv.d  ft0, ft0, ft1        # s / L[i*n+i]

    slli    t1, s6, 3
    add     t1, s5, t1
    fsd     ft0, 0(t1)           # y[i] = result

    addi    s6, s6, 1
    j       chol_fwd_i

    # ── Backward substitution: Lᵀ*x = y ─────────────────────────────────
chol_bwd:
    addi    s6, s3, -1          # i = n-1
chol_bwd_i:
    bltz    s6, chol_solve_done

    # s = y[i]
    slli    t0, s6, 3
    add     t0, s5, t0
    fld     ft0, 0(t0)           # ft0 = y[i]

    # subtract: for k=i+1..n-1: s -= L[k*n+i]*x[k]
    addi    s8, s6, 1            # k = i+1
chol_bwd_k:
    bge     s8, s3, chol_bwd_k_done

    # L[k*n+i]  (Lᵀ[i*n+k] = L[k*n+i])
    mul     t0, s8, s3
    add     t0, t0, s6
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)

    # x[k]
    slli    t1, s8, 3
    add     t1, s2, t1
    fld     ft2, 0(t1)

    fmul.d  ft3, ft1, ft2
    fsub.d  ft0, ft0, ft3

    addi    s8, s8, 1
    j       chol_bwd_k
chol_bwd_k_done:

    # x[i] = s / L[i*n+i]
    mul     t0, s6, s3
    add     t0, t0, s6
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)
    fdiv.d  ft0, ft0, ft1

    slli    t1, s6, 3
    add     t1, s2, t1
    fsd     ft0, 0(t1)           # x[i] = result

    addi    s6, s6, -1
    j       chol_bwd_i

chol_solve_done:
    ld      ra,  184(sp)
    ld      s0,  176(sp)
    ld      s1,  168(sp)
    ld      s2,  160(sp)
    ld      s3,  152(sp)
    ld      s4,  144(sp)
    ld      s5,  136(sp)
    ld      s6,  128(sp)
    ld      s7,  120(sp)
    ld      s8,  112(sp)
    addi    sp, sp, 192
    ret


# =============================================================================
# 9. compute_gain(double* P, double* H, double* S, double* K)
#    K = P * Hᵀ * S⁻¹
#    Sizes: P(12×12), H(3×12), S(3×3), K(12×3)
#    Method: for each column j of K:
#      1. solve S * e_j = I[:,j]  → Sinv_col  (3×1)
#      2. K[:,j] = PHt * Sinv_col  (12×3 * 3×1 = 12×1)
#
#    Args:
#      a0 = double* P    (12×12)
#      a1 = double* H    (3×12)
#      a2 = double* S    (3×3)
#      a3 = double* K    (12×3)
#
#    Stack locals:
#      Ht[12*3=36 doubles = 288 bytes]
#      PHt[12*3=36 doubles = 288 bytes]
#      ej[3 doubles = 24 bytes]
#      Sinv_col[3 doubles = 24 bytes]
#      Total locals: 624 bytes + saved regs
# =============================================================================
    .global compute_gain
compute_gain:
    addi    sp, sp, -736
    sd      ra,  728(sp)
    sd      s0,  720(sp)
    sd      s1,  712(sp)
    sd      s2,  704(sp)
    sd      s3,  696(sp)
    sd      s4,  688(sp)
    sd      s5,  680(sp)
    sd      s6,  672(sp)

    mv      s0, a0              # P
    mv      s1, a1              # H
    mv      s2, a2              # S
    mv      s3, a3              # K

    # local buffers on stack
    addi    s4, sp, 0           # Ht      at sp+0   (288 bytes)
    addi    s5, sp, 288         # PHt     at sp+288 (288 bytes)
    addi    s6, sp, 576         # ej      at sp+576 (24 bytes)
    addi    t6, sp, 600         # Sinv_col at sp+600 (24 bytes)
    # save t6 into a saved reg since t6 is caller-saved
    sd      s7, 664(sp)
    mv      s7, t6              # s7 = &Sinv_col

    # ── compute Ht = Hᵀ  (H is 3×12, Ht is 12×3) ─────────────────────
    mv      a0, s1
    mv      a1, s4
    li      a2, 3
    li      a3, 12
    call    mat_transpose       # mat_transpose(H, Ht, 3, 12)

    # ── compute PHt = P * Ht  (12×12 * 12×3 = 12×3) ─────────────────────
    mv      a0, s0
    mv      a1, s4
    mv      a2, s5
    li      a3, 12
    li      a4, 12
    li      a5, 3
    call    mat_mul             # mat_mul(P, Ht, PHt, 12, 12, 3)

    # ── for j = 0, 1, 2: solve S*ej = I[:,j], then K[:,j] = PHt*Sinv ─────
    li      s6, 0               # j = 0

    # rebuild s6 as ej pointer — we lost it above, reload
    addi    s6, sp, 576         # s6 = &ej

compute_gain_j:
    # figure out which column j we're on from a counter
    # use t4 as j counter (caller-saved, safe between calls only if we save)
    # simpler: unroll for j=0,1,2 using a counter in a saved reg
    # Let's use s8 for j
    sd      s8, 656(sp)

    li      s8, 0               # j = 0
compute_gain_loop:
    li      t0, 3
    bge     s8, t0, compute_gain_done

    # zero ej[0..2]
    fmv.d.x ft0, zero
    fsd     ft0, 0(s6)
    fsd     ft0, 8(s6)
    fsd     ft0, 16(s6)

    # ej[j] = 1.0
    li      t0, 1
    fcvt.d.w ft1, t0            # ft1 = 1.0
    slli    t1, s8, 3           # j*8
    add     t1, s6, t1
    fsd     ft1, 0(t1)          # ej[j] = 1.0

    # solve S * Sinv_col = ej
    mv      a0, s2              # S
    mv      a1, s6              # ej
    mv      a2, s7              # Sinv_col
    li      a3, 3
    call    chol_solve

    # K[:,j] = PHt * Sinv_col  (12×3 * 3×1 = 12×1)
    # PHt is at s5 (12×3), Sinv_col is at s7 (3×1)
    # result column j of K: K[i*3+j] for i=0..11
    li      t2, 0               # i = 0
compute_gain_col:
    li      t0, 12
    bge     t2, t0, compute_gain_col_done

    # dot product: sum = PHt[i*3+0]*Sinv[0] + PHt[i*3+1]*Sinv[1] + PHt[i*3+2]*Sinv[2]
    fmv.d.x ft0, zero           # accumulator

    li      t3, 0               # inner k=0,1,2
compute_gain_dot:
    li      t0, 3
    bge     t3, t0, compute_gain_dot_done

    # PHt[i*3+k]
    li      t0, 3
    mul     t4, t2, t0          # i*3
    add     t4, t4, t3          # i*3+k
    slli    t4, t4, 3
    add     t4, s5, t4
    fld     ft1, 0(t4)

    # Sinv_col[k]
    slli    t4, t3, 3
    add     t4, s7, t4
    fld     ft2, 0(t4)

    fmadd.d ft0, ft1, ft2, ft0  # accumulate

    addi    t3, t3, 1
    j       compute_gain_dot
compute_gain_dot_done:

    # store K[i*3+j] = sum
    li      t0, 3
    mul     t4, t2, t0          # i*3
    add     t4, t4, s8          # i*3+j
    slli    t4, t4, 3
    add     t4, s3, t4
    fsd     ft0, 0(t4)

    addi    t2, t2, 1
    j       compute_gain_col
compute_gain_col_done:

    addi    s8, s8, 1
    j       compute_gain_loop

compute_gain_done:
    ld      ra,  728(sp)
    ld      s0,  720(sp)
    ld      s1,  712(sp)
    ld      s2,  704(sp)
    ld      s3,  696(sp)
    ld      s4,  688(sp)
    ld      s5,  680(sp)
    ld      s6,  672(sp)
    ld      s7,  664(sp)
    ld      s8,  656(sp)
    addi    sp, sp, 736
    ret


# =============================================================================
# 10. lkf_predict(F, Ft, P, Q, x, x_pred, P_pred, FP, FPFt)
#     x_pred   = F * x
#     P_pred   = F * P * Fᵀ + Q
#
#     Args (all pointers, 12×12 matrices unless noted):
#       a0 = F      (12×12)
#       a1 = Ft     (12×12)
#       a2 = P      (12×12)
#       a3 = Q      (12×12)
#       a4 = x      (12×1)
#       a5 = x_pred (12×1)
#       a6 = P_pred (12×12)
#       a7 = FP     (12×12) scratch
#     Stack arg:
#       0(sp) on entry = FPFt (12×12) scratch
# =============================================================================
    .global lkf_predict
lkf_predict:
    addi    sp, sp, -96
    sd      ra,  88(sp)
    sd      s0,  80(sp)
    sd      s1,  72(sp)
    sd      s2,  64(sp)
    sd      s3,  56(sp)
    sd      s4,  48(sp)
    sd      s5,  40(sp)
    sd      s6,  32(sp)
    sd      s7,  24(sp)
    sd      s8,  16(sp)

    mv      s0, a0              # F
    mv      s1, a1              # Ft
    mv      s2, a2              # P
    mv      s3, a3              # Q
    mv      s4, a4              # x
    mv      s5, a5              # x_pred
    mv      s6, a6              # P_pred
    mv      s7, a7              # FP
    # FPFt is passed on stack — before we modified sp
    # caller's stack arg was at old_sp+0, now at sp+96
    ld      s8, 96(sp)          # s8 = FPFt

    # ── x_pred = F * x  (12×12 * 12×1 = 12×1) ──────────────────────────
    mv      a0, s0
    mv      a1, s4
    mv      a2, s5
    li      a3, 12
    li      a4, 12
    li      a5, 1
    call    mat_mul

    # ── FP = F * P  (12×12 * 12×12 = 12×12) ────────────────────────────
    mv      a0, s0
    mv      a1, s2
    mv      a2, s7
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # ── FPFt = FP * Ft  (12×12 * 12×12 = 12×12) ────────────────────────
    mv      a0, s7
    mv      a1, s1
    mv      a2, s8
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # ── P_pred = FPFt + Q ────────────────────────────────────────────────
    mv      a0, s8
    mv      a1, s3
    mv      a2, s6
    li      a3, 12
    li      a4, 12
    call    mat_add

    ld      ra,  88(sp)
    ld      s0,  80(sp)
    ld      s1,  72(sp)
    ld      s2,  64(sp)
    ld      s3,  56(sp)
    ld      s4,  48(sp)
    ld      s5,  40(sp)
    ld      s6,  32(sp)
    ld      s7,  24(sp)
    ld      s8,  16(sp)
    addi    sp, sp, 96
    ret


# =============================================================================
# 11. lkf_update(K, H, P_pred, R, x_pred, y, x, P,
#                KH, IKH, tmp1, tmp2, KR, KRKt, Kt, I12)
#     Joseph-form covariance update:
#       x    = x_pred + K*y
#       IKH  = I - K*H
#       P    = IKH * P_pred * IKHᵀ + K*R*Kᵀ
#
#     Args (a0–a7 = first 8, rest on stack):
#       a0 = K      (12×3)
#       a1 = H      (3×12)
#       a2 = P_pred (12×12)
#       a3 = R      (3×3)
#       a4 = x_pred (12×1)
#       a5 = y      (3×1)
#       a6 = x      (12×1)  output
#       a7 = P      (12×12) output
#     Stack (8-byte each, at old_sp+0,8,...):
#       +0  = KH    (12×12)
#       +8  = IKH   (12×12)
#       +16 = tmp1  (12×12)
#       +24 = tmp2  (12×12)
#       +32 = KR    (12×3)
#       +40 = KRKt  (12×12)
#       +48 = Kt    (3×12)
#       +56 = I12   (12×12)
# =============================================================================
    .global lkf_update
lkf_update:
    addi    sp, sp, -112
    sd      ra,  104(sp)
    sd      s0,   96(sp)
    sd      s1,   88(sp)
    sd      s2,   80(sp)
    sd      s3,   72(sp)
    sd      s4,   64(sp)
    sd      s5,   56(sp)
    sd      s6,   48(sp)
    sd      s7,   40(sp)
    sd      s8,   32(sp)
    sd      s9,   24(sp)
    sd      s10,  16(sp)
    sd      s11,   8(sp)

    mv      s0,  a0             # K
    mv      s1,  a1             # H
    mv      s2,  a2             # P_pred
    mv      s3,  a3             # R
    mv      s4,  a4             # x_pred
    mv      s5,  a5             # y
    mv      s6,  a6             # x (output)
    mv      s7,  a7             # P (output)

    # load stack args — they were at old_sp+0..+56, now at sp+112+0..
    ld      s8,  112(sp)        # KH
    ld      s9,  120(sp)        # IKH
    ld      s10, 128(sp)        # tmp1
    ld      s11, 136(sp)        # tmp2
    # need more saved regs — use t regs carefully after calls
    ld      t3,  144(sp)        # KR
    ld      t4,  152(sp)        # KRKt
    ld      t5,  160(sp)        # Kt
    ld      t6,  168(sp)        # I12
    # save t3–t6 as we'll need them after calls
    addi    sp, sp, -32
    sd      t3, 24(sp)
    sd      t4, 16(sp)
    sd      t5,  8(sp)
    sd      t6,  0(sp)

    # ── x = x_pred + K*y ────────────────────────────────────────────────
    # first compute Ky = K*y into x (temporary), then add x_pred
    # use x as temp for K*y, then add
    mv      a0, s0              # K
    mv      a1, s5              # y
    mv      a2, s6              # x (store K*y here first)
    li      a3, 12
    li      a4, 3
    li      a5, 1
    call    mat_mul             # x = K*y  (temporary)

    mv      a0, s4              # x_pred
    mv      a1, s6              # K*y
    mv      a2, s6              # x = x_pred + K*y
    li      a3, 12
    li      a4, 1
    call    mat_add

    # ── KH = K * H  (12×3 * 3×12 = 12×12) ──────────────────────────────
    mv      a0, s0
    mv      a1, s1
    mv      a2, s8
    li      a3, 12
    li      a4, 3
    li      a5, 12
    call    mat_mul

    # ── IKH = I12 - KH ──────────────────────────────────────────────────
    ld      t6, 0(sp)           # I12
    mv      a0, t6
    mv      a1, s8
    mv      a2, s9
    li      a3, 12
    li      a4, 12
    call    mat_sub

    # ── tmp1 = IKH * P_pred  (12×12 * 12×12) ───────────────────────────
    mv      a0, s9
    mv      a1, s2
    mv      a2, s10
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # ── IKHt = IKHᵀ  — store into Kt temporarily (reuse buffer) ────────
    ld      t5, 8(sp)           # Kt buffer (3×12), but we need 12×12
    # Actually Kt is 3×12 — not big enough for IKHt (12×12)
    # Use KRKt as scratch for IKHt since we compute it after
    ld      t4, 16(sp)          # KRKt
    mv      a0, s9              # IKH
    mv      a1, t4              # use KRKt as IKHt scratch
    li      a2, 12
    li      a3, 12
    call    mat_transpose        # IKHt = IKHᵀ  stored in KRKt buffer

    # ── tmp2 = tmp1 * IKHt  (12×12 * 12×12) ────────────────────────────
    mv      a0, s10
    mv      a1, t4              # IKHt (stored in KRKt buffer)
    mv      a2, s11
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # ── KR = K * R  (12×3 * 3×3 = 12×3) ────────────────────────────────
    ld      t3, 24(sp)          # KR
    mv      a0, s0
    mv      a1, s3
    mv      a2, t3
    li      a3, 12
    li      a4, 3
    li      a5, 3
    call    mat_mul

    # ── Kt = Kᵀ  (K is 12×3, Kt is 3×12) ───────────────────────────────
    ld      t5, 8(sp)
    mv      a0, s0
    mv      a1, t5
    li      a2, 12
    li      a3, 3
    call    mat_transpose

    # ── KRKt = KR * Kt  (12×3 * 3×12 = 12×12) ──────────────────────────
    # now put real KRKt into t4
    ld      t4, 16(sp)
    mv      a0, t3
    mv      a1, t5
    mv      a2, t4
    li      a3, 12
    li      a4, 3
    li      a5, 12
    call    mat_mul

    # ── P = tmp2 + KRKt ──────────────────────────────────────────────────
    mv      a0, s11
    mv      a1, t4
    mv      a2, s7
    li      a3, 12
    li      a4, 12
    call    mat_add

    addi    sp, sp, 32          # undo inner frame
    ld      ra,  104(sp)
    ld      s0,   96(sp)
    ld      s1,   88(sp)
    ld      s2,   80(sp)
    ld      s3,   72(sp)
    ld      s4,   64(sp)
    ld      s5,   56(sp)
    ld      s6,   48(sp)
    ld      s7,   40(sp)
    ld      s8,   32(sp)
    ld      s9,   24(sp)
    ld      s10,  16(sp)
    ld      s11,   8(sp)
    addi    sp, sp, 112
    ret

# end of lkf_asm.s