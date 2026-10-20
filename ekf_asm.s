# =============================================================================
#  ekf_asm.s  —  Extended Kalman Filter: RISC-V scalar assembly
#  Milestone 3
#
#  This file contains ALL functions needed for EKF:
#    Shared matrix primitives (same as lkf_asm.s):
#      1.  mat_zero
#      2.  mat_eye
#      3.  mat_mul
#      4.  mat_add
#      5.  mat_sub
#      6.  mat_transpose
#      7.  mat_scale
#      8.  chol_solve
#      9.  compute_gain
#
#    EKF-specific functions:
#     10.  manual_atan        internal helper: atan approximation on [-1,1]
#     11.  manual_atan2       internal helper: full quadrant atan2
#     12.  h_spherical        h(x): Cartesian → spherical measurement
#     13.  compute_jacobian   dh/dx: 3×12 Jacobian at x_pred
#     14.  ekf_predict        predict step (identical math to LKF)
#     15.  ekf_update         Joseph-form update using Jacobian J
#
#  Calling convention: RISC-V LP64D
#    Integer args : a0–a7   (x10–x17)
#    FP args      : fa0–fa7 (f10–f17)
#    Saved regs   : s0–s11, fs0–fs11  — MUST be preserved across calls
#    Temporaries  : t0–t6,  ft0–ft11  — caller-saved, free to clobber
#    Return addr  : ra (x1)           — MUST save before any call
#
#  arctan2 approximation (manual_atan2):
#    atan(x) ≈ x*(π/4 + 0.273*(1-|x|))  for |x| ≤ 1
#    Extended to full quadrants exactly as in the M2 C++ code.
#    Max error: ~0.0038 rad (~0.22°) — acceptable for gait tracking.
#
#  All matrices row-major doubles (8 bytes).
#  Element [i][j] of r×c matrix = base + (i*c + j)*8
# =============================================================================

    .section .text

# =============================================================================
# 1. mat_zero(double* A, int r, int c)
# =============================================================================
    .global mat_zero
mat_zero:
    mul     a3, a1, a2
    beqz    a3, mat_zero_done
    fmv.d.x ft0, zero
mat_zero_loop:
    fsd     ft0, 0(a0)
    addi    a0, a0, 8
    addi    a3, a3, -1
    bnez    a3, mat_zero_loop
mat_zero_done:
    ret


# =============================================================================
# 2. mat_eye(double* A, int n)
# =============================================================================
    .global mat_eye
mat_eye:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0, 16(sp)
    sd      s1,  8(sp)

    mv      s0, a0
    mv      s1, a1

    mv      a2, a1
    call    mat_zero

    fmv.d.x ft1, zero
    li      t0, 1
    fcvt.d.w ft1, t0            # ft1 = 1.0
    mv      a0, s0
    mv      a4, s1
    addi    t1, s1, 1
    slli    t1, t1, 3           # diagonal stride = (n+1)*8
mat_eye_diag:
    beqz    a4, mat_eye_done
    fsd     ft1, 0(a0)
    add     a0, a0, t1
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
#    C = A*B  using fmadd.d for multiply-accumulate
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

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3              # r
    mv      s4, a4              # k
    mv      s5, a5              # c

    li      s6, 0               # i
mat_mul_i:
    bge     s6, s3, mat_mul_done
    li      s7, 0               # j
mat_mul_j:
    bge     s7, s5, mat_mul_j_done
    fmv.d.x ft0, zero           # accumulator
    li      s8, 0               # p
mat_mul_p:
    bge     s8, s4, mat_mul_p_done

    mul     t0, s6, s4
    add     t0, t0, s8
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft1, 0(t0)          # A[i*k+p]

    mul     t1, s8, s5
    add     t1, t1, s7
    slli    t1, t1, 3
    add     t1, s1, t1
    fld     ft2, 0(t1)          # B[p*c+j]

    fmadd.d ft0, ft1, ft2, ft0  # acc += A*B  (fused multiply-add)

    addi    s8, s8, 1
    j       mat_mul_p
mat_mul_p_done:
    mul     t2, s6, s5
    add     t2, t2, s7
    slli    t2, t2, 3
    add     t2, s2, t2
    fsd     ft0, 0(t2)          # C[i*c+j] = acc

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
# =============================================================================
    .global mat_add
mat_add:
    mul     a3, a3, a4
    beqz    a3, mat_add_done
mat_add_loop:
    fld     ft0, 0(a0)
    fld     ft1, 0(a1)
    fadd.d  ft2, ft0, ft1
    fsd     ft2, 0(a2)
    addi    a0, a0, 8
    addi    a1, a1, 8
    addi    a2, a2, 8
    addi    a3, a3, -1
    bnez    a3, mat_add_loop
mat_add_done:
    ret


# =============================================================================
# 5. mat_sub(double* A, double* B, double* C, int r, int c)
# =============================================================================
    .global mat_sub
mat_sub:
    mul     a3, a3, a4
    beqz    a3, mat_sub_done
mat_sub_loop:
    fld     ft0, 0(a0)
    fld     ft1, 0(a1)
    fsub.d  ft2, ft0, ft1
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
#    B = Aᵀ
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

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2              # r
    mv      s3, a3              # c

    li      s4, 0               # i
mat_T_i:
    bge     s4, s2, mat_T_done
    li      s5, 0               # j
mat_T_j:
    bge     s5, s3, mat_T_j_done

    mul     t0, s4, s3
    add     t0, t0, s5
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # A[i*c+j]

    mul     t1, s5, s2
    add     t1, t1, s4
    slli    t1, t1, 3
    add     t1, s1, t1
    fsd     ft0, 0(t1)          # B[j*r+i]

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
#    B = s * A
#    Note: scalar s is in fa0 (first FP argument register)
# =============================================================================
    .global mat_scale
mat_scale:
    mul     a2, a2, a3
    beqz    a2, mat_scale_done
mat_scale_loop:
    fld     ft1, 0(a0)
    fmul.d  ft2, ft1, fa0       # ft2 = s * A[i]
    fsd     ft2, 0(a1)
    addi    a0, a0, 8
    addi    a1, a1, 8
    addi    a2, a2, -1
    bnez    a2, mat_scale_loop
mat_scale_done:
    ret


# =============================================================================
# 8. chol_solve(double* A, double* b, double* x, int n)
#    Solves A*x = b  (A is n×n SPD, n=3 in our filter)
#    Uses Cholesky: A = L*Lᵀ, then forward + backward substitution
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

    addi    s4, sp, 0           # L at sp+0   (72 bytes max for 3×3)
    addi    s5, sp, 72          # y at sp+72  (24 bytes max for 3)

    # zero L
    fmv.d.x ft0, zero
    mul     t0, s3, s3
    mv      t1, s4
chol_zero_L:
    beqz    t0, chol_decomp
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    j       chol_zero_L

    # Cholesky decomposition
chol_decomp:
    li      s6, 0               # i
chol_i:
    bge     s6, s3, chol_fwd
    li      s7, 0               # j
chol_j:
    bgt     s7, s6, chol_j_done

    # s = A[i*n+j]
    mul     t0, s6, s3
    add     t0, t0, s7
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)

    # s -= sum(L[i*n+k]*L[j*n+k], k=0..j-1)
    li      s8, 0
chol_k:
    bge     s8, s7, chol_k_done
    mul     t0, s6, s3
    add     t0, t0, s8
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)          # L[i*n+k]

    mul     t1, s7, s3
    add     t1, t1, s8
    slli    t1, t1, 3
    add     t1, s4, t1
    fld     ft2, 0(t1)          # L[j*n+k]

    fmul.d  ft3, ft1, ft2
    fsub.d  ft0, ft0, ft3
    addi    s8, s8, 1
    j       chol_k
chol_k_done:

    bne     s6, s7, chol_off_diag

    # diagonal: clamp then sqrt
    li      t0, 1
    fcvt.d.w ft4, t0
    li      t0, 100000000000000
    fcvt.d.l ft5, t0
    fdiv.d  ft5, ft4, ft5       # ft5 = 1e-15
    flt.d   t0, ft0, ft5
    beqz    t0, chol_sqrt
    fmv.d   ft0, ft5
chol_sqrt:
    fsqrt.d ft0, ft0
    j       chol_store_L

chol_off_diag:
    mul     t1, s7, s3
    add     t1, t1, s7
    slli    t1, t1, 3
    add     t1, s4, t1
    fld     ft4, 0(t1)          # L[j*n+j]
    fdiv.d  ft0, ft0, ft4

chol_store_L:
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

    # Forward substitution: L*y = b
chol_fwd:
    li      s6, 0
chol_fwd_i:
    bge     s6, s3, chol_bwd

    slli    t0, s6, 3
    add     t0, s1, t0
    fld     ft0, 0(t0)          # s = b[i]

    li      s8, 0
chol_fwd_k:
    bge     s8, s6, chol_fwd_store
    mul     t0, s6, s3
    add     t0, t0, s8
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)          # L[i*n+k]

    slli    t1, s8, 3
    add     t1, s5, t1
    fld     ft2, 0(t1)          # y[k]

    fmul.d  ft3, ft1, ft2
    fsub.d  ft0, ft0, ft3
    addi    s8, s8, 1
    j       chol_fwd_k
chol_fwd_store:
    mul     t0, s6, s3
    add     t0, t0, s6
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)          # L[i*n+i]
    fdiv.d  ft0, ft0, ft1

    slli    t1, s6, 3
    add     t1, s5, t1
    fsd     ft0, 0(t1)          # y[i]

    addi    s6, s6, 1
    j       chol_fwd_i

    # Backward substitution: Lᵀ*x = y
chol_bwd:
    addi    s6, s3, -1          # i = n-1
chol_bwd_i:
    bltz    s6, chol_done

    slli    t0, s6, 3
    add     t0, s5, t0
    fld     ft0, 0(t0)          # s = y[i]

    addi    s8, s6, 1           # k = i+1
chol_bwd_k:
    bge     s8, s3, chol_bwd_store
    mul     t0, s8, s3
    add     t0, t0, s6
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)          # L[k*n+i]  = Lᵀ[i*n+k]

    slli    t1, s8, 3
    add     t1, s2, t1
    fld     ft2, 0(t1)          # x[k]

    fmul.d  ft3, ft1, ft2
    fsub.d  ft0, ft0, ft3
    addi    s8, s8, 1
    j       chol_bwd_k
chol_bwd_store:
    mul     t0, s6, s3
    add     t0, t0, s6
    slli    t0, t0, 3
    add     t0, s4, t0
    fld     ft1, 0(t0)          # L[i*n+i]
    fdiv.d  ft0, ft0, ft1

    slli    t1, s6, 3
    add     t1, s2, t1
    fsd     ft0, 0(t1)          # x[i]

    addi    s6, s6, -1
    j       chol_bwd_i

chol_done:
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
# 9. compute_gain(double* P, double* J, double* S, double* K)
#    K = P * Jᵀ * S⁻¹
#    P(12×12), J(3×12), S(3×3), K(12×3)
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
    sd      s7,  664(sp)
    sd      s8,  656(sp)

    mv      s0, a0              # P
    mv      s1, a1              # J (plays role of H in EKF)
    mv      s2, a2              # S
    mv      s3, a3              # K

    addi    s4, sp, 0           # Jt  at sp+0   (288 bytes: 12×3 doubles)
    addi    s5, sp, 288         # PJt at sp+288 (288 bytes: 12×3 doubles)
    addi    s6, sp, 576         # ej  at sp+576 (24 bytes)
    addi    s7, sp, 600         # Sinv_col at sp+600 (24 bytes)

    # Jt = Jᵀ  (J is 3×12, Jt is 12×3)
    mv      a0, s1
    mv      a1, s4
    li      a2, 3
    li      a3, 12
    call    mat_transpose

    # PJt = P * Jt  (12×12 * 12×3 = 12×3)
    mv      a0, s0
    mv      a1, s4
    mv      a2, s5
    li      a3, 12
    li      a4, 12
    li      a5, 3
    call    mat_mul

    # for j=0,1,2: K[:,j] = PJt * S⁻¹[:,j]
    li      s8, 0               # j
cg_loop:
    li      t0, 3
    bge     s8, t0, cg_done

    # zero ej
    fmv.d.x ft0, zero
    fsd     ft0,  0(s6)
    fsd     ft0,  8(s6)
    fsd     ft0, 16(s6)

    # ej[j] = 1.0
    li      t0, 1
    fcvt.d.w ft1, t0
    slli    t1, s8, 3
    add     t1, s6, t1
    fsd     ft1, 0(t1)

    # Sinv_col = S⁻¹ * ej  via chol_solve
    mv      a0, s2
    mv      a1, s6
    mv      a2, s7
    li      a3, 3
    call    chol_solve

    # K[:,j] = PJt * Sinv_col  (12×3 * 3×1 = 12×1)
    li      t2, 0               # row i
cg_col:
    li      t0, 12
    bge     t2, t0, cg_col_done

    fmv.d.x ft0, zero           # dot product accumulator
    li      t3, 0               # inner k
cg_dot:
    li      t0, 3
    bge     t3, t0, cg_dot_done

    # PJt[i*3+k]
    li      t0, 3
    mul     t4, t2, t0
    add     t4, t4, t3
    slli    t4, t4, 3
    add     t4, s5, t4
    fld     ft1, 0(t4)

    # Sinv_col[k]
    slli    t4, t3, 3
    add     t4, s7, t4
    fld     ft2, 0(t4)

    fmadd.d ft0, ft1, ft2, ft0

    addi    t3, t3, 1
    j       cg_dot
cg_dot_done:

    # K[i*3+j] = dot product result
    li      t0, 3
    mul     t4, t2, t0
    add     t4, t4, s8
    slli    t4, t4, 3
    add     t4, s3, t4
    fsd     ft0, 0(t4)

    addi    t2, t2, 1
    j       cg_col
cg_col_done:
    addi    s8, s8, 1
    j       cg_loop

cg_done:
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
# 10. manual_atan (INTERNAL — not called from C)
#     Approximates atan(x) for x in [-1, 1]
#     Formula: atan(x) ≈ x * (π/4 + 0.273*(1 - |x|))
#              (Remez-style minimax on [-1,1], max err ~0.0038 rad)
#
#     Input : fa0 = x  (must satisfy |x| <= 1)
#     Output: fa0 = atan(x)
#     Clobbers: ft0–ft5
#     Does NOT touch integer regs — safe to call mid-computation
# =============================================================================
manual_atan:
    # constants
    # PI/4 = 0.7853981633974483
    # C    = 0.273  (approximation coefficient)

    # load PI/4 into ft1
    lui     t0, %hi(.Lpi4)
    fld     ft1, %lo(.Lpi4)(t0)

    # load C=0.273 into ft2
    lui     t0, %hi(.Lc273)
    fld     ft2, %lo(.Lc273)(t0)

    # ax = |x| = fabs(fa0)
    fabs.d  ft3, fa0            # ft3 = |x|

    # term1 = C * ax  →  ft4 = 0.273 * |x|
    fmul.d  ft4, ft2, ft3

    # term2 = PI/4 + (C - term1)  =  PI/4 + C*(1 - |x|)
    # = PI/4 + C - C*|x|
    lui     t0, %hi(.Lc273)
    fld     ft5, %lo(.Lc273)(t0)   # ft5 = C again
    fsub.d  ft5, ft5, ft4           # ft5 = C - C*|x| = C*(1-|x|)
    fadd.d  ft1, ft1, ft5           # ft1 = PI/4 + C*(1-|x|)

    # result = x * ft1
    fmul.d  fa0, fa0, ft1

    ret

# =============================================================================
# 11. manual_atan2 (INTERNAL — not called from C directly)
#     Full quadrant atan2(y, x) using manual_atan
#     Mirrors the M2 C++ logic exactly.
#
#     Input : fa0 = y,  fa1 = x
#     Output: fa0 = atan2(y, x)
#     Uses   : ft0–ft7, t0–t2
#     Saves  : ra  (because it calls manual_atan)
# =============================================================================
manual_atan2:
    addi    sp, sp, -32
    sd      ra,  24(sp)
    fsd     fs0, 16(sp)         # save y
    fsd     fs1,  8(sp)         # save x

    fmv.d   fs0, fa0            # fs0 = y
    fmv.d   fs1, fa1            # fs1 = x

    # load constants
    lui     t0, %hi(.Lpi)
    fld     ft6, %lo(.Lpi)(t0)  # ft6 = PI
    lui     t0, %hi(.Lpi2)
    fld     ft7, %lo(.Lpi2)(t0) # ft7 = PI/2
    fmv.d.x ft5, zero           # ft5 = 0.0

    # if x==0 and y==0: return 0
    feq.d   t0, fs1, ft5        # x == 0?
    feq.d   t1, fs0, ft5        # y == 0?
    and     t0, t0, t1
    beqz    t0, atan2_not_both_zero
    fmv.d   fa0, ft5            # return 0.0
    j       atan2_done

atan2_not_both_zero:
    # choose branch: |x| >= |y| or |y| > |x|
    fabs.d  ft0, fs1            # |x|
    fabs.d  ft1, fs0            # |y|
    fle.d   t0, ft1, ft0        # |y| <= |x|  →  use atan(y/x) branch
    beqz    t0, atan2_y_branch

    # ── branch: |x| >= |y|  →  atan(y/x) ──────────────────────────────
    # check x==0 to avoid div by zero (add tiny EPS)
    lui     t0, %hi(.Leps)
    fld     ft2, %lo(.Leps)(t0) # ft2 = EPS = 1e-9
    feq.d   t0, fs1, ft5
    beqz    t0, atan2_x_nonzero
    fadd.d  fs1, fs1, ft2       # x += EPS
atan2_x_nonzero:
    fdiv.d  fa0, fs0, fs1       # fa0 = y/x
    call    manual_atan          # fa0 = atan(y/x)
    # if x < 0: add ±PI
    flt.d   t0, fs1, ft5        # x < 0?
    beqz    t0, atan2_x_branch_done
    flt.d   t1, fs0, ft5        # y < 0?
    beqz    t1, atan2_add_pi
    fsub.d  fa0, fa0, ft6       # result -= PI
    j       atan2_x_branch_done
atan2_add_pi:
    fadd.d  fa0, fa0, ft6       # result += PI
atan2_x_branch_done:
    j       atan2_done

    # ── branch: |y| > |x|  →  PI/2 - atan(x/y) ────────────────────────
atan2_y_branch:
    lui     t0, %hi(.Leps)
    fld     ft2, %lo(.Leps)(t0)
    feq.d   t0, fs0, ft5
    beqz    t0, atan2_y_nonzero
    fadd.d  fs0, fs0, ft2       # y += EPS
atan2_y_nonzero:
    fdiv.d  fa0, fs1, fs0       # fa0 = x/y
    call    manual_atan          # fa0 = atan(x/y)
    fsub.d  fa0, ft7, fa0       # fa0 = PI/2 - atan(x/y)
    flt.d   t0, fs0, ft5        # y < 0?
    beqz    t0, atan2_done
    fsub.d  fa0, fa0, ft6       # result -= PI

atan2_done:
    fld     fs0, 16(sp)
    fld     fs1,  8(sp)
    ld      ra,  24(sp)
    addi    sp, sp, 32
    ret


# =============================================================================
# 12. h_spherical(double* x, double* z_sph)
#     Converts Cartesian state x (12-dim) to spherical measurement (3-dim)
#       z_sph[0] = r     = sqrt(px² + py² + pz²)
#       z_sph[1] = theta = atan2(py, px)
#       z_sph[2] = phi   = atan2(pz, sqrt(px²+py²))
#
#     State layout: x[0]=px, x[4]=py, x[8]=pz  (stride 4 because of
#     the 12-state vector: px vx ax jx | py vy ay jy | pz vz az jz)
#
#     Args:
#       a0 = double* x      (12×1 state)
#       a1 = double* z_sph  (3×1 output)
# =============================================================================
    .global h_spherical
h_spherical:
    addi    sp, sp, -48
    sd      ra,  40(sp)
    sd      s0,  32(sp)
    sd      s1,  24(sp)
    fsd     fs0, 16(sp)
    fsd     fs1,  8(sp)
    fsd     fs2,  0(sp)

    mv      s0, a0              # x
    mv      s1, a1              # z_sph

    # load px = x[0],  py = x[4*8=32],  pz = x[8*8=64]
    fld     fs0,  0(s0)         # fs0 = px  (x[0])
    fld     fs1, 32(s0)         # fs1 = py  (x[4])
    fld     fs2, 64(s0)         # fs2 = pz  (x[8])

    # load EPS
    lui     t0, %hi(.Leps)
    fld     ft6, %lo(.Leps)(t0) # ft6 = 1e-9

    # rho = sqrt(px²+py²) + EPS
    fmul.d  ft0, fs0, fs0       # px²
    fmul.d  ft1, fs1, fs1       # py²
    fadd.d  ft0, ft0, ft1       # px²+py²
    fsqrt.d ft0, ft0            # sqrt(px²+py²)
    fadd.d  ft0, ft0, ft6       # rho = sqrt(px²+py²) + EPS

    # r = sqrt(px²+py²+pz²) + EPS
    fmul.d  ft2, fs2, fs2       # pz²
    fadd.d  ft1, ft0, ft2       # rho² approx — wrong, need px²+py²+pz²
    # redo: r² = px²+py²+pz²
    fmul.d  ft3, fs0, fs0
    fmul.d  ft4, fs1, fs1
    fmul.d  ft5, fs2, fs2
    fadd.d  ft3, ft3, ft4
    fadd.d  ft3, ft3, ft5       # ft3 = px²+py²+pz²
    fsqrt.d ft3, ft3            # ft3 = r (before EPS)
    fadd.d  ft3, ft3, ft6       # ft3 = r + EPS

    # z_sph[0] = r
    fsd     ft3, 0(s1)

    # z_sph[1] = atan2(py, px)
    fmv.d   fa0, fs1            # y = py
    fmv.d   fa1, fs0            # x = px
    call    manual_atan2
    fsd     fa0, 8(s1)          # z_sph[1] = theta

    # z_sph[2] = atan2(pz, rho)
    fmv.d   fa0, fs2            # y = pz
    fmv.d   fa1, ft0            # x = rho  — ft0 still valid? Check below
    # ft0 was rho — but manual_atan2 may clobber ft0
    # so reload rho from saved values
    fmul.d  ft0, fs0, fs0
    fmul.d  ft1, fs1, fs1
    fadd.d  ft0, ft0, ft1
    fsqrt.d ft0, ft0
    fadd.d  ft0, ft0, ft6       # rho again (safe recompute)
    fmv.d   fa1, ft0            # x = rho
    call    manual_atan2
    fsd     fa0, 16(s1)         # z_sph[2] = phi

    fld     fs0, 16(sp)
    fld     fs1,  8(sp)
    fld     fs2,  0(sp)
    ld      ra,  40(sp)
    ld      s0,  32(sp)
    ld      s1,  24(sp)
    addi    sp, sp, 48
    ret


# =============================================================================
# 13. compute_jacobian(double* x_pred, double* J)
#     Computes the 3×12 Jacobian dh/dx at state x_pred.
#
#     J[0][0] =  px/r             J[0][4] =  py/r          J[0][8]  = pz/r
#     J[1][0] = -py/rho²          J[1][4] =  px/rho²       J[1][8]  = 0
#     J[2][0] = -px*pz/(r²*rho)   J[2][4] = -py*pz/(r²*rho) J[2][8] = rho/r²
#     All other entries = 0
#
#     Args:
#       a0 = double* x_pred  (12×1)
#       a1 = double* J       (3×12 output, row-major)
# =============================================================================
    .global compute_jacobian
compute_jacobian:
    addi    sp, sp, -32
    sd      ra,  24(sp)
    sd      s0,  16(sp)
    sd      s1,   8(sp)

    mv      s0, a0              # x_pred
    mv      s1, a1              # J

    # zero J first (3*12 = 36 doubles)
    mv      a0, s1
    li      a1, 3
    li      a2, 12
    call    mat_zero

    # load px, py, pz from state (indices 0, 4, 8 → bytes 0, 32, 64)
    fld     ft0,  0(s0)         # ft0 = px
    fld     ft1, 32(s0)         # ft1 = py
    fld     ft2, 64(s0)         # ft2 = pz

    # load EPS
    lui     t0, %hi(.Leps)
    fld     ft6, %lo(.Leps)(t0)

    # rho² = px²+py²  +EPS
    fmul.d  ft3, ft0, ft0       # px²
    fmul.d  ft4, ft1, ft1       # py²
    fadd.d  ft3, ft3, ft4       # rho² (no EPS yet)
    fadd.d  ft3, ft3, ft6       # rho² + EPS  (ft3 = rho²)

    # rho = sqrt(rho²)
    fsqrt.d ft4, ft3            # ft4 = rho

    # r² = px²+py²+pz²  +EPS
    fmul.d  ft5, ft2, ft2       # pz²
    fadd.d  ft5, ft5, ft3       # r² = rho²+pz² (+EPS already in rho²)

    # r = sqrt(r²)
    fsqrt.d ft7, ft5            # ft7 = r

    # ── Row 0: dr/d(px,py,pz) ──────────────────────────────────────────
    # J[0*12+0] = px/r
    fdiv.d  fa0, ft0, ft7
    fsd     fa0, 0(s1)          # J[0][0]

    # J[0*12+4] = py/r  → byte offset = (0*12+4)*8 = 32
    fdiv.d  fa0, ft1, ft7
    fsd     fa0, 32(s1)         # J[0][4]

    # J[0*12+8] = pz/r  → byte offset = (0*12+8)*8 = 64
    fdiv.d  fa0, ft2, ft7
    fsd     fa0, 64(s1)         # J[0][8]

    # ── Row 1: dtheta/d(px,py,pz)  (theta = atan2(py,px)) ─────────────
    # J[1*12+0] = -py/rho²  → byte offset = (1*12+0)*8 = 96
    fdiv.d  fa0, ft1, ft3       # py/rho²
    fneg.d  fa0, fa0            # -py/rho²
    fsd     fa0, 96(s1)         # J[1][0]

    # J[1*12+4] = px/rho²  → byte offset = (1*12+4)*8 = 128
    fdiv.d  fa0, ft0, ft3       # px/rho²
    fsd     fa0, 128(s1)        # J[1][4]

    # J[1*12+8] = 0  (already zeroed)

    # ── Row 2: dphi/d(px,py,pz)  (phi = atan2(pz,rho)) ────────────────
    # denominator = r²*rho
    fmul.d  fa1, ft5, ft4       # r²*rho  (ft5=r², ft4=rho)

    # J[2*12+0] = -(px*pz)/(r²*rho)  → byte offset = (2*12+0)*8 = 192
    fmul.d  fa0, ft0, ft2       # px*pz
    fdiv.d  fa0, fa0, fa1       # px*pz/(r²*rho)
    fneg.d  fa0, fa0
    fsd     fa0, 192(s1)        # J[2][0]

    # J[2*12+4] = -(py*pz)/(r²*rho)  → byte offset = (2*12+4)*8 = 224
    fmul.d  fa0, ft1, ft2       # py*pz
    fdiv.d  fa0, fa0, fa1
    fneg.d  fa0, fa0
    fsd     fa0, 224(s1)        # J[2][4]

    # J[2*12+8] = rho/r²  → byte offset = (2*12+8)*8 = 256
    fdiv.d  fa0, ft4, ft5       # rho/r²
    fsd     fa0, 256(s1)        # J[2][8]

    ld      ra,  24(sp)
    ld      s0,  16(sp)
    ld      s1,   8(sp)
    addi    sp, sp, 32
    ret


# =============================================================================
# 14. ekf_predict(F, Ft, P, Q, x, x_pred, P_pred, FP, FPFt)
#     Identical math to lkf_predict:
#       x_pred = F * x
#       P_pred = F * P * Fᵀ + Q
# =============================================================================
    .global ekf_predict
ekf_predict:
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
    ld      s8, 96(sp)          # FPFt (9th arg on stack)

    # x_pred = F * x  (12×12 * 12×1)
    mv      a0, s0
    mv      a1, s4
    mv      a2, s5
    li      a3, 12
    li      a4, 12
    li      a5, 1
    call    mat_mul

    # FP = F * P  (12×12 * 12×12)
    mv      a0, s0
    mv      a1, s2
    mv      a2, s7
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # FPFt = FP * Ft  (12×12 * 12×12)
    mv      a0, s7
    mv      a1, s1
    mv      a2, s8
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # P_pred = FPFt + Q
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
# 15. ekf_update(K, J, P_pred, R, x_pred, y, x, P,
#                KJ, IKJ, tmp1, tmp2, KR, KRKt, Kt, I12)
#     Joseph-form update using Jacobian J instead of H:
#       x    = x_pred + K*y
#       IKJ  = I - K*J
#       P    = IKJ*P_pred*IKJᵀ + K*R*Kᵀ
# =============================================================================
    .global ekf_update
ekf_update:
    # Frame: 13 saved regs = 104 bytes
    addi    sp, sp, -104
    sd      ra,  96(sp)
    sd      s0,  88(sp)
    sd      s1,  80(sp)
    sd      s2,  72(sp)
    sd      s3,  64(sp)
    sd      s4,  56(sp)
    sd      s5,  48(sp)
    sd      s6,  40(sp)
    sd      s7,  32(sp)
    sd      s8,  24(sp)
    sd      s9,  16(sp)
    sd      s10,  8(sp)
    sd      s11,  0(sp)

    # save register args
    mv      s0,  a0             # K
    mv      s1,  a1             # J
    mv      s2,  a2             # P_pred
    mv      s3,  a3             # R
    mv      s4,  a4             # x_pred
    mv      s5,  a5             # y
    mv      s6,  a6             # x (output)
    mv      s7,  a7             # P (output)

    # stack args at old_sp+0..+56  =  current_sp+104+0..
    ld      s8,  104(sp)        # KJ
    ld      s9,  112(sp)        # IKJ
    ld      s10, 120(sp)        # tmp1
    ld      s11, 128(sp)        # tmp2

    # save remaining 4 stack args in a small inner frame
    addi    sp, sp, -32
    sd      zero, 24(sp)        # placeholder KR
    sd      zero, 16(sp)        # placeholder KRKt
    sd      zero,  8(sp)        # placeholder Kt
    sd      zero,  0(sp)        # placeholder I12

    # stack args KR..I12 were at old_sp+32..+56 = current_sp+32+104+32=168..
    ld      t0,  168(sp)        # KR
    ld      t1,  176(sp)        # KRKt
    ld      t2,  184(sp)        # Kt
    ld      t3,  192(sp)        # I12
    sd      t0,  24(sp)
    sd      t1,  16(sp)
    sd      t2,   8(sp)
    sd      t3,   0(sp)

    # ── x = x_pred + K*y ─────────────────────────────────────────────────
    mv      a0, s0
    mv      a1, s5
    mv      a2, s6
    li      a3, 12
    li      a4, 3
    li      a5, 1
    call    mat_mul             # x = K*y

    mv      a0, s4
    mv      a1, s6
    mv      a2, s6
    li      a3, 12
    li      a4, 1
    call    mat_add             # x = x_pred + K*y

    # ── KJ = K * J  (12x3 * 3x12 = 12x12) ───────────────────────────────
    mv      a0, s0
    mv      a1, s1
    mv      a2, s8
    li      a3, 12
    li      a4, 3
    li      a5, 12
    call    mat_mul

    # ── IKJ = I12 - KJ ───────────────────────────────────────────────────
    ld      t3,  0(sp)          # I12
    mv      a0, t3
    mv      a1, s8
    mv      a2, s9
    li      a3, 12
    li      a4, 12
    call    mat_sub

    # ── tmp1 = IKJ * P_pred ───────────────────────────────────────────────
    mv      a0, s9
    mv      a1, s2
    mv      a2, s10
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul

    # ── IKJt stored in s11 (tmp2) ─────────────────────────────────────────
    mv      a0, s9
    mv      a1, s11
    li      a2, 12
    li      a3, 12
    call    mat_transpose

    # ── joseph = tmp1 * IKJt → store in KRKt buffer ──────────────────────
    ld      t1,  16(sp)         # KRKt buffer
    mv      a0, s10
    mv      a1, s11
    mv      a2, t1
    li      a3, 12
    li      a4, 12
    li      a5, 12
    call    mat_mul             # t1 = IKJ*P_pred*IKJt

    # ── KR = K * R  (12x3 * 3x3 = 12x3) ─────────────────────────────────
    ld      t0,  24(sp)         # KR
    mv      a0, s0
    mv      a1, s3
    mv      a2, t0
    li      a3, 12
    li      a4, 3
    li      a5, 3
    call    mat_mul

    # ── Kt = Kᵀ  (12x3 → 3x12) ───────────────────────────────────────────
    ld      t2,   8(sp)         # Kt
    mv      a0, s0
    mv      a1, t2
    li      a2, 12
    li      a3, 3
    call    mat_transpose

    # ── KRKt_result = KR * Kt → store in s11 (tmp2, reuse) ───────────────
    mv      a0, t0              # KR
    mv      a1, t2              # Kt
    mv      a2, s11             # result
    li      a3, 12
    li      a4, 3
    li      a5, 12
    call    mat_mul

    # ── P = joseph + KRKt ─────────────────────────────────────────────────
    mv      a0, t1              # IKJ*P_pred*IKJt
    mv      a1, s11             # KRKt
    mv      a2, s7              # P output
    li      a3, 12
    li      a4, 12
    call    mat_add

    # ── restore ───────────────────────────────────────────────────────────
    addi    sp, sp, 32
    ld      ra,  96(sp)
    ld      s0,  88(sp)
    ld      s1,  80(sp)
    ld      s2,  72(sp)
    ld      s3,  64(sp)
    ld      s4,  56(sp)
    ld      s5,  48(sp)
    ld      s6,  40(sp)
    ld      s7,  32(sp)
    ld      s8,  24(sp)
    ld      s9,  16(sp)
    ld      s10,  8(sp)
    ld      s11,  0(sp)
    addi    sp, sp, 104
    ret

# =============================================================================
#  Read-only constants
# =============================================================================
    .section .rodata
    .align 3

.Lpi:
    .double 3.14159265358979323846

.Lpi2:
    .double 1.5707963267948966

.Lpi4:
    .double 0.7853981633974483

.Lc273:
    .double 0.273

.Leps:
    .double 1.0e-9

# end of ekf_asm.s