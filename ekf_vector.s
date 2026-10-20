# =============================================================================
# ekf_vector.s — Vectorised Extended Kalman Filter (RVV)
# FIXED VERSION: full 276-state EKF, ordered reductions, safe stack-passed pointers
# Milestone 4 — Member 2: Shehryar Tahir Awan
#
# System:  23-joint human body, 276-dim state, 69-dim spherical measurement
#          x_j = [px, vx, ax, jx, py, vy, ay, jy, pz, vz, az, jz]^T  (12 per joint)
#          F ∈ R^{276×276}, Q ∈ R^{276×276} (block-diagonal, 23 × 12×12 blocks)
#          H/J ∈ R^{69×276}  (block-diagonal, 23 × 3×12 blocks)
#          R ∈ R^{69×69}     (block-diagonal, 23 × 3×3  blocks)
#
# ABI:     RISC-V LP64D  (same as M3 scalar)
#          a0–a7   integer arguments / return value
#          fa0–fa7 FP arguments
#          s0–s11  callee-saved integer
#          fs0–fs11 callee-saved FP
#          t0–t6   caller-saved temporaries
#          ft0–ft11 caller-saved FP temporaries
#
# RVV conventions (coordinate with Member 1 / lkf_vector.s):
#   • Element width : SEW = e64  (64-bit IEEE-754 double)
#   • LMUL          : m1  (baseline) — increase to m2/m4 only where noted
#   • Tail policy   : ta  (tail-agnostic)
#   • Mask policy   : ma  (mask-agnostic)
#   • vsetvli is called at the top of every vectorised loop to (re)set VL
#   • vl (t0) is reloaded after any call that may clobber it
#   • Unit-stride loads/stores  : vle64.v / vse64.v
#   • Strided loads/stores      : vlse64.v / vsse64.v  (column access / transpose)
#   • FMA accumulation          : vfmacc.vv  (dest += src1 * src2)
#   • FMS subtraction-accum     : vfmsac.vv  (dest = src1 * src2 - dest)
#
# Pointer alignment:
#   All matrix/vector buffers are allocated via posix_memalign(64) in the
#   C driver, guaranteeing 64-byte cache-line alignment.  This allows the
#   hardware to issue full cache-line RVV unit-stride loads without
#   split-line penalties on the 276×276 row-major matrices.
#   The 8-byte alignment required by vle64.v is trivially satisfied.
#
# Functions exported (drop-in replacements for scalar ekf_asm.s):
#   vec_mat_zero        (A, r, c)
#   vec_mat_eye         (A, n)
#   vec_mat_add         (A, B, C, r, c)
#   vec_mat_sub         (A, B, C, r, c)
#   vec_mat_scale       (A, B, alpha, r, c)   [alpha in fa0]
#   vec_mat_mul         (A, B, C, r, k, c)
#   vec_mat_transpose   (A, B, r, c)
#   ekf_predict         (F, Ft, P, Q, x, x_pred, P_pred, FP, FPFt)
#   ekf_update          (K, J, P_pred, R, x_pred, y, x, P,
#                        KJ, IKJ, tmp1, tmp2, KR, KRKt, Kt, I276)
#
# Scalar helpers retained from M3 assembly (not C/C++; linked from ekf_asm.s):
#   manual_atan, manual_atan2, h_spherical, compute_jacobian, chol_solve,
#   compute_gain, mat_eye (scalar/assembly, used where vectorisation is not useful)
# =============================================================================

    .section .rodata
    .align 3
.Lzero_d:   .double 0.0
.Lone_d:    .double 1.0
.Leps:      .double 1.0e-9
.Lpi:       .double 3.14159265358979323846
.Lpi2:      .double 1.57079632679489661923
.Lpi4:      .double 0.78539816339744830962
.L0273:     .double 0.273

    .section .text

# =============================================================================
# vec_mat_zero(A, r, c)
#   a0 = A base pointer
#   a1 = rows r
#   a2 = cols c
#   Zeroes all r*c doubles using RVV unit-stride stores.
# =============================================================================
    .globl vec_mat_zero
    .type  vec_mat_zero, @function
vec_mat_zero:
    # total elements = r * c
    mul      a3, a1, a2          # a3 = r*c
    # load 0.0 for broadcast
    lui      t1, %hi(.Lzero_d)
    fld      ft0, %lo(.Lzero_d)(t1)
    # build a vector register of all-zeros via vmv.v.x (integer zero)
    # then use a store loop
.Lmzero_loop:
    beqz     a3, .Lmzero_done
    vsetvli  t0, a3, e64, m4, ta, ma   # m4 → process 4× more elements per iter
    vfmv.v.f v0, ft0                    # v0 = {0.0, 0.0, ...}
    vse64.v  v0, (a0)                   # store vl doubles
    slli     t1, t0, 3                  # byte offset = vl * 8
    add      a0, a0, t1
    sub      a3, a3, t0
    j        .Lmzero_loop
.Lmzero_done:
    ret
    .size vec_mat_zero, .-vec_mat_zero

# =============================================================================
# vec_mat_eye(A, n)
#   a0 = A base pointer
#   a1 = n  (square matrix dimension)
#   Sets A = I_n.  Calls vec_mat_zero then writes diagonal.
# =============================================================================
    .globl vec_mat_eye
    .type  vec_mat_eye, @function
vec_mat_eye:
    addi     sp, sp, -32
    sd       ra, 24(sp)
    sd       s0, 16(sp)
    sd       s1,  8(sp)
    mv       s0, a0          # save base
    mv       s1, a1          # save n
    # zero the whole matrix: vec_mat_zero(A, n, n)
    mv       a2, a1          # c = n (do NOT pass n*n here)
    call     vec_mat_zero
    # write 1.0 on diagonal, stride = (n+1)*8 bytes
    lui      t0, %hi(.Lone_d)
    fld      ft0, %lo(.Lone_d)(t0)
    addi     t1, s1, 1
    slli     t1, t1, 3        # diagonal stride bytes
    mv       t2, s1           # loop counter
    mv       a0, s0
.Leye_loop:
    beqz     t2, .Leye_done
    fsd      ft0, 0(a0)
    add      a0, a0, t1
    addi     t2, t2, -1
    j        .Leye_loop
.Leye_done:
    ld       s1,  8(sp)
    ld       s0, 16(sp)
    ld       ra, 24(sp)
    addi     sp, sp, 32
    ret
    .size vec_mat_eye, .-vec_mat_eye

# =============================================================================
# vec_mat_add(A, B, C, r, c)
#   a0=A, a1=B, a2=C, a3=r, a4=c
#   C = A + B  elementwise, RVV unit-stride.
# =============================================================================
    .globl vec_mat_add
    .type  vec_mat_add, @function
vec_mat_add:
    mul      a3, a3, a4       # total elements
.Lmadd_loop:
    beqz     a3, .Lmadd_done
    vsetvli  t0, a3, e64, m2, ta, ma
    vle64.v  v0, (a0)
    vle64.v  v2, (a1)
    vfadd.vv v4, v0, v2
    vse64.v  v4, (a2)
    slli     t1, t0, 3
    add      a0, a0, t1
    add      a1, a1, t1
    add      a2, a2, t1
    sub      a3, a3, t0
    j        .Lmadd_loop
.Lmadd_done:
    ret
    .size vec_mat_add, .-vec_mat_add

# =============================================================================
# vec_mat_sub(A, B, C, r, c)
#   a0=A, a1=B, a2=C, a3=r, a4=c
#   C = A - B  elementwise, RVV unit-stride.
# =============================================================================
    .globl vec_mat_sub
    .type  vec_mat_sub, @function
vec_mat_sub:
    mul      a3, a3, a4
.Lmsub_loop:
    beqz     a3, .Lmsub_done
    vsetvli  t0, a3, e64, m2, ta, ma
    vle64.v  v0, (a0)
    vle64.v  v2, (a1)
    vfsub.vv v4, v0, v2
    vse64.v  v4, (a2)
    slli     t1, t0, 3
    add      a0, a0, t1
    add      a1, a1, t1
    add      a2, a2, t1
    sub      a3, a3, t0
    j        .Lmsub_loop
.Lmsub_done:
    ret
    .size vec_mat_sub, .-vec_mat_sub

# =============================================================================
# vec_mat_scale(A, B, r, c)   — scalar alpha arrives in fa0
#   a0=A (input), a1=B (output), a2=r, a3=c
#   B = alpha * A  elementwise.
# =============================================================================
    .globl vec_mat_scale
    .type  vec_mat_scale, @function
vec_mat_scale:
    mul      a2, a2, a3       # total elements
.Lmscale_loop:
    beqz     a2, .Lmscale_done
    vsetvli  t0, a2, e64, m2, ta, ma
    vle64.v  v0, (a0)
    vfmul.vf v2, v0, fa0      # multiply every element by scalar fa0
    vse64.v  v2, (a1)
    slli     t1, t0, 3
    add      a0, a0, t1
    add      a1, a1, t1
    sub      a2, a2, t0
    j        .Lmscale_loop
.Lmscale_done:
    ret
    .size vec_mat_scale, .-vec_mat_scale

# =============================================================================
# vec_mat_mul(A, B, C, r, k, c)
#   a0=A(r×k), a1=B(k×c), a2=C(r×c), a3=r, a4=k, a5=c
#
#   C = A · B   (row-major storage throughout)
#
#   Strategy:
#     Outer loops: i over rows of A, j over cols of B (scalar index).
#     Inner loop : dot product of row i of A with col j of B, vectorised
#                  over the k dimension using vfmacc.vv with strided col load.
#
#   For each (i, j):
#     acc = 0
#     for p in [0, k):
#         acc += A[i*k+p] * B[p*c+j]          (scalar version)
#
#   Vectorised inner loop replaces the p-loop:
#     vle64.v  v_rowA  ← A[i*k + 0 .. k-1]   (unit stride, one row of A)
#     vlse64.v v_colB  ← B[0*c+j, 1*c+j, ...] (stride = c*8 bytes, one col of B)
#     C[i,j] = dot(v_rowA, v_colB) via vfmacc + vfredosum
#
#   Note: vfmacc.vv is used for the partial-sum accumulation where multiple
#   vector registers of partial products are accumulated into a result vector.
#   For the final scalar reduction, vfredosum.vs gives a reproducible ordered sum into lane 0.
# =============================================================================
    .globl vec_mat_mul
    .type  vec_mat_mul, @function
vec_mat_mul:
    # Prologue — callee-saved registers
    addi     sp, sp, -80
    sd       ra, 72(sp)
    sd       s0, 64(sp)
    sd       s1, 56(sp)
    sd       s2, 48(sp)
    sd       s3, 40(sp)
    sd       s4, 32(sp)
    sd       s5, 24(sp)
    sd       s6, 16(sp)
    sd       s7,  8(sp)

    mv       s0, a0          # A base
    mv       s1, a1          # B base
    mv       s2, a2          # C base
    mv       s3, a3          # r
    mv       s4, a4          # k
    mv       s5, a5          # c

    # stride for column access in B = c * 8 bytes
    slli     s6, s5, 3       # s6 = c*8 (column stride of B in bytes)

    # constant 0.0 for vector/reduction initialisation
    fmv.d.x  ft2, zero

    # outer loop i = 0..r-1
    li       s7, 0           # i = 0
.Lmm_i_loop:
    bge      s7, s3, .Lmm_done   # i >= r → done

    # pointer to row i of A: A + i*k*8
    mul      t0, s7, s4
    slli     t0, t0, 3
    add      t3, s0, t0       # t3 = &A[i][0]

    # pointer to row i of C: C + i*c*8
    mul      t0, s7, s5
    slli     t0, t0, 3
    add      t4, s2, t0       # t4 = &C[i][0]

    # middle loop j = 0..c-1
    li       t5, 0            # j = 0
.Lmm_j_loop:
    bge      t5, s5, .Lmm_j_done

    # pointer to col j of B: B + j*8
    slli     t0, t5, 3
    add      t6, s1, t0       # t6 = &B[0][j]

    # ---- vectorised dot product: sum_p A[i][p] * B[p][j] ----
    # IMPORTANT FIX:
    #   The old version kept a running vector accumulator across chunks and
    #   then reduced it once. With tail-agnostic policy, inactive lanes in the
    #   last chunk may become undefined. This version reduces each active chunk
    #   immediately with ordered reduction, then accumulates the scalar chunk
    #   sums in ft0 in p-order. This is safer for the 1e-9 verification target.
    mv       a0, t3           # row i of A (unit stride)
    mv       a1, t6           # col j of B (strided by s6)
    mv       a2, s4           # remaining length k
    fmv.d.x  ft0, zero        # scalar accumulator

.Lmm_dot_loop:
    beqz     a2, .Lmm_dot_done
    vsetvli  t0, a2, e64, m1, ta, ma   # t0 = vl ≤ remaining

    vle64.v  v0, (a0)                   # A chunk, unit stride
    vlse64.v v2, (a1), s6               # B column chunk, stride c*8

    # v4 = v0 * v2 using FMA form required by the milestone
    vfmv.v.f v4, ft2                    # v4 = 0.0
    vfmacc.vv v4, v0, v2                # v4 += v0 * v2

    # ordered horizontal reduction of the active chunk
    vfmv.s.f v6, ft2                    # reduction initial value = 0.0
    vfredosum.vs v6, v4, v6             # v6[0] = ordered sum(v4)
    vfmv.f.s ft1, v6                    # ft1 = chunk sum
    fadd.d   ft0, ft0, ft1              # scalar accumulator += chunk sum

    slli     t1, t0, 3
    add      a0, a0, t1                 # advance A pointer by vl*8
    mul      t1, t0, s6
    add      a1, a1, t1                 # advance B column pointer by vl*c*8
    sub      a2, a2, t0
    j        .Lmm_dot_loop

.Lmm_dot_done:
    # store scalar accumulator to C[i][j]
    slli     t0, t5, 3
    add      t0, t4, t0
    fsd      ft0, 0(t0)

    addi     t5, t5, 1        # j++
    j        .Lmm_j_loop

.Lmm_j_done:
    addi     s7, s7, 1        # i++
    j        .Lmm_i_loop

.Lmm_done:
    ld       s7,  8(sp)
    ld       s6, 16(sp)
    ld       s5, 24(sp)
    ld       s4, 32(sp)
    ld       s3, 40(sp)
    ld       s2, 48(sp)
    ld       s1, 56(sp)
    ld       s0, 64(sp)
    ld       ra, 72(sp)
    addi     sp, sp, 80
    ret
    .size vec_mat_mul, .-vec_mat_mul

# =============================================================================
# vec_mat_transpose(A, B, r, c)
#   a0=A(r×c), a1=B(c×r), a2=r, a3=c
#
#   B = A^T
#
#   Vectorised strategy:
#     For each row i of A (length c), use vsse64.v with stride r*8 to
#     scatter row i of A into column i of B.
#     This is more efficient than scalar element-by-element writes because
#     the read of row i is a unit-stride vle64.v, batching c reads at once.
# =============================================================================
    .globl vec_mat_transpose
    .type  vec_mat_transpose, @function
vec_mat_transpose:
    addi     sp, sp, -48
    sd       ra, 40(sp)
    sd       s0, 32(sp)
    sd       s1, 24(sp)
    sd       s2, 16(sp)
    sd       s3,  8(sp)

    mv       s0, a0          # A
    mv       s1, a1          # B
    mv       s2, a2          # r
    mv       s3, a3          # c

    # stride for scattered write into B = r*8 bytes (each col of A spans r rows in B)
    slli     t3, s2, 3       # t3 = r*8

    li       t4, 0           # i = 0
.Lmtrans_loop:
    bge      t4, s2, .Lmtrans_done

    # src: row i of A  → A + i*c*8  (unit stride, length c)
    mul      t0, t4, s3
    slli     t0, t0, 3
    add      a0, s0, t0      # &A[i][0]

    # dst: col i of B  → B + i*8    (stride r*8, length c)
    slli     t1, t4, 3
    add      a1, s1, t1      # &B[0][i]

    mv       a2, s3          # remaining = c
.Lmtrans_row_loop:
    beqz     a2, .Lmtrans_row_done
    vsetvli  t0, a2, e64, m1, ta, ma
    vle64.v  v0, (a0)            # load vl elements from row i of A
    vsse64.v v0, (a1), t3        # scatter into col i of B with stride r*8
    slli     t1, t0, 3
    add      a0, a0, t1
    mul      t1, t0, t3
    add      a1, a1, t1
    sub      a2, a2, t0
    j        .Lmtrans_row_loop
.Lmtrans_row_done:
    addi     t4, t4, 1
    j        .Lmtrans_loop
.Lmtrans_done:
    ld       s3,  8(sp)
    ld       s2, 16(sp)
    ld       s1, 24(sp)
    ld       s0, 32(sp)
    ld       ra, 40(sp)
    addi     sp, sp, 48
    ret
    .size vec_mat_transpose, .-vec_mat_transpose

# =============================================================================
# ekf_predict(F, Ft, P, Q, x, x_pred, P_pred, FP, FPFt)
#   a0=F(276×276), a1=Ft(276×276), a2=P(276×276), a3=Q(276×276),
#   a4=x(276×1),   a5=x_pred(276×1), a6=P_pred(276×276), a7=FP(276×276)
#   9th arg FPFt(276×276) passed on stack by caller → retrieved with ld
#
#   Computes:
#     x_pred = F · x                    (276×276 · 276×1 → 276×1)
#     FP     = F · P                    (276×276 · 276×276)
#     FPFt   = FP · Ft                  (276×276 · 276×276)
#     P_pred = FPFt + Q                 (elementwise add)
#
#   All heavy lifting delegated to vec_mat_mul / vec_mat_add defined above.
#   This mirrors the M3 scalar ekf_predict but calls vectorised kernels.
# =============================================================================
    .globl ekf_predict
    .type  ekf_predict, @function
ekf_predict:
    addi     sp, sp, -96
    sd       ra, 88(sp)
    sd       s0, 80(sp)
    sd       s1, 72(sp)
    sd       s2, 64(sp)
    sd       s3, 56(sp)
    sd       s4, 48(sp)
    sd       s5, 40(sp)
    sd       s6, 32(sp)
    sd       s7, 24(sp)
    sd       s8, 16(sp)

    mv       s0, a0          # F
    mv       s1, a1          # Ft
    mv       s2, a2          # P
    mv       s3, a3          # Q
    mv       s4, a4          # x
    mv       s5, a5          # x_pred
    mv       s6, a6          # P_pred
    mv       s7, a7          # FP
    # 9th argument: FPFt passed on caller's stack above our frame
    ld       s8, 96(sp)      # FPFt  (96 = our frame size)

    # ---- x_pred = F * x  (276×276 · 276×1 → 276×1) ----
    mv       a0, s0          # F
    mv       a1, s4          # x
    mv       a2, s5          # x_pred (output)
    li       a3, 276         # r
    li       a4, 276         # k
    li       a5, 1           # c=1 (column vector)
    call     vec_mat_mul

    # ---- FP = F * P  (276×276 · 276×276) ----
    mv       a0, s0          # F
    mv       a1, s2          # P
    mv       a2, s7          # FP (scratch)
    li       a3, 276
    li       a4, 276
    li       a5, 276
    call     vec_mat_mul

    # ---- FPFt = FP * Ft  (276×276 · 276×276) ----
    mv       a0, s7          # FP
    mv       a1, s1          # Ft
    mv       a2, s8          # FPFt (scratch, 9th arg)
    li       a3, 276
    li       a4, 276
    li       a5, 276
    call     vec_mat_mul

    # ---- P_pred = FPFt + Q ----
    mv       a0, s8          # FPFt
    mv       a1, s3          # Q
    mv       a2, s6          # P_pred (output)
    li       a3, 276
    li       a4, 276
    call     vec_mat_add

    ld       s8, 16(sp)
    ld       s7, 24(sp)
    ld       s6, 32(sp)
    ld       s5, 40(sp)
    ld       s4, 48(sp)
    ld       s3, 56(sp)
    ld       s2, 64(sp)
    ld       s1, 72(sp)
    ld       s0, 80(sp)
    ld       ra, 88(sp)
    addi     sp, sp, 96
    ret
    .size ekf_predict, .-ekf_predict

# =============================================================================
# ekf_update(K, J, P_pred, R, x_pred, y, x, P,
#            KJ, IKJ, tmp1, tmp2, KR, KRKt, Kt, I276)
#
#   Register args (a0–a7):
#     a0=K(276×69), a1=J(69×276), a2=P_pred(276×276), a3=R(69×69)
#     a4=x_pred(276×1), a5=y(69×1), a6=x(276×1 out), a7=P(276×276 out)
#
#   Stack args (8 pointers pushed by caller, at sp+0 .. sp+56 before our frame):
#     KJ(276×276), IKJ(276×276), tmp1(276×276), tmp2(276×276),
#     KR(276×69),  KRKt(276×276), Kt(69×276),   I276(276×276)
#
#   Implements Joseph-form update (guarantees P stays PSD over 3040 steps):
#     x    = x_pred + K·y
#     KJ   = K·J
#     IKJ  = I_{276} − KJ
#     P    = IKJ·P_pred·IKJ^T + K·R·K^T
#
#   The Jacobian J replaces the constant H from LKF — identical structure.
#   vfmacc.vv is used throughout the vectorised matrix multiplications via
#   the vec_mat_mul helper above.
# =============================================================================
    .globl ekf_update
    .type  ekf_update, @function
ekf_update:
    # Frame layout: 176 bytes
    #   0(sp)   = KR pointer     local spill because t-registers are caller-saved
    #   8(sp)   = KRKt pointer
    #   16(sp)  = Kt pointer
    #   24(sp)  = I276 pointer
    #   72..168 = saved ra/s-registers
    addi     sp, sp, -176
    sd       ra,  168(sp)
    sd       s0,  160(sp)
    sd       s1,  152(sp)
    sd       s2,  144(sp)
    sd       s3,  136(sp)
    sd       s4,  128(sp)
    sd       s5,  120(sp)
    sd       s6,  112(sp)
    sd       s7,  104(sp)
    sd       s8,   96(sp)
    sd       s9,   88(sp)
    sd       s10,  80(sp)
    sd       s11,  72(sp)

    # Save register args
    mv       s0,  a0         # K  (276×69)
    mv       s1,  a1         # J  (69×276)
    mv       s2,  a2         # P_pred
    mv       s3,  a3         # R  (69×69)
    mv       s4,  a4         # x_pred
    mv       s5,  a5         # y  (69×1 innovation)
    mv       s6,  a6         # x  output
    mv       s7,  a7         # P  output

    # Stack-passed args were at old sp+0..56, so after this frame they are at
    # current sp+176..232.
    ld       s8,  176(sp)    # KJ      (276×276)
    ld       s9,  184(sp)    # IKJ     (276×276)
    ld       s10, 192(sp)    # tmp1    (276×276)
    ld       s11, 200(sp)    # tmp2    (276×276), used for IKJ^T

    # IMPORTANT FIX:
    #   The old file loaded KR/KRKt/Kt/I276 into t3–t6 once and used them after
    #   calls. That is unsafe because t-registers are caller-saved and all
    #   vec_mat_* callees freely clobber them. Store those pointers in local
    #   stack slots and reload immediately before use.
    ld       t0,  208(sp)    # KR      (276×69)
    sd       t0,  0(sp)
    ld       t0,  216(sp)    # KRKt    (276×276)
    sd       t0,  8(sp)
    ld       t0,  224(sp)    # Kt      (69×276)
    sd       t0, 16(sp)
    ld       t0,  232(sp)    # I276    (276×276)
    sd       t0, 24(sp)

    # ---- x = x_pred + K·y  (K:276×69, y:69×1 → 276×1) ----
    mv       a0, s0           # K
    mv       a1, s5           # y
    mv       a2, s6           # x temp = K·y
    li       a3, 276
    li       a4, 69
    li       a5, 1
    call     vec_mat_mul

    mv       a0, s4           # x_pred
    mv       a1, s6           # K·y
    mv       a2, s6           # x output
    li       a3, 276
    li       a4, 1
    call     vec_mat_add

    # ---- KJ = K·J  (276×69 · 69×276 → 276×276) ----
    mv       a0, s0
    mv       a1, s1
    mv       a2, s8
    li       a3, 276
    li       a4, 69
    li       a5, 276
    call     vec_mat_mul

    # ---- IKJ = I276 − KJ ----
    ld       t0, 24(sp)       # I276
    mv       a0, t0
    mv       a1, s8
    mv       a2, s9
    li       a3, 276
    li       a4, 276
    call     vec_mat_sub

    # ---- Joseph term: IKJ · P_pred · IKJ^T ----
    # tmp1 = IKJ · P_pred
    mv       a0, s9
    mv       a1, s2
    mv       a2, s10
    li       a3, 276
    li       a4, 276
    li       a5, 276
    call     vec_mat_mul

    # tmp2 = IKJ^T
    mv       a0, s9
    mv       a1, s11
    li       a2, 276
    li       a3, 276
    call     vec_mat_transpose

    # KRKt buffer temporarily stores Joseph term = tmp1 · IKJ^T
    ld       t0, 8(sp)        # KRKt
    mv       a0, s10
    mv       a1, s11
    mv       a2, t0
    li       a3, 276
    li       a4, 276
    li       a5, 276
    call     vec_mat_mul

    # ---- Noise term: K · R · K^T ----
    # KR = K · R
    ld       t0, 0(sp)        # KR
    mv       a0, s0
    mv       a1, s3
    mv       a2, t0
    li       a3, 276
    li       a4, 69
    li       a5, 69
    call     vec_mat_mul

    # Kt = K^T
    ld       t1, 16(sp)       # Kt
    mv       a0, s0
    mv       a1, t1
    li       a2, 276
    li       a3, 69
    call     vec_mat_transpose

    # tmp1 = KR · Kt = K·R·K^T
    ld       t0, 0(sp)        # KR
    ld       t1, 16(sp)       # Kt
    mv       a0, t0
    mv       a1, t1
    mv       a2, s10
    li       a3, 276
    li       a4, 69
    li       a5, 276
    call     vec_mat_mul

    # ---- P = Joseph_term + Noise_term ----
    ld       t0, 8(sp)        # Joseph term stored in KRKt buffer
    mv       a0, t0
    mv       a1, s10
    mv       a2, s7
    li       a3, 276
    li       a4, 276
    call     vec_mat_add

    # Epilogue
    ld       s11,  72(sp)
    ld       s10,  80(sp)
    ld       s9,   88(sp)
    ld       s8,   96(sp)
    ld       s7,  104(sp)
    ld       s6,  112(sp)
    ld       s5,  120(sp)
    ld       s4,  128(sp)
    ld       s3,  136(sp)
    ld       s2,  144(sp)
    ld       s1,  152(sp)
    ld       s0,  160(sp)
    ld       ra,  168(sp)
    addi     sp, sp, 176
    ret
    .size ekf_update, .-ekf_update

# =============================================================================
# Scalar helpers — retained verbatim from M3 ekf_asm.s
# These are NOT redefined here; they are linked from ekf_asm.s (or copied
# below if a single-file build is required).  Listed for documentation:
#
#   manual_atan(x)            — minimax polynomial atan, |x|≤1
#   manual_atan2(y, x)        — four-quadrant atan2 via manual_atan
#   h_spherical(xj, hj)       — spherical measurement for one joint
#   compute_jacobian(xj, J)   — analytic 3×12 Jacobian dh/dx
#   chol_solve(A, b, x, n)    — Cholesky solve for 3×3 SPD system
#   compute_gain(P, J, S, K)  — K = P·J^T·S^{-1} via Cholesky columns
#   mat_eye(A, n)             — scalar diagonal-1 initialiser (I_276 setup)
#
# These functions are deliberately left scalar because:
#   • manual_atan/atan2: operate on 1-3 scalar doubles; no loop to vectorise.
#   • h_spherical:        one joint at a time; inner work is 3 scalar fdivs.
#   • compute_jacobian:   9 hardcoded scalar stores; already O(1) ops.
#   • chol_solve:         3×3 triangular solve; scalar is optimal at this size.
#   • compute_gain:       calls chol_solve 3 times; bottleneck is 3×3 Cholesky.
#
# The vectorisation speedup comes entirely from the O(276³) matrix multiply
# kernels (ekf_predict, ekf_update) which dominate runtime per timestep.
# =============================================================================

# =============================================================================
# END OF ekf_vector.s
# =============================================================================
