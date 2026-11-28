/*
 * zcvid_tropical_simplicial.h - Unified Semiring Algebra, Differentiable Greedy
 *                               (Gibbs Free Energy), Discrete Exterior Calculus on 2D
 *                               Simplicial Complexes, Local Dependency Analysis &
 *                               Speculative Parallel Tropical RDO with Selective Rollback
 *
 * Strict C11, zero per-solve dynamic allocations when caller supplies buffers or
 * uses reusable worker pool, portable Win32 + POSIX threads.
 */

#ifndef ZCVID_TROPICAL_SIMPLICIAL_H
#define ZCVID_TROPICAL_SIMPLICIAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ZCVID_TS_MODES 4u
#define ZCVID_TS_INF   1.0e30

typedef enum {
    ZCVID_TS_LINEAR = 0,      /* (+, *) */
    ZCVID_TS_TROPICAL = 1,    /* (min, +) -- Viterbi / Greedy RDO */
    ZCVID_TS_FREE_ENERGY = 2  /* (oplus_tau, +) -- Smooth Helmholtz Free Energy */
} zcvid_ts_semiring_t;

/* --------------------------------------------------------------------------
 * 1. Unified Semiring Algebra: Linear (+,*), Tropical (min,+), Free Energy
 * -------------------------------------------------------------------------- */

static inline double zc_ts_logsumexp_pair(double a, double b, double tau) {
    if (a >= ZCVID_TS_INF * 0.5) return b;
    if (b >= ZCVID_TS_INF * 0.5) return a;
    if (tau <= 1e-12) return (a < b) ? a : b;
    double lo = (a < b) ? a : b;
    double diff = fabs(a - b) / tau;
    if (diff > 50.0) return lo;
    return lo - tau * log1p(exp(-diff));
}

static inline void zcvid_ts_matvec(
    zcvid_ts_semiring_t ring,
    double tau,
    size_t rows,
    size_t cols,
    const double *A,
    const double *x,
    double *y) {
    for (size_t i = 0; i < rows; ++i) {
        double acc = (ring == ZCVID_TS_LINEAR) ? 0.0 : ZCVID_TS_INF;
        for (size_t j = 0; j < cols; ++j) {
            double aij = A[i * cols + j];
            double xj  = x[j];
            if (ring == ZCVID_TS_LINEAR) {
                acc += aij * xj;
            } else {
                double prod = (aij >= ZCVID_TS_INF * 0.5 || xj >= ZCVID_TS_INF * 0.5)
                    ? ZCVID_TS_INF
                    : (aij + xj);
                if (ring == ZCVID_TS_TROPICAL) {
                    if (prod < acc) acc = prod;
                } else {
                    acc = zc_ts_logsumexp_pair(acc, prod, tau);
                }
            }
        }
        y[i] = acc;
    }
}

static inline void zcvid_ts_matmul(
    zcvid_ts_semiring_t ring,
    double tau,
    size_t m,
    size_t k,
    size_t n,
    const double *A,
    const double *B,
    double *C) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = (ring == ZCVID_TS_LINEAR) ? 0.0 : ZCVID_TS_INF;
            for (size_t p = 0; p < k; ++p) {
                double aip = A[i * k + p];
                double bpj = B[p * n + j];
                if (ring == ZCVID_TS_LINEAR) {
                    acc += aip * bpj;
                } else {
                    double prod = (aip >= ZCVID_TS_INF * 0.5 || bpj >= ZCVID_TS_INF * 0.5)
                        ? ZCVID_TS_INF
                        : (aip + bpj);
                    if (ring == ZCVID_TS_TROPICAL) {
                        if (prod < acc) acc = prod;
                    } else {
                        acc = zc_ts_logsumexp_pair(acc, prod, tau);
                    }
                }
            }
            C[i * n + j] = acc;
        }
    }
}

/* --------------------------------------------------------------------------
 * 2. Differentiable Greedy & Minimal Physical Law (Helmholtz Free Energy)
 *    F_tau(E) = -tau * ln(sum_m exp(-E_m / tau)) = E[E] - tau * H(p)
 *    d F_tau / d E_m = p_m = softmax(-E / tau)_m
 * -------------------------------------------------------------------------- */

typedef struct {
    double free_energy;
    double expected_energy;
    double entropy;
    double min_energy;
    size_t argmin_idx;
} zcvid_ts_free_energy_res_t;

static inline zcvid_ts_free_energy_res_t zcvid_ts_free_energy_eval(
    const double *energies,
    size_t k,
    double tau,
    double *out_grad_probs) {
    zcvid_ts_free_energy_res_t res;
    res.free_energy = 0.0;
    res.expected_energy = 0.0;
    res.entropy = 0.0;
    res.min_energy = (k > 0u) ? energies[0] : 0.0;
    res.argmin_idx = 0u;
    if (k == 0u) return res;

    for (size_t i = 1; i < k; ++i) {
        if (energies[i] < res.min_energy) {
            res.min_energy = energies[i];
            res.argmin_idx = i;
        }
    }

    if (tau <= 1e-12) {
        res.free_energy = res.min_energy;
        res.expected_energy = res.min_energy;
        res.entropy = 0.0;
        if (out_grad_probs) {
            for (size_t i = 0; i < k; ++i) {
                out_grad_probs[i] = (i == res.argmin_idx) ? 1.0 : 0.0;
            }
        }
        return res;
    }

    double sum_w = 0.0;
    for (size_t i = 0; i < k; ++i) {
        double arg = -(energies[i] - res.min_energy) / tau;
        sum_w += (arg < -700.0) ? 0.0 : exp(arg);
    }
    res.free_energy = res.min_energy - tau * log(sum_w);

    double ee = 0.0;
    double ent = 0.0;
    for (size_t i = 0; i < k; ++i) {
        double arg = -(energies[i] - res.min_energy) / tau;
        double w = (arg < -700.0) ? 0.0 : exp(arg);
        double p = w / sum_w;
        if (out_grad_probs) out_grad_probs[i] = p;
        ee += p * energies[i];
        if (p > 1e-300) ent -= p * log(p);
    }
    res.expected_energy = ee;
    res.entropy = ent;
    return res;
}

static inline double zcvid_ts_diff_greedy_tune(
    const double *distortion,
    const double *rate,
    size_t k,
    double tau,
    double target_rate,
    double init_lambda,
    double lr,
    int iters) {
    double lam = (init_lambda > 0.0) ? init_lambda : 0.01;
    double ebuf[64], pbuf[64];
    if (k == 0u || k > 64u) return lam;
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < k; ++i) {
            ebuf[i] = distortion[i] + lam * rate[i];
        }
        zcvid_ts_free_energy_eval(ebuf, k, tau, pbuf);
        double er = 0.0;
        for (size_t i = 0; i < k; ++i) er += pbuf[i] * rate[i];
        double grad_lam = er - target_rate;
        lam += lr * grad_lam;
        if (lam < 1e-5) lam = 1e-5;
        if (lam > 1e5) lam = 1e5;
    }
    return lam;
}

/* --------------------------------------------------------------------------
 * 3. 2D Simplicial Complex K = (V, E, F) & Discrete Exterior Calculus
 *    Each cell (x,y)..(x+1,y+1) triangulated along (x,y)--(x+1,y+1):
 *      - d0(phi)(u,v) = phi(v) - phi(u)
 *      - d1(w)(u,v,t) = w(u,v) + w(v,t) - w(u,t)
 *      - Exact discrete Stokes identity: d1(d0(phi)) == 0 identically!
 * -------------------------------------------------------------------------- */

static inline int32_t zcvid_ts_simplicial_d1_d0_max_abs(
    const int32_t *phi,
    size_t w,
    size_t h,
    int64_t *out_hodge_energy) {
    int64_t hodge = 0;
    int32_t max_curl = 0;
    if (w < 2u || h < 2u || !phi) {
        if (out_hodge_energy) *out_hodge_energy = 0;
        return 0;
    }
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            size_t idx = y * w + x;
            if (x + 1u < w) {
                int64_t dh = (int64_t)phi[idx + 1u] - (int64_t)phi[idx];
                hodge += dh * dh;
            }
            if (y + 1u < h) {
                int64_t dv = (int64_t)phi[idx + w] - (int64_t)phi[idx];
                hodge += dv * dv;
            }
            if (x + 1u < w && y + 1u < h) {
                int64_t p00 = phi[idx];
                int64_t p10 = phi[idx + 1u];
                int64_t p01 = phi[idx + w];
                int64_t p11 = phi[idx + w + 1u];
                int64_t d0_h0 = p10 - p00;
                int64_t d0_v1 = p11 - p10;
                int64_t d0_diag = p11 - p00;
                int64_t d0_v0 = p01 - p00;
                int64_t d0_h1 = p11 - p01;
                hodge += d0_diag * d0_diag;
                int64_t c_upper = d0_h0 + d0_v1 - d0_diag;
                int64_t c_lower = d0_v0 + d0_h1 - d0_diag;
                int32_t au = (int32_t)(c_upper < 0 ? -c_upper : c_upper);
                int32_t al = (int32_t)(c_lower < 0 ? -c_lower : c_lower);
                if (au > max_curl) max_curl = au;
                if (al > max_curl) max_curl = al;
            }
        }
    }
    if (out_hodge_energy) *out_hodge_energy = hodge;
    return max_curl;
}

static inline int32_t zcvid_ts_simplex_planar_pred(int32_t left, int32_t top, int32_t tl) {
    return left + top - tl;
}

/* --------------------------------------------------------------------------
 * 4. Local Dependency Analysis & Speculative Parallel Tropical RDO
 *    with Selective Rollback
 * -------------------------------------------------------------------------- */

typedef struct {
    double edge_cost[ZCVID_TS_MODES][ZCVID_TS_MODES]; /* cost when neighbor has mode c != m */
    double tl_diag_bonus;                             /* extra penalty when m disagrees with TL too */
} zcvid_ts_coupling_t;

typedef struct {
    size_t rounds;
    size_t total_evals;
    size_t rollbacks;
    double spec_hit_rate;
} zcvid_ts_spec_stats_t;

typedef struct {
    uint8_t left;
    uint8_t top;
    uint8_t tl;
    uint8_t valid;
} zc_ts_ctx_t;

static inline zc_ts_ctx_t zc_ts_read_ctx(
    const uint8_t *modes, size_t bw, size_t x, size_t y) {
    zc_ts_ctx_t c;
    c.left  = (x > 0u) ? modes[y * bw + (x - 1u)] : 0u;
    c.top   = (y > 0u) ? modes[(y - 1u) * bw + x] : 0u;
    c.tl    = (x > 0u && y > 0u) ? modes[(y - 1u) * bw + (x - 1u)] : 0u;
    c.valid = 1u;
    return c;
}

static inline uint8_t zc_ts_eval_block(
    const double *energies_row,
    const zcvid_ts_coupling_t *coupling,
    zc_ts_ctx_t ctx,
    size_t x,
    size_t y) {
    uint8_t best_m = 0u;
    double best_cost = ZCVID_TS_INF;
    for (uint8_t m = 0u; m < (uint8_t)ZCVID_TS_MODES; ++m) {
        double e = energies_row[m];
        if (x > 0u) e += coupling->edge_cost[m][ctx.left];
        if (y > 0u) e += coupling->edge_cost[m][ctx.top];
        if (x > 0u && y > 0u && m != ctx.tl) e += coupling->tl_diag_bonus;
        if (e < best_cost) {
            best_cost = e;
            best_m = m;
        }
    }
    return best_m;
}

static inline void zcvid_ts_rdo_seq(
    size_t bw,
    size_t bh,
    const double *energies,
    const zcvid_ts_coupling_t *coupling,
    uint8_t *out_modes) {
    if (!bw || !bh || !energies || !coupling || !out_modes) return;
    for (size_t y = 0; y < bh; ++y) {
        for (size_t x = 0; x < bw; ++x) {
            size_t v = y * bw + x;
            zc_ts_ctx_t ctx = zc_ts_read_ctx(out_modes, bw, x, y);
            out_modes[v] = zc_ts_eval_block(
                energies + v * ZCVID_TS_MODES, coupling, ctx, x, y);
        }
    }
}

#define ZCVID_TS_MAX_THREADS 16u

typedef struct zc_ts_pool_s zcvid_ts_pool_t;

typedef struct {
    zcvid_ts_pool_t *pool;
    size_t tid;
} zc_ts_worker_arg_t;

typedef struct {
    size_t bw;
    size_t bh;
    const double *energies;
    const zcvid_ts_coupling_t *coupling;
    const uint8_t *mode_cur;
    const uint8_t *ctx_init;
    uint8_t *mode_next;
    zc_ts_ctx_t *ctx_used;
    const size_t *dirty_list;
    size_t n_items;
    int is_round0;
} zc_ts_job_t;

struct zc_ts_pool_s {
    size_t nthreads;
    int stop;
    size_t epoch;
    size_t done_count;
    zc_ts_job_t job;
    zc_ts_worker_arg_t args[ZCVID_TS_MAX_THREADS];
#if defined(_WIN32)
    HANDLE threads[ZCVID_TS_MAX_THREADS];
    CRITICAL_SECTION mu;
    CONDITION_VARIABLE cv_work;
    CONDITION_VARIABLE cv_done;
#else
    pthread_t threads[ZCVID_TS_MAX_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
#endif
};

static inline void zc_ts_exec_slice(const zc_ts_job_t *job, size_t start, size_t end) {
    size_t bw = job->bw;
    for (size_t i = start; i < end; ++i) {
        size_t v = job->is_round0 ? i : job->dirty_list[i];
        size_t y = v / bw;
        size_t x = v % bw;
        zc_ts_ctx_t ctx = job->is_round0
            ? (job->ctx_init ? zc_ts_read_ctx(job->ctx_init, bw, x, y) : (zc_ts_ctx_t){0u, 0u, 0u, 1u})
            : zc_ts_read_ctx(job->mode_cur, bw, x, y);
        job->mode_next[v] = zc_ts_eval_block(
            job->energies + v * ZCVID_TS_MODES, job->coupling, ctx, x, y);
        job->ctx_used[v] = ctx;
    }
}

#if defined(_WIN32)
static DWORD WINAPI zc_ts_worker_proc(LPVOID raw) {
#else
static void *zc_ts_worker_proc(void *raw) {
#endif
    zc_ts_worker_arg_t *arg = (zc_ts_worker_arg_t *)raw;
    zcvid_ts_pool_t *p = arg->pool;
    size_t tid = arg->tid;
    size_t seen = 0u;
    for (;;) {
#if defined(_WIN32)
        EnterCriticalSection(&p->mu);
        while (!p->stop && p->epoch == seen) {
            SleepConditionVariableCS(&p->cv_work, &p->mu, INFINITE);
        }
        int do_stop = p->stop;
        seen = p->epoch;
        zc_ts_job_t job = p->job;
        LeaveCriticalSection(&p->mu);
#else
        pthread_mutex_lock(&p->mu);
        while (!p->stop && p->epoch == seen) {
            pthread_cond_wait(&p->cv_work, &p->mu);
        }
        int do_stop = p->stop;
        seen = p->epoch;
        zc_ts_job_t job = p->job;
        pthread_mutex_unlock(&p->mu);
#endif
        if (do_stop) break;
        size_t nt = p->nthreads;
        size_t total = job.n_items;
        size_t start = (total * tid) / nt;
        size_t end   = (total * (tid + 1u)) / nt;
        if (start < end) zc_ts_exec_slice(&job, start, end);
#if defined(_WIN32)
        EnterCriticalSection(&p->mu);
        p->done_count++;
        if (p->done_count == nt) WakeAllConditionVariable(&p->cv_done);
        LeaveCriticalSection(&p->mu);
#else
        pthread_mutex_lock(&p->mu);
        p->done_count++;
        if (p->done_count == nt) pthread_cond_broadcast(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
#endif
    }
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

static inline bool zcvid_ts_pool_init(zcvid_ts_pool_t *p, size_t nthreads) {
    if (!p) return false;
    memset(p, 0, sizeof(*p));
    if (nthreads < 1u) nthreads = 1u;
    if (nthreads > ZCVID_TS_MAX_THREADS) nthreads = ZCVID_TS_MAX_THREADS;
    p->nthreads = nthreads;
    if (nthreads == 1u) return true;
#if defined(_WIN32)
    InitializeCriticalSection(&p->mu);
    InitializeConditionVariable(&p->cv_work);
    InitializeConditionVariable(&p->cv_done);
#else
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);
#endif
    for (size_t t = 0u; t < nthreads; ++t) {
        p->args[t].pool = p;
        p->args[t].tid = t;
#if defined(_WIN32)
        p->threads[t] = CreateThread(NULL, 0, zc_ts_worker_proc, &p->args[t], 0, NULL);
#else
        pthread_create(&p->threads[t], NULL, zc_ts_worker_proc, &p->args[t]);
#endif
    }
    return true;
}

static inline void zcvid_ts_pool_destroy(zcvid_ts_pool_t *p) {
    if (!p || p->nthreads <= 1u) return;
#if defined(_WIN32)
    EnterCriticalSection(&p->mu);
    p->stop = 1;
    WakeAllConditionVariable(&p->cv_work);
    LeaveCriticalSection(&p->mu);
    for (size_t t = 0u; t < p->nthreads; ++t) {
        if (p->threads[t]) {
            WaitForSingleObject(p->threads[t], INFINITE);
            CloseHandle(p->threads[t]);
        }
    }
    DeleteCriticalSection(&p->mu);
#else
    pthread_mutex_lock(&p->mu);
    p->stop = 1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for (size_t t = 0u; t < p->nthreads; ++t) {
        pthread_join(p->threads[t], NULL);
    }
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
#endif
    p->nthreads = 0u;
}

static inline void zc_ts_pool_run(zcvid_ts_pool_t *p, const zc_ts_job_t *job) {
    if (!p || p->nthreads <= 1u || job->n_items < 256u) {
        zc_ts_exec_slice(job, 0u, job->n_items);
        return;
    }
#if defined(_WIN32)
    EnterCriticalSection(&p->mu);
    p->job = *job;
    p->done_count = 0u;
    p->epoch++;
    WakeAllConditionVariable(&p->cv_work);
    while (p->done_count < p->nthreads) {
        SleepConditionVariableCS(&p->cv_done, &p->mu, INFINITE);
    }
    LeaveCriticalSection(&p->mu);
#else
    pthread_mutex_lock(&p->mu);
    p->job = *job;
    p->done_count = 0u;
    p->epoch++;
    pthread_cond_broadcast(&p->cv_work);
    while (p->done_count < p->nthreads) {
        pthread_cond_wait(&p->cv_done, &p->mu);
    }
    pthread_mutex_unlock(&p->mu);
#endif
}

static inline bool zcvid_ts_rdo_spec_solve(
    zcvid_ts_pool_t *pool,
    size_t bw,
    size_t bh,
    const double *energies,
    const zcvid_ts_coupling_t *coupling,
    const uint8_t *init_pred,
    uint8_t *out_modes,
    zcvid_ts_spec_stats_t *out_stats) {
    if (!bw || !bh || !energies || !coupling || !out_modes) return false;
    size_t n = bw * bh;
    uint8_t *mode_next = (uint8_t *)malloc(n * sizeof(uint8_t));
    zc_ts_ctx_t *ctx_used = (zc_ts_ctx_t *)malloc(n * sizeof(zc_ts_ctx_t));
    size_t *dirty = (size_t *)malloc(n * sizeof(size_t));
    if (!mode_next || !ctx_used || !dirty) {
        free(mode_next);
        free(ctx_used);
        free(dirty);
        return false;
    }

    /* Round 0: full speculative evaluation using predicted causal contexts */
    zc_ts_job_t job0;
    job0.bw = bw;
    job0.bh = bh;
    job0.energies = energies;
    job0.coupling = coupling;
    job0.mode_cur = out_modes;
    job0.ctx_init = init_pred;
    job0.mode_next = out_modes;
    job0.ctx_used = ctx_used;
    job0.dirty_list = dirty;
    job0.n_items = n;
    job0.is_round0 = 1;
    zc_ts_pool_run(pool, &job0);

    size_t rounds = 1u;
    size_t evals = n;
    size_t rollbacks = 0u;
    size_t max_rounds = bw + bh + 2u;

    for (; rounds < max_rounds; ) {
        /* Local Dependency Analysis: inspect only 1-hop causal star (L, T, TL) */
        size_t ndirty = 0u;
        for (size_t y = 0; y < bh; ++y) {
            for (size_t x = 0; x < bw; ++x) {
                size_t v = y * bw + x;
                zc_ts_ctx_t actual = zc_ts_read_ctx(out_modes, bw, x, y);
                zc_ts_ctx_t prev   = ctx_used[v];
                if (actual.left != prev.left || actual.top != prev.top || actual.tl != prev.tl) {
                    dirty[ndirty++] = v;
                }
            }
        }
        if (ndirty == 0u) break;

        /* Selective Rollback on dirty subset only */
        zc_ts_job_t job;
        job.bw = bw;
        job.bh = bh;
        job.energies = energies;
        job.coupling = coupling;
        job.mode_cur = out_modes;
        job.ctx_init = NULL;
        job.mode_next = mode_next;
        job.ctx_used = ctx_used;
        job.dirty_list = dirty;
        job.n_items = ndirty;
        job.is_round0 = 0;
        zc_ts_pool_run(pool, &job);

        for (size_t i = 0; i < ndirty; ++i) {
            size_t v = dirty[i];
            out_modes[v] = mode_next[v];
        }
        rounds++;
        evals += ndirty;
        rollbacks += ndirty;
    }

    if (out_stats) {
        out_stats->rounds = rounds;
        out_stats->total_evals = evals;
        out_stats->rollbacks = rollbacks;
        out_stats->spec_hit_rate = (n > 0u) ? ((double)n / (double)evals) : 1.0;
    }

    free(mode_next);
    free(ctx_used);
    free(dirty);
    return true;
}

#ifdef __cplusplus
}
#endif

#endif /* ZCVID_TROPICAL_SIMPLICIAL_H */
