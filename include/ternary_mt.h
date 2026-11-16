/**
 * @file ternary_mt.h
 * @brief Zero-alloc-per-frame multi-threaded slice engine for ternary_codec.h
 *
 * Uses a persistent worker pool with C11 atomics + OS sleep/wake (Win32 Events
 * or POSIX condvars) to encode and decode horizontal frame slices in parallel.
 */

#ifndef ZCRUDP_TERNARY_MT_H
#define ZCRUDP_TERNARY_MT_H

#include "ternary_codec.h"
#include <stdatomic.h>

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

#define ZCRUDP_TERNARY_MT_MAX_THREADS 16U

/* Spin-wait hint, portable across GCC/Clang and MSVC, x86 and others. */
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#define ZCRUDP_MT_CPU_RELAX() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define ZCRUDP_MT_CPU_RELAX() __builtin_ia32_pause()
#else
#define ZCRUDP_MT_CPU_RELAX() ((void)0)
#endif

typedef enum {
    ZCRUDP_MT_OP_NONE = 0,
    ZCRUDP_MT_OP_ENCODE_PLANE = 1,
    ZCRUDP_MT_OP_DECAY_PLANE  = 2,
    ZCRUDP_MT_OP_DECODE_PKTS  = 3,
    ZCRUDP_MT_OP_ENCODE_INTRA = 4,
    ZCRUDP_MT_OP_QUIT         = 5
} zcrudp_mt_op_t;

typedef struct {
    uint8_t *pkts_base;
    size_t  *lens_base;
    size_t   mtu_stride;
    size_t   max_pkts;
    size_t   count;
    size_t   total_bytes;
    size_t   dropped_pkts;
} zcrudp_mt_slice_sink_t;

typedef struct zcrudp_ternary_mt_pool zcrudp_ternary_mt_pool_t;

typedef struct {
    zcrudp_ternary_mt_pool_t *pool;
    uint32_t tid;
    uint8_t  pkt_scratch[1500];
    zcrudp_mt_slice_sink_t sink;
#if defined(_WIN32)
    HANDLE handle;
#else
    pthread_t handle;
#endif
} zcrudp_mt_worker_t;

struct zcrudp_ternary_mt_pool {
    uint32_t num_threads;
    zcrudp_mt_worker_t workers[ZCRUDP_TERNARY_MT_MAX_THREADS];

    atomic_uint_fast32_t gen;
    atomic_uint_fast32_t active_workers;
    atomic_int           op;

#if defined(_WIN32)
    SRWLOCK              work_lock;
    CONDITION_VARIABLE   work_cv;
#else
    pthread_mutex_t      work_lock;
    pthread_cond_t       work_cv;
#endif

    /* Shared per-dispatch job descriptor */
    const uint8_t               *src;
    uint8_t                     *ref_or_canvas;
    uint8_t                     *delta_buf;
    size_t                       plane_len;
    size_t                       total_blocks;
    size_t                       intra_begin;
    size_t                       intra_end;
    uint8_t                      plane_id;
    uint16_t                     frame_id;
    const zcrudp_ternary_config_t *cfg;

    /* For parallel packet decode */
    const uint8_t *pkt_in_base;
    const size_t  *pkt_in_lens;
    size_t         pkt_in_stride;
    size_t         pkt_in_count;
    size_t         y_sz;
};

static inline void zcrudp_mt_slice_cb(const uint8_t *data, size_t len, void *ctx) {
    zcrudp_mt_slice_sink_t *s = (zcrudp_mt_slice_sink_t *)ctx;
    if (s->count < s->max_pkts && len <= s->mtu_stride) {
        memcpy(s->pkts_base + s->count * s->mtu_stride, data, len);
        s->lens_base[s->count] = len;
        s->count++;
        s->total_bytes += len;
    } else {
        s->dropped_pkts++;
    }
}

static inline void zcrudp_mt_wake_workers(zcrudp_ternary_mt_pool_t *p) {
#if defined(_WIN32)
    AcquireSRWLockExclusive(&p->work_lock);
    ReleaseSRWLockExclusive(&p->work_lock);
    WakeAllConditionVariable(&p->work_cv);
#else
    pthread_mutex_lock(&p->work_lock);
    pthread_cond_broadcast(&p->work_cv);
    pthread_mutex_unlock(&p->work_lock);
#endif
}

static inline void zcrudp_mt_run_task(zcrudp_ternary_mt_pool_t *p, uint32_t tid) {
    int op = atomic_load_explicit(&p->op, memory_order_relaxed);
    uint32_t n = p->num_threads;

    if (op == ZCRUDP_MT_OP_ENCODE_PLANE) {
        size_t blks_per = (p->total_blocks + n - 1U) / n;
        /* Align slice boundary to 8 blocks (80 pixels) for clean AVX2 alignment */
        blks_per = ((blks_per + 7U) / 8U) * 8U;
        size_t b0 = (size_t)tid * blks_per;
        size_t b1 = b0 + blks_per;
        zcrudp_mt_worker_t *w = &p->workers[tid];
        w->sink.count = 0;
        w->sink.total_bytes = 0;
        w->sink.dropped_pkts = 0;
        if (b0 < p->total_blocks) {
            if (b1 > p->total_blocks) b1 = p->total_blocks;
            zcrudp_ternary_config_t slice_cfg = *p->cfg;
            if (slice_cfg.ec_max_blocks > 0U && n > 1U) {
                uint32_t share = (slice_cfg.ec_max_blocks + n - 1U) / n;
                slice_cfg.ec_max_blocks = (share > 0U) ? share : 1U;
            }
            zcrudp_ternary_encode_plane_range(
                p->src, p->ref_or_canvas, p->delta_buf,
                p->plane_len, b0, b1,
                p->plane_id, p->frame_id, &slice_cfg,
                w->pkt_scratch, zcrudp_mt_slice_cb, &w->sink);
        }
    } else if (op == ZCRUDP_MT_OP_DECAY_PLANE) {
        size_t chunk = (p->plane_len + n - 1U) / n;
        chunk = ((chunk + 31U) / 32U) * 32U;
        size_t i0 = (size_t)tid * chunk;
        size_t i1 = i0 + chunk;
        if (i0 < p->plane_len) {
            if (i1 > p->plane_len) i1 = p->plane_len;
            zcrudp_ternary_decay_plane(p->ref_or_canvas + i0, i1 - i0,
                                       p->cfg->gamma_q8, p->cfg->neutral_y);
        }
    } else if (op == ZCRUDP_MT_OP_DECODE_PKTS) {
        size_t cnt = p->pkt_in_count;
        size_t per = (cnt + n - 1U) / n;
        size_t s0 = (size_t)tid * per;
        size_t s1 = s0 + per;
        if (s1 > cnt) s1 = cnt;
        for (size_t idx = s0; idx < s1; idx++) {
            const uint8_t *pd = p->pkt_in_base + idx * p->pkt_in_stride;
            size_t plen = p->pkt_in_lens[idx];
            zcrudp_ternary_pkt_hdr_t hdr;
            if (plen < ZCRUDP_TERNARY_PKT_HDR_SZ ||
                !zcrudp_ternary_pkt_hdr_read_le(pd, &hdr)) continue;
            /* hdr.plane comes off the wire: it selects the destination buffer,
             * so reject planes the configured pixel format does not have. */
            uint8_t num_planes = (p->cfg->pixfmt == ZCRUDP_TERNARY_FMT_RGBA) ? 3U
                               : (p->cfg->pixfmt == ZCRUDP_TERNARY_FMT_NV12) ? 2U : 1U;
            if (hdr.plane >= num_planes) continue;
            size_t plane_off = (hdr.plane == 0) ? 0 : ((size_t)hdr.plane * p->y_sz);
            size_t plane_len = (p->cfg->pixfmt == ZCRUDP_TERNARY_FMT_NV12 && hdr.plane == 1)
                ? (p->y_sz / 2U) : p->y_sz;
            zcrudp_ternary_decode_packet(
                pd, plen,
                p->ref_or_canvas + plane_off,
                p->delta_buf ? (p->delta_buf + plane_off) : NULL,
                plane_len, p->cfg, NULL);
        }
    } else if (op == ZCRUDP_MT_OP_ENCODE_INTRA) {
        size_t span = (p->intra_end > p->intra_begin) ? (p->intra_end - p->intra_begin) : 0U;
        size_t per = (span + n - 1U) / n;
        size_t b0 = p->intra_begin + (size_t)tid * per;
        size_t b1 = b0 + per;
        if (b1 > p->intra_end) b1 = p->intra_end;
        zcrudp_mt_worker_t *w = &p->workers[tid];
        w->sink.count = 0;
        w->sink.total_bytes = 0;
        w->sink.dropped_pkts = 0;
        if (b0 < b1) {
            zcrudp_ternary_encode_intra_range(
                p->src, p->ref_or_canvas, p->delta_buf,
                b0, b1, p->plane_len,
                p->plane_id, p->frame_id, p->cfg,
                w->pkt_scratch, zcrudp_mt_slice_cb, &w->sink);
        }
    }
}

#if defined(_WIN32)
static DWORD WINAPI zcrudp_mt_worker_entry(LPVOID arg) {
#else
static void *zcrudp_mt_worker_entry(void *arg) {
#endif
    zcrudp_mt_worker_t *w = (zcrudp_mt_worker_t *)arg;
    zcrudp_ternary_mt_pool_t *p = w->pool;
    uint32_t seen_gen = 0;

    for (;;) {
        uint32_t cur_gen;
        uint32_t spins = 0;
        while ((cur_gen = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen_gen) {
            if (++spins < 256U) {
                ZCRUDP_MT_CPU_RELAX();
            } else {
#if defined(_WIN32)
                AcquireSRWLockExclusive(&p->work_lock);
                while ((cur_gen = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen_gen) {
                    SleepConditionVariableSRW(&p->work_cv, &p->work_lock, INFINITE, 0);
                }
                ReleaseSRWLockExclusive(&p->work_lock);
#else
                pthread_mutex_lock(&p->work_lock);
                while ((cur_gen = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen_gen) {
                    pthread_cond_wait(&p->work_cv, &p->work_lock);
                }
                pthread_mutex_unlock(&p->work_lock);
#endif
                break;
            }
        }
        seen_gen = cur_gen;
        if (atomic_load_explicit(&p->op, memory_order_relaxed) == ZCRUDP_MT_OP_QUIT) {
            break;
        }
        zcrudp_mt_run_task(p, w->tid);
        atomic_fetch_sub_explicit(&p->active_workers, 1U, memory_order_release);
    }

#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

static inline void zcrudp_mt_dispatch(zcrudp_ternary_mt_pool_t *p, int op) {
    atomic_store_explicit(&p->op, op, memory_order_relaxed);
    if (p->num_threads > 1U) {
        atomic_store_explicit(&p->active_workers, p->num_threads - 1U, memory_order_release);
        atomic_fetch_add_explicit(&p->gen, 1U, memory_order_release);
        zcrudp_mt_wake_workers(p);
    }
    zcrudp_mt_run_task(p, 0);
    if (p->num_threads > 1U) {
        while (atomic_load_explicit(&p->active_workers, memory_order_acquire) != 0U) {
            ZCRUDP_MT_CPU_RELAX();
        }
    }
}

/**
 * @brief Initializes a persistent worker pool (num_threads clamped to [1, 16]).
 *        Each slice sink uses caller-provided per-thread packet buffers.
 *        If the OS refuses a thread the pool shrinks to the workers that did
 *        start (p->num_threads); it always keeps the calling thread.
 */
static inline void zcrudp_ternary_mt_init(zcrudp_ternary_mt_pool_t *p,
                                          uint32_t num_threads,
                                          uint8_t *slice_pkts_buf,
                                          size_t *slice_lens_buf,
                                          size_t max_pkts_per_thread,
                                          size_t mtu_stride) {
    if (num_threads < 1U) num_threads = 1U;
    if (num_threads > ZCRUDP_TERNARY_MT_MAX_THREADS) num_threads = ZCRUDP_TERNARY_MT_MAX_THREADS;

    memset(p, 0, sizeof(*p));
    p->num_threads = num_threads;
    atomic_init(&p->gen, 0U);
    atomic_init(&p->active_workers, 0U);
    atomic_init(&p->op, ZCRUDP_MT_OP_NONE);
#if defined(_WIN32)
    InitializeSRWLock(&p->work_lock);
    InitializeConditionVariable(&p->work_cv);
#else
    pthread_mutex_init(&p->work_lock, NULL);
    pthread_cond_init(&p->work_cv, NULL);
#endif

    for (uint32_t t = 0; t < num_threads; t++) {
        p->workers[t].pool = p;
        p->workers[t].tid = t;
        p->workers[t].sink.pkts_base = slice_pkts_buf + (size_t)t * max_pkts_per_thread * mtu_stride;
        p->workers[t].sink.lens_base = slice_lens_buf + (size_t)t * max_pkts_per_thread;
        p->workers[t].sink.mtu_stride = mtu_stride;
        p->workers[t].sink.max_pkts = max_pkts_per_thread;
        if (t > 0) {
            /* A worker that never started would never acknowledge a dispatch
             * and the pool would hang: stop at the last thread that exists. */
#if defined(_WIN32)
            p->workers[t].handle = CreateThread(NULL, 0, zcrudp_mt_worker_entry,
                                                &p->workers[t], 0, NULL);
            bool started = (p->workers[t].handle != NULL);
#else
            bool started = (pthread_create(&p->workers[t].handle, NULL,
                                           zcrudp_mt_worker_entry,
                                           &p->workers[t]) == 0);
#endif
            if (!started) {
                p->num_threads = t;
                break;
            }
        }
    }
}

static inline void zcrudp_ternary_mt_destroy(zcrudp_ternary_mt_pool_t *p) {
    if (p->num_threads > 1U) {
        atomic_store_explicit(&p->op, ZCRUDP_MT_OP_QUIT, memory_order_relaxed);
        atomic_fetch_add_explicit(&p->gen, 1U, memory_order_release);
        zcrudp_mt_wake_workers(p);
        for (uint32_t t = 1; t < p->num_threads; t++) {
#if defined(_WIN32)
            WaitForSingleObject(p->workers[t].handle, INFINITE);
            CloseHandle(p->workers[t].handle);
#else
            pthread_join(p->workers[t].handle, NULL);
#endif
        }
    }
#if !defined(_WIN32)
    pthread_cond_destroy(&p->work_cv);
    pthread_mutex_destroy(&p->work_lock);
#endif
}

static inline size_t zcrudp_ternary_mt_encode_plane(zcrudp_ternary_mt_pool_t *p,
                                                    const uint8_t *src,
                                                    uint8_t *ref,
                                                    uint8_t *delta_buf,
                                                    size_t len,
                                                    uint8_t plane_id,
                                                    uint16_t frame_id,
                                                    const zcrudp_ternary_config_t *cfg,
                                                    size_t *out_total_pkts) {
    /* Workers packetize into a fixed per-thread scratch buffer. */
    if (cfg->mtu_payload > sizeof(p->workers[0].pkt_scratch)) {
        if (out_total_pkts) *out_total_pkts = 0;
        return 0;
    }
    p->src = src;
    p->ref_or_canvas = ref;
    p->delta_buf = delta_buf;
    p->plane_len = len;
    p->total_blocks = (len + 9U) / 10U;
    p->plane_id = plane_id;
    p->frame_id = frame_id;
    p->cfg = cfg;

    zcrudp_mt_dispatch(p, ZCRUDP_MT_OP_ENCODE_PLANE);

    size_t bytes = 0;
    size_t pkts = 0;
    for (uint32_t t = 0; t < p->num_threads; t++) {
        bytes += p->workers[t].sink.total_bytes;
        pkts += p->workers[t].sink.count;
    }
    if (out_total_pkts) *out_total_pkts = pkts;
    return bytes;
}

static inline void zcrudp_ternary_mt_decay_plane(zcrudp_ternary_mt_pool_t *p,
                                                 uint8_t *canvas,
                                                 size_t len,
                                                 const zcrudp_ternary_config_t *cfg) {
    p->ref_or_canvas = canvas;
    p->plane_len = len;
    p->cfg = cfg;
    zcrudp_mt_dispatch(p, ZCRUDP_MT_OP_DECAY_PLANE);
}

/** True for packets that overwrite samples (anchors) rather than add to them. */
static inline bool zcrudp_mt_pkt_is_anchor(const uint8_t *pkt, size_t len) {
    if (len < ZCRUDP_TERNARY_PKT_HDR_SZ) {
        return false;
    }
    uint8_t ptype = (uint8_t)((pkt[2] >> 4) & 0x0FU);
    return ptype == ZCRUDP_TERNARY_PKT_INTRA || ptype == ZCRUDP_TERNARY_PKT_INTRA_EC ||
           ptype == ZCRUDP_TERNARY_PKT_INTRA_Q_EC;
}

/**
 * @brief Decodes a batch of packets with the same result as decoding them one
 *        by one in batch order.
 *
 * Within a frame, two delta packets never cover the same samples, nor do two
 * anchor packets, but an anchor does overwrite the deltas of its region. The
 * batch is therefore cut where the kind of packet changes (delta <-> anchor):
 * each run is decoded in parallel, the runs one after the other. A frame sent
 * as "deltas, then anchors" costs two dispatches.
 *
 * Range-coded packets carry many more blocks each, so a frame has fewer of
 * them to spread over the threads.
 */
static inline void zcrudp_ternary_mt_decode_packets(zcrudp_ternary_mt_pool_t *p,
                                                    uint8_t *canvas,
                                                    uint8_t *delta_buf,
                                                    size_t y_sz,
                                                    const zcrudp_ternary_config_t *cfg,
                                                    const uint8_t *pkts_base,
                                                    const size_t *pkt_lens,
                                                    size_t mtu_stride,
                                                    size_t pkt_count) {
    p->ref_or_canvas = canvas;
    p->delta_buf = delta_buf;
    p->y_sz = y_sz;
    p->cfg = cfg;
    p->pkt_in_stride = mtu_stride;

    size_t run_begin = 0;
    while (run_begin < pkt_count) {
        bool anchor = zcrudp_mt_pkt_is_anchor(pkts_base + run_begin * mtu_stride,
                                              pkt_lens[run_begin]);
        size_t run_end = run_begin + 1U;
        while (run_end < pkt_count &&
               zcrudp_mt_pkt_is_anchor(pkts_base + run_end * mtu_stride,
                                       pkt_lens[run_end]) == anchor) {
            run_end++;
        }
        p->pkt_in_base = pkts_base + run_begin * mtu_stride;
        p->pkt_in_lens = pkt_lens + run_begin;
        p->pkt_in_count = run_end - run_begin;
        zcrudp_mt_dispatch(p, ZCRUDP_MT_OP_DECODE_PKTS);
        run_begin = run_end;
    }
}

/**
 * @brief Parallelizes zcrudp_ternary_encode_intra_range across worker threads.
 *        Each thread encodes an independent contiguous byte slice into its own
 *        per-thread sink buffer (disjoint ref/delta_buf writes, zero data race).
 */
static inline size_t zcrudp_ternary_mt_encode_intra_range(zcrudp_ternary_mt_pool_t *p,
                                                          const uint8_t *src,
                                                          uint8_t *ref,
                                                          uint8_t *delta_buf,
                                                          size_t byte_begin,
                                                          size_t byte_end,
                                                          size_t plane_len,
                                                          uint8_t plane_id,
                                                          uint16_t frame_id,
                                                          const zcrudp_ternary_config_t *cfg,
                                                          size_t *out_total_pkts) {
    if (cfg->mtu_payload > sizeof(p->workers[0].pkt_scratch)) {
        if (out_total_pkts) *out_total_pkts = 0U;
        return 0U;
    }
    if (byte_end > plane_len) byte_end = plane_len;
    p->src = src;
    p->ref_or_canvas = ref;
    p->delta_buf = delta_buf;
    p->intra_begin = byte_begin;
    p->intra_end = byte_end;
    p->plane_len = plane_len;
    p->plane_id = plane_id;
    p->frame_id = frame_id;
    p->cfg = cfg;

    zcrudp_mt_dispatch(p, ZCRUDP_MT_OP_ENCODE_INTRA);

    size_t bytes = 0U, pkts = 0U;
    for (uint32_t t = 0U; t < p->num_threads; t++) {
        bytes += p->workers[t].sink.total_bytes;
        pkts += p->workers[t].sink.count;
    }
    if (out_total_pkts) *out_total_pkts = pkts;
    return bytes;
}

#ifdef __cplusplus
}
#endif

#endif /* ZCRUDP_TERNARY_MT_H */
