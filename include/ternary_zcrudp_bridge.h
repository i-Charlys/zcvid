/**
 * @file ternary_zcrudp_bridge.h
 * @brief Channel-aware packet router for integrating tern_vid with zcrudp::nextgen
 *        and Winsock RIO in zcstream (streaming/src/zcrudp_video_bridge.c).
 *
 * Separates:
 *  - Reliable / Cauchy-FEC channel: INTRA anchor slices & TILE_MV headers
 *  - Unreliable / RIO batch TX channel: DELTA & DELTA_2X trit packets
 */

#ifndef ZCRUDP_TERNARY_ZCRUDP_BRIDGE_H
#define ZCRUDP_TERNARY_ZCRUDP_BRIDGE_H

#include "ternary_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ZCRUDP_TERNARY_CHAN_UNRELIABLE_RIO = 0, /**< Fast-path RIO batch TX (DELTA / DELTA_2X) */
    ZCRUDP_TERNARY_CHAN_RELIABLE_FEC   = 1  /**< FEC/Reliable TX (INTRA / TILE_MV) */
} zcrudp_ternary_channel_t;

/**
 * @brief Inspects a ternary packet header and returns which zcrudp channel
 *        should transport it.
 */
static inline zcrudp_ternary_channel_t zcrudp_ternary_route_packet(const uint8_t *pkt,
                                                                   size_t pkt_len) {
    if (pkt_len < ZCRUDP_TERNARY_PKT_HDR_SZ) {
        return ZCRUDP_TERNARY_CHAN_UNRELIABLE_RIO;
    }
    zcrudp_ternary_pkt_hdr_t hdr;
    if (!zcrudp_ternary_pkt_hdr_read_le(pkt, &hdr)) {
        return ZCRUDP_TERNARY_CHAN_UNRELIABLE_RIO;
    }
    uint8_t ptype = ZCRUDP_TERNARY_HDR_TYPE(&hdr);
    if (ptype == ZCRUDP_TERNARY_PKT_INTRA || ptype == ZCRUDP_TERNARY_PKT_TILE_MV) {
        return ZCRUDP_TERNARY_CHAN_RELIABLE_FEC;
    }
    return ZCRUDP_TERNARY_CHAN_UNRELIABLE_RIO;
}

/**
 * @brief Callback adapter for zc_video_tx_send_frame / Winsock RIO batching.
 */
typedef void (*zcrudp_ternary_bridge_tx_fn)(const uint8_t *pkt_data,
                                            size_t pkt_len,
                                            zcrudp_ternary_channel_t chan,
                                            void *bridge_ctx);

typedef struct {
    zcrudp_ternary_bridge_tx_fn tx_fn;
    void *bridge_ctx;
    uint64_t reliable_bytes;
    uint64_t unreliable_bytes;
    uint64_t reliable_pkts;
    uint64_t unreliable_pkts;
} zcrudp_ternary_bridge_router_t;

static inline void zcrudp_ternary_bridge_pkt_cb(const uint8_t *pkt_data,
                                                size_t pkt_len,
                                                void *user_ctx) {
    zcrudp_ternary_bridge_router_t *r = (zcrudp_ternary_bridge_router_t *)user_ctx;
    zcrudp_ternary_channel_t chan = zcrudp_ternary_route_packet(pkt_data, pkt_len);
    if (chan == ZCRUDP_TERNARY_CHAN_RELIABLE_FEC) {
        r->reliable_bytes += pkt_len;
        r->reliable_pkts++;
    } else {
        r->unreliable_bytes += pkt_len;
        r->unreliable_pkts++;
    }
    if (r->tx_fn != NULL) {
        r->tx_fn(pkt_data, pkt_len, chan, r->bridge_ctx);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* ZCRUDP_TERNARY_ZCRUDP_BRIDGE_H */
