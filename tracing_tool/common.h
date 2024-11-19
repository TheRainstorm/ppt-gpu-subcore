#include <stdint.h>

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
    uint64_t mem_addrs1[32];
    uint64_t mem_addrs2[32];

    // 14 x 4 Byte
    int pred_num;
    int sm_id;
    int cta_id_x;
    int cta_id_y;
    int cta_id_z;
    int warp_id;
    int opcode_id;
    int pc;
    int dst_oprnd;
    int src_oprnds[5];
    uint32_t active_mask;
    uint32_t predicate_mask;

    // 10 x 1 Byte
    // 0, 1
    char pred_inst;
    char is_mem_inst;
    char mref_id;  // 1, 2

    char pred_active_threads;  // [0, 32]
    char dst_oprnd_type;    // [1, 3]
    char src_oprnds_type[5];

} inst_access_t;


#define cta_addresses_size_width  10000
#define cta_addresses_size_depth  10000