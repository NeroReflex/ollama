/*
 * IsoQuant CUDA dequantization kernels
 * Implements GPU support for quaternion 4D block rotation + 3-bit quantization
 */

#pragma once

#include "common.cuh"

// ---- IsoQuant dequantization ----
// Block size: 128 elements (32 groups of 4D each), 3-bit indices + 1-bit signs + norm
// Same block layout as turbo3_0

static __constant__ float ISO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// Unit quaternions: one per 4D group (32 groups for 128 dimensions)
// Each group applies quaternion sandwich product: v' = q * v * q^*
static __constant__ float ISO_QW[32] = {
    0.5765609741f, 0.3176580369f, -0.3234235942f, -0.5127438903f, 0.9233905673f, -0.3323571086f, 0.5468608141f, -0.2500519454f,
    -0.5812215805f, 0.3228830695f, -0.7299832702f, -0.4535493255f, -0.7338157296f, -0.2884652913f, -0.9000198841f, -0.0377033800f,
    0.5104404092f, 0.2033989877f, -0.2462528497f, 0.2314069420f, 0.0072374810f, 0.3923372924f, 0.4958070219f, -0.7235037088f,
    -0.9383618832f, 0.4430379272f, -0.2075705230f, 0.1983736306f, -0.8834578991f, 0.7389573455f, -0.0156172011f, 0.7738668919f
};

static __constant__ float ISO_QX[32] = {
    0.4450169504f, -0.5780548453f, 0.7089627385f, -0.3940812945f, -0.0897334740f, 0.4727236331f, 0.5542563796f, 0.0450818054f,
    -0.3657043576f, -0.4298477769f, 0.4666220546f, 0.7556306720f, -0.5284956098f, 0.7042509317f, 0.0230921544f, 0.7110687494f,
    0.3024962246f, -0.1157865301f, 0.7490812540f, -0.2582575679f, -0.2255804837f, 0.3838746250f, -0.3209520578f, -0.3477301002f,
    0.1824720055f, 0.4032751918f, 0.8433781862f, 0.9533935785f, -0.0620501526f, 0.0927560627f, 0.2964956462f, 0.2402082384f
};

static __constant__ float ISO_QY[32] = {
    0.2695076466f, -0.0201656222f, -0.1687686443f, -0.5415957570f, -0.2796611190f, 0.3510629535f, 0.2609911859f, -0.2715902030f,
    -0.0937586129f, 0.3095585108f, -0.4123268127f, -0.4394895136f, 0.0626545250f, -0.4811822474f, -0.0407132693f, -0.4566248953f,
    0.7834537029f, -0.6187923551f, 0.0809760988f, -0.8879503012f, -0.8928058147f, 0.8350352049f, -0.6994170547f, 0.5606835485f,
    0.2933705449f, 0.7377059460f, 0.4534837306f, -0.0009816211f, -0.3632916510f, -0.3959124386f, 0.1631654203f, 0.5088164806f
};

static __constant__ float ISO_QZ[32] = {
    -0.6300023794f, -0.7513582706f, -0.6035611629f, 0.5370919704f, 0.2471584976f, 0.7367672324f, 0.5706370473f, 0.9282674193f,
    0.7208684087f, -0.7843156457f, -0.2817355990f, -0.1736787707f, 0.4222335219f, -0.4350655377f, 0.4333281815f, 0.5333415866f,
    0.1847889870f, 0.7498788238f, 0.6096553802f, -0.3021556735f, -0.3898189068f, 0.0377884321f, 0.4024685621f, 0.2031257302f,
    0.0107116764f, -0.3112498820f, 0.1999502629f, -0.2273492515f, 0.2892593443f, 0.5372074246f, 0.9408631325f, 0.2907505929f
};

// Hamilton product for quaternion multiplication: v' = q * v * q^*
// For pure quaternion v = (0, vx, vy, vz), returns (0, rx, ry, rz) as separate components
static __device__ __forceinline__ void quat_mul_sandwich(
    float qw, float qx, float qy, float qz,
    float vx, float vy, float vz,
    float * rx, float * ry, float * rz
) {
    // q * v (treating v as pure quaternion (0, vx, vy, vz))
    float qv_w = -qx*vx - qy*vy - qz*vz;
    float qv_x =  qw*vx + qy*vz - qz*vy;
    float qv_y =  qw*vy + qz*vx - qx*vz;
    float qv_z =  qw*vz + qx*vy - qy*vx;
    
    // (q * v) * q^* where q^* = (qw, -qx, -qy, -qz)
    *rx = qv_w*(-qx) + qv_x*qw + qv_y*qz - qv_z*qy;
    *ry = qv_w*(-qy) + qv_y*qw + qv_z*qx - qv_x*qz;
    *rz = qv_w*(-qz) + qv_z*qw + qv_x*qy - qv_y*qx;
}

// Iso3: 3-bit RotorQuant via quaternion 4D rotation, block size 128
// For a group of 4 elements starting at index 4*group_idx
static __device__ __forceinline__ void dequantize_iso3_element(
    const block_iso3_0 * x, int elem_idx, float norm,
    float * out_v0, float * out_v1
) {
    // Extract quantized indices and signs
    const uint8_t * qs = x->qs;
    const uint8_t * signs = x->signs;
    
    // Byte position and bit offset
    const int qs_byte_idx = elem_idx / 4;
    const int qs_bit_off = (elem_idx % 4) * 2;
    const int sign_byte_idx = elem_idx / 8;
    const int sign_bit_off = (elem_idx % 8);
    
    // Extract 2-bit indices
    uint8_t qs_byte = qs[qs_byte_idx];
    int idx0 = (qs_byte >> qs_bit_off) & 0x3;
    int idx1 = (qs_byte >> (qs_bit_off + 2)) & 0x3;
    
    // Extract sign bits
    uint8_t sign_byte = signs[sign_byte_idx];
    float s0 = ((sign_byte >> sign_bit_off) & 1) ? -1.0f : 1.0f;
    float s1 = ((sign_byte >> ((sign_bit_off + 1) & 7)) & 1) ? -1.0f : 1.0f;
    
    // Dequantize
    *out_v0 = ISO_CENTROIDS_3BIT[idx0] * s0 * norm;
    *out_v1 = ISO_CENTROIDS_3BIT[idx1] * s1 * norm;
}

// Iso3: dequantize pair of elements from 4D group
// iqs is element index within block (even), produces elements iqs and iqs+1
static __device__ __forceinline__ void dequantize_iso3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_iso3_0 * x = (const block_iso3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    
    // Dequantize both elements
    float q0, q1;
    dequantize_iso3_element(x, iqs, norm, &q0, &q1);
    
    // Determine which 4D group this pair belongs to
    int group_idx = iqs / 4;
    int elem_in_group = iqs % 4;
    
    if (group_idx < 32) {
        // Apply quaternion rotation to restore correlation
        float qw = ISO_QW[group_idx];
        float qx = ISO_QX[group_idx];
        float qy = ISO_QY[group_idx];
        float qz = ISO_QZ[group_idx];
        
        if (elem_in_group < 2) {
            // Pair (0, 1) of the 4D group: apply quaternion to (q0, q1, 0)
            float rx, ry, rz;
            quat_mul_sandwich(qw, qx, qy, qz, q0, q1, 0.0f, &rx, &ry, &rz);
            v.x = rx;
            v.y = ry;
        } else {
            // Pair (2, 3) of the 4D group: apply quaternion to (q0, q1, 0) but represents 3rd and 4th elements
            float rx, ry, rz;
            quat_mul_sandwich(qw, qx, qy, qz, q0, q1, 0.0f, &rx, &ry, &rz);
            v.x = ry;
            v.y = rz;
        }
    } else {
        // No rotation for out-of-range groups
        v.x = q0;
        v.y = q1;
    }
}
