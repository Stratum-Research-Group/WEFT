#pragma once

#include <cstdint>
#include <string>
#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace weft {

enum class InstructionSet {
    SCALAR = 0,
    AVX2 = 1,
    AVX512_BASE = 2,
    AVX512_VPOPCNTDQ = 3
};

struct CpuFeatures {
    bool has_avx2 = false;
    bool has_fma = false;
    bool has_popcnt = false;
    bool has_avx512f = false;
    bool has_avx512dq = false;
    bool has_avx512bw = false;
    bool has_avx512vl = false;
    bool has_avx512_vpopcntdq = false;

    InstructionSet optimal_isa = InstructionSet::SCALAR;

    std::string isa_name() const {
        switch (optimal_isa) {
            case InstructionSet::AVX512_VPOPCNTDQ:
                return "AVX-512 + VPOPCNTDQ (Zen 4 / Sapphire Rapids)";
            case InstructionSet::AVX512_BASE:
                return "AVX-512 Foundation / Skylake-X";
            case InstructionSet::AVX2:
                return "AVX2 + POPCNT (Haswell / Zen 1-3)";
            default:
                return "Scalar Portable C++";
        }
    }
};

inline CpuFeatures detect_cpu_features() {
    CpuFeatures feat;

#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;

    // CPUID leaf 1: EAX=1
    __cpuid(1, eax, ebx, ecx, edx);
    feat.has_popcnt = (ecx & (1 << 23)) != 0;
    feat.has_fma    = (ecx & (1 << 12)) != 0;

    // CPUID leaf 7, sub-leaf 0: EAX=7, ECX=0
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    feat.has_avx2     = (ebx & (1 << 5)) != 0;
    feat.has_avx512f  = (ebx & (1 << 16)) != 0;
    feat.has_avx512dq = (ebx & (1 << 17)) != 0;
    feat.has_avx512bw = (ebx & (1 << 30)) != 0;
    feat.has_avx512vl = (ebx & (1 << 31)) != 0;
    feat.has_avx512_vpopcntdq = (ecx & (1 << 14)) != 0;

    if (feat.has_avx512f && feat.has_avx512dq && feat.has_avx512bw && feat.has_avx512_vpopcntdq) {
        feat.optimal_isa = InstructionSet::AVX512_VPOPCNTDQ;
    } else if (feat.has_avx512f) {
        feat.optimal_isa = InstructionSet::AVX512_BASE;
    } else if (feat.has_avx2 && feat.has_popcnt) {
        feat.optimal_isa = InstructionSet::AVX2;
    } else {
        feat.optimal_isa = InstructionSet::SCALAR;
    }
#else
    feat.optimal_isa = InstructionSet::SCALAR;
#endif

    return feat;
}

inline const CpuFeatures& get_cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}

} // namespace weft
