// onchip_check.cu -- low latency on-chip storage integrity check for
// Iluvatar ix GPUs. Self-contained: public interface, implementation and
// validator in one file.
//
// Covers shared memory (SLB/SRAM), the register file, L1 and L2, and is meant
// to be called immediately before and after a workload. A non-zero return means
// the hardware did not read back what was written.
//
//   onchip_check_init(0);
//   ...
//   OnchipResult r;
//   if (onchip_check_run(&r, seed, stream) != ONCHIP_OK) { /* quarantine */ }
//
// Use the launch/collect pair instead to enqueue the check on the workload's own
// stream and pick up the verdict later, keeping it off the critical path.
//
// Build:
//   as a validator      clang++ -x ivcore --cuda-gpu-arch=ivcore11 \
//                         -DONCHIP_CHECK_STANDALONE onchip_check.cu -lcudart
//   inside an app       add onchip_check.cu to the build, or #include it from
//                       one translation unit; without ONCHIP_CHECK_STANDALONE
//                       no main() is emitted.
//
// Design targets, in order:
//   1. every SLB word, every register of every lane, and the L1/L2 lines of every
//      CU are written and read back with complementary values, so a stuck bit or
//      a mis-decoded address cannot go unnoticed;
//   2. total call latency far below a millisecond.
//
// Three kernels, because the levels want conflicting resources: the SLB test
// needs one block per CU holding the whole SLB, the register test needs as many
// threads per CU as the register file allows, and the L2 test needs a wide grid
// for bandwidth.
//
// What makes the coverage claims hold rather than merely likely:
//   - a grid-wide rendezvous runs before the SLB/L1 and register tests; each
//     block holds a whole CU's SLB, so all blocks being simultaneously past it
//     proves they sit on distinct CUs. Without it the scheduler would be free to
//     run every block on one CU and the test would pass having checked 1/N of
//     the SRAM.
//   - the block count and register count come from
//     cudaOccupancyMaxActiveBlocksPerMultiprocessor rather than from dividing
//     capacities by hand, because a few bytes of unaccounted overhead silently
//     halves residency, and therefore coverage, with no error reported.
//   - registers are tested by name. Values are pinned to v0..v255 in four
//     windows, so which registers were covered follows from the source instead
//     of from trusting the allocator, and init rejects any window the compiler
//     could not hold: cudaFuncGetAttributes must report zero spill, checked
//     again by the validator. Without that check the widest kernel wins a sizing
//     sweep while quietly reading a third of its values back from local memory,
//     which is what the adaptive fallback used to do while reporting 100%.
//   - the built-in validator corrupts every location and checks the reported
//     error count against the number of locations that should have been read
//     back. Single point injection would only prove the reporting path works.

#ifndef ONCHIP_CHECK_CU
#define ONCHIP_CHECK_CU

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <unistd.h>

// ==========================================================================
// Public interface
// ==========================================================================

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ONCHIP_OK = 0,
  ONCHIP_FAULT = 1,       // a location did not read back correctly
  ONCHIP_INCOMPLETE = 2,  // grid could not cover every CU; verdict inconclusive
  ONCHIP_NOT_READY = 3,
  ONCHIP_CUDA_ERROR = 4
};

enum { ONCHIP_LV_SLB = 0, ONCHIP_LV_REG, ONCHIP_LV_L1, ONCHIP_LV_L2,
       ONCHIP_LV_COUNT };

// Registers per lane on ivcore11: 64KB per NR over a 64 lane warp. The register
// level tests them by name in four windows of ONCHIP_REG_LANE/4, so this is both
// the width of a lane's file and the size of the injection index space per
// thread.
enum { ONCHIP_REG_LANE = 256 };

typedef struct {
  unsigned errors[ONCHIP_LV_COUNT];
  unsigned firstLevel;      // which level the first mismatch came from
  unsigned firstBlock;
  unsigned firstThread;
  unsigned firstWord;       // word index within that level's slice
  unsigned firstExpected;
  unsigned firstGot;
  unsigned firstPhase;
  unsigned claimed;         // internal
  unsigned arrived;         // internal rendezvous counter, SLB/L1 kernel
  unsigned arrivedReg;      // internal rendezvous counter, register kernel
  unsigned peakResident;    // equals slbGrid when every CU was covered
  unsigned peakResidentReg; // equals regGrid when every CU was covered
  unsigned coverageIncomplete;
} OnchipResult;

typedef struct {
  int cuCount;
  int slbGrid, slbThreads;
  size_t slbBytesPerBlock, slbBytesTotal;
  int regGrid, regThreads, regPerThread;
  size_t regBytesTotal, regBytesCapacity;
  size_t l1BytesTotal, l2BytesTotal;
  // What the compiler actually produced for the register kernel. regBytesTotal
  // assumes each thread holds regPerThread values in registers, so these are
  // what makes that claim checkable: regLocalBytes must be 0, because anything
  // spilled is verified in local memory instead of the register file.
  int regNumRegs;
  size_t regLocalBytes;
} OnchipConfig;

// levelMask selects which levels to test, e.g.
// (1<<ONCHIP_LV_SLB)|(1<<ONCHIP_LV_REG). Pass 0 for all four.
int onchip_check_init(unsigned levelMask);
int onchip_check_config(OnchipConfig *c);
int onchip_check_run(OnchipResult *out, unsigned seed, cudaStream_t stream);
int onchip_check_launch(unsigned seed, cudaStream_t stream);
int onchip_check_collect(OnchipResult *out, cudaStream_t stream);
// Corrupts one location per level on purpose, to verify detection still works.
int onchip_check_run_inject(OnchipResult *out, unsigned seed,
                            cudaStream_t stream);
// Validation: mode 1 corrupts the single location `index`, mode 2 corrupts
// every location, always at the given bit. Comparing the resulting error count
// with the number of locations that should have been checked is what proves the
// test skips nothing.
int onchip_check_run_inject_mode(OnchipResult *out, unsigned seed,
                                 unsigned mode, unsigned bit,
                                 unsigned long long index,
                                 cudaStream_t stream);
// Runs the full validation suite and prints a report. Returns the number of
// failed checks, so 0 means the checker itself is trustworthy on this device.
int onchip_check_validate(int verbose);
// Mirror a FAULT or INCOMPLETE verdict through the driver XID path
// (ITRCTL_XID_OCCUR -> XID_CP_HUNG), so existing XID/dmesg alert pipelines pick
// it up. Off by default, because a library linked into an application must not
// raise XIDs unless asked. Setting ONCHIP_CHECK_KMSG=1 in the environment
// before init enables it without a code change.
void onchip_check_set_kmsg(int enable);
// 1 if /dev/itrctl can be opened for the XID ioctl. Reporting stays silent when
// the control device is absent, so a caller that must not lose a fault report
// checks this once at startup rather than discovering it when a card goes bad.
int onchip_check_kmsg_ready(void);
int onchip_check_report_xid(unsigned cardid, unsigned xid, const char *message);
void onchip_check_free(void);
const char *onchip_check_last_error(void);
const char *onchip_level_name(int level);

// How many times each level reads back each location per call. The validator
// multiplies the location counts by these, so a change to a kernel that is not
// reflected here fails loudly instead of silently reducing coverage.
enum {
  ONCHIP_PASSES_SLB = 3,  // v, ~v, v
  ONCHIP_PASSES_REG = 2,  // value and complement
  ONCHIP_PASSES_L1 = 4,   // 2 patterns x 2 read repetitions
  ONCHIP_PASSES_L2 = 2    // value and complement
};

#ifdef __cplusplus
}
#endif

// ==========================================================================
// Implementation
// ==========================================================================

static char g_err[256] = "";

#define CK(call)                                                               \
  do {                                                                         \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
      snprintf(g_err, sizeof(g_err), "%s:%d %s -> %s", __FILE__, __LINE__,      \
               #call, cudaGetErrorString(e_));                                  \
      return ONCHIP_CUDA_ERROR;                                                 \
    }                                                                           \
  } while (0)

#define KOP_NO_L1 1   // serviced by L2
#define KOP_NO_L2 2   // serviced by L1, the only policy that isolates it

// ------------------------------------------------------------------ device

__device__ __host__ __forceinline__ unsigned mix(unsigned a, unsigned b) {
  unsigned h = (a * 2654435761u) ^ (b * 0x9e3779b9u);
  h ^= h >> 15;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  return h;
}

// Fault injection used only by the validator. Mode 1 proves the reporting path
// works; mode 2 corrupts every location so the reported error count can be
// checked against the number of locations that should have been verified, which
// is what actually proves nothing is skipped.
struct Inject {
  unsigned mode;              // 0 none, 1 single location, 2 every location
  unsigned bit;
  unsigned long long index;   // target for mode 1

  __device__ __forceinline__ unsigned flip(unsigned long long i) const {
    if (mode == 2) return 1u << bit;
    if (mode == 1 && i == index) return 1u << bit;
    return 0u;
  }
};

__device__ __forceinline__ unsigned gload(const unsigned *p, int kop) {
  return __ivcorex_ml_mem_a64_load_u32((const volatile unsigned *)p, kop);
}
__device__ __forceinline__ void gstore(unsigned *p, unsigned v, int kop) {
  __ivcorex_ml_mem_a64_store_u32(v, (const volatile unsigned *)p, kop);
}

__device__ void record_n(OnchipResult *out, unsigned lv, unsigned count,
                         unsigned word, unsigned e, unsigned g,
                         unsigned phase) {
  atomicAdd(&out->errors[lv], count);
  if (atomicCAS(&out->claimed, 0u, 1u) == 0u) {
    out->firstLevel = lv;
    out->firstBlock = blockIdx.x;
    out->firstThread = threadIdx.x;
    out->firstWord = word;
    out->firstExpected = e;
    out->firstGot = g;
    out->firstPhase = phase;
  }
}

extern __shared__ unsigned smem[];

// Coverage rendezvous; bounded so an occupied GPU reports "incomplete" rather
// than hanging. The flag is broadcast through dynamic shared memory because a
// static __shared__ byte would push the block over the per-CU SLB budget and
// halve residency.
__device__ __forceinline__ bool rendezvous(volatile unsigned *scratch,
                                           unsigned *arrived,
                                           unsigned long long timeout) {
  __syncthreads();
  if (threadIdx.x == 0) {
    scratch[0] = 1u;
    atomicAdd(arrived, 1u);
    unsigned long long t0 = clock64();
    while (atomicAdd(arrived, 0u) < gridDim.x) {
      if ((unsigned long long)(clock64() - t0) > timeout) {
        scratch[0] = 0u;
        break;
      }
    }
  }
  __syncthreads();
  return scratch[0] != 0u;
}

// ---- shared memory and L1, one block per CU
//
// Three passes v, ~v, v: two already catch every stuck-at fault, the third
// makes each cell perform both the 0->1 and the 1->0 transition. Values are
// unique per (block, word) so a read returning a neighbour's data fails, and
// each thread verifies words written by a different thread, which exercises the
// bank crossbar at no extra cost.
// The whole SLB is claimed as dynamic shared memory even when only L1 is under
// test: that allocation is what keeps residency at one block per CU, which is
// what makes the per-CU L1 slices land on distinct CUs.
__global__ void k_slb_l1(OnchipResult *out, unsigned nwords, unsigned seed,
                         Inject inj, unsigned long long timeout,
                         unsigned *l1buf, unsigned l1Words, int doSLB,
                         int doL1) {
  volatile unsigned *s = smem;
  const unsigned tid = threadIdx.x, nt = blockDim.x, blk = blockIdx.x;

  if (!rendezvous(s, &out->arrived, timeout)) {
    if (tid == 0) atomicExch(&out->coverageIncomplete, 1u);
  } else if (tid == 0) {
    atomicMax(&out->peakResident, gridDim.x);
  }

  const unsigned mask = nwords - 1;          // nwords is a power of two, so the
  const unsigned shift = (nwords >> 1) + 17; // wrap is a mask, not a division
  const unsigned base = mix(blk, seed);      // per-block, so the inner loop is
                                             // one multiply-xor per word

#pragma unroll 1
  for (int phase = 0; phase < 3 && doSLB; ++phase) {
    for (unsigned i = tid; i < nwords; i += nt) {
      unsigned v = base ^ (i * 0x9E3779B9u);
      if (phase == 1) v = ~v;
      // Corruption is applied by the thread that owns the word, before the
      // barrier, so detection is deterministic rather than racing the readers.
      v ^= inj.flip(blk * (unsigned long long)nwords + i);
      s[i] = v;
    }
    __syncthreads();

    unsigned bad = 0, fW = 0, fE = 0, fG = 0, found = 0;
    for (unsigned i = tid; i < nwords; i += nt) {
      unsigned j = (i + shift) & mask;
      unsigned e = base ^ (j * 0x9E3779B9u);
      if (phase == 1) e = ~e;
      unsigned g = s[j];
      if (g != e) {
        ++bad;
        if (!found) { found = 1; fW = j; fE = e; fG = g; }
      }
    }
    if (bad) record_n(out, ONCHIP_LV_SLB, bad, fW, fE, fG, phase);
    __syncthreads();
  }

  if (!doL1) return;

  // L1: a private slice per CU, accessed with kop=2 so a hit can only be
  // answered by L1 itself. The first read pass populates it, the second is
  // served from it; both are valid correctness checks either way.
  unsigned *b = l1buf + (size_t)blk * l1Words;
  for (int phase = 0; phase < 2; ++phase) {
    for (unsigned i = tid; i < l1Words; i += nt) {
      unsigned v = mix(blk ^ 0x5a5au, i ^ seed);
      if (phase) v = ~v;
      v ^= inj.flip(blk * (unsigned long long)l1Words + i);
      gstore(b + i, v, KOP_NO_L2);
    }
    __syncthreads();
    for (int rep = 0; rep < 2; ++rep) {
      unsigned bad = 0, fW = 0, fE = 0, fG = 0, found = 0;
      for (unsigned i = tid; i < l1Words; i += nt) {
        unsigned e = mix(blk ^ 0x5a5au, i ^ seed);
        if (phase) e = ~e;
        unsigned g = gload(b + i, KOP_NO_L2);
        if (g != e) {
          ++bad;
          if (!found) { found = 1; fW = i; fE = e; fG = g; }
        }
      }
      if (bad) record_n(out, ONCHIP_LV_L1, bad, fW, fE, fG, phase);
    }
    __syncthreads();
  }
}

// ---- register file
//
// N values per thread are generated, made opaque so the compiler cannot
// rematerialise them, held, then verified. No memory traffic at all, which is
// both what keeps this cheap and the property that makes the coverage claim
// mean anything: N is chosen so the compiler spills nothing, since a value that
// went to local memory and came back says nothing about the register file.

// Compares N held registers without emitting a reporting path per comparison:
// the first mismatch is carried along in a few scalars and reported once at the
// end. Inlining the report into all N unrolled compares blew the kernel up to
// 16k instructions and made it instruction-fetch bound.
//
// How much state this loop carries decides how much of the register file can be
// tested at all, because it competes with the N values for the same 256
// per-lane registers. Carrying the mismatching pair itself (expected and got)
// measured at 4 registers per value, which spilled everything above N=60 and
// capped honest coverage at 31%. The two accumulators below cost 1.07 per value
// and reach N=239, or 93% of the file:
//
//   - fIdx by minimum rather than a first-flag select chain;
//   - the failing bit patterns OR-ed together, which is a reduction with no
//     dependency on the previous iteration, unlike a conditional copy.
//
// The pair is then reconstructed: expected is recomputed from fIdx after the
// loop, out of two values that are live anyway, and got follows from the xor.
// With one failing slot per thread, which is the usual case and always the case
// under single-point injection, that is exact. With several, the xor is their
// union, so got shows every bit that differed rather than one slot's contents --
// still the right thing to act on, but it is not a literal read-out.
template <int N>
__device__ __forceinline__ void verify_regs(OnchipResult *out,
                                            const unsigned *r, unsigned base,
                                            unsigned gid, unsigned phase) {
  unsigned bad = 0, fIdx = ~0u, xacc = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    unsigned x = r[i] ^ (base ^ (unsigned)(i * 0x9E3779B9u));
    unsigned ne = x ? 1u : 0u;
    bad += ne;
    xacc |= x;
    if (ne && (unsigned)i < fIdx) fIdx = (unsigned)i;
  }
  if (bad) {
    const unsigned e = base ^ (unsigned)(fIdx * 0x9E3779B9u);
    record_n(out, ONCHIP_LV_REG, bad, gid * N + fIdx, e, e ^ xacc, phase);
  }
}

template <int N>
__global__ void k_reg(OnchipResult *out, unsigned seed,
                      unsigned long long hold, Inject inj,
                      unsigned long long timeout) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Same coverage argument as the SLB kernel: without forcing every block to be
  // resident at once, the scheduler could run them all on one CU and the test
  // would pass having exercised a single CU's register file. Shared memory is
  // not the constrained resource here, so a static flag is fine.
  {
    __shared__ unsigned s_ok;
    __syncthreads();
    if (threadIdx.x == 0) {
      s_ok = 1u;
      atomicAdd(&out->arrivedReg, 1u);
      unsigned long long t0 = clock64();
      while (atomicAdd(&out->arrivedReg, 0u) < gridDim.x) {
        if ((unsigned long long)(clock64() - t0) > timeout) { s_ok = 0u; break; }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (s_ok == 0u) atomicExch(&out->coverageIncomplete, 1u);
      else atomicMax(&out->peakResidentReg, gridDim.x);
    }
  }

  // One mixed base per thread, then a compile-time constant per slot: the loop
  // is fully unrolled so i * GOLDEN folds away and each value costs a single
  // xor. Hashing every slot instead made this kernel ALU bound and several
  // times slower, for no gain in bit coverage.
  const unsigned base = mix(gid, seed);
  unsigned r[N];

#pragma unroll
  for (int i = 0; i < N; ++i) {
    unsigned v = base ^ (unsigned)(i * 0x9E3779B9u);
    v ^= inj.flip((unsigned long long)gid * N + i);
    asm volatile("" : "+r"(v));
    r[i] = v;
  }

  if (hold) {
    unsigned long long t0 = clock64();
    while ((unsigned long long)(clock64() - t0) < hold) {}
  }

  verify_regs<N>(out, r, base, gid, 0);

  // Second pass over the complement, so every register bit is held as both a 0
  // and a 1.
#pragma unroll
  for (int i = 0; i < N; ++i) {
    // Regenerate rather than invert in place: inverting an already corrupted
    // value would land back on what the checker expects and hide the fault.
    unsigned v = ~(base ^ (unsigned)(i * 0x9E3779B9u));
    v ^= inj.flip((unsigned long long)gid * N + i);
    asm volatile("" : "+r"(v));
    r[i] = v;
  }
  if (hold) {
    unsigned long long t0 = clock64();
    while ((unsigned long long)(clock64() - t0) < hold) {}
  }
  // ~(base ^ K) == (~base) ^ K, so the same comparison serves the complement.
  verify_regs<N>(out, r, ~base, gid, 1);
}

// ---- register file, by name
//
// The adaptive kernel above covers whatever the compiler agrees to hold, which
// measured 239 of the 256 registers a lane has: the rest carry the loop's own
// state and there is no way to tell which those are. Naming the registers
// removes both problems. A value declared as
//
//   register unsigned r7 asm("v7");
//
// is in v7 at every asm that names it, so a window of registers can be written
// and read back by number and the coverage claim follows from the source rather
// than from trusting the allocator.
//
// The price is that clang reserves a named register for the whole function and
// needs about two scratch registers per pinned value, so a window has to be
// small: measured spill-free at 64 and spilling at 96. Four windows of 64 tile
// the file exactly, so the register level runs four kernels back to back and
// their union is all 256 registers, each written and read back by name.
//
// All four waves share one rendezvous counter. They run in order on one stream,
// so wave w waits for (w+1) * gridDim.x arrivals: every block of wave w is past
// the barrier before wave w+1 starts counting.

#define REG_W0(X)                                                             \
  X(0)X(1)X(2)X(3)X(4)X(5)X(6)X(7)X(8)X(9)X(10)X(11)X(12)X(13)X(14)X(15) \
  X(16)X(17)X(18)X(19)X(20)X(21)X(22)X(23)X(24)X(25)X(26)X(27)X(28)X(29)X(30)X(31) \
  X(32)X(33)X(34)X(35)X(36)X(37)X(38)X(39)X(40)X(41)X(42)X(43)X(44)X(45)X(46)X(47) \
  X(48)X(49)X(50)X(51)X(52)X(53)X(54)X(55)X(56)X(57)X(58)X(59)X(60)X(61)X(62)X(63)

#define REG_W1(X)                                                             \
  X(64)X(65)X(66)X(67)X(68)X(69)X(70)X(71)X(72)X(73)X(74)X(75)X(76)X(77)X(78)X(79) \
  X(80)X(81)X(82)X(83)X(84)X(85)X(86)X(87)X(88)X(89)X(90)X(91)X(92)X(93)X(94)X(95) \
  X(96)X(97)X(98)X(99)X(100)X(101)X(102)X(103)X(104)X(105)X(106)X(107)X(108)X(109)X(110)X(111) \
  X(112)X(113)X(114)X(115)X(116)X(117)X(118)X(119)X(120)X(121)X(122)X(123)X(124)X(125)X(126)X(127)

#define REG_W2(X)                                                             \
  X(128)X(129)X(130)X(131)X(132)X(133)X(134)X(135)X(136)X(137)X(138)X(139)X(140)X(141)X(142)X(143) \
  X(144)X(145)X(146)X(147)X(148)X(149)X(150)X(151)X(152)X(153)X(154)X(155)X(156)X(157)X(158)X(159) \
  X(160)X(161)X(162)X(163)X(164)X(165)X(166)X(167)X(168)X(169)X(170)X(171)X(172)X(173)X(174)X(175) \
  X(176)X(177)X(178)X(179)X(180)X(181)X(182)X(183)X(184)X(185)X(186)X(187)X(188)X(189)X(190)X(191)

#define REG_W3(X)                                                             \
  X(192)X(193)X(194)X(195)X(196)X(197)X(198)X(199)X(200)X(201)X(202)X(203)X(204)X(205)X(206)X(207) \
  X(208)X(209)X(210)X(211)X(212)X(213)X(214)X(215)X(216)X(217)X(218)X(219)X(220)X(221)X(222)X(223) \
  X(224)X(225)X(226)X(227)X(228)X(229)X(230)X(231)X(232)X(233)X(234)X(235)X(236)X(237)X(238)X(239) \
  X(240)X(241)X(242)X(243)X(244)X(245)X(246)X(247)X(248)X(249)X(250)X(251)X(252)X(253)X(254)X(255)

#define REG_GOLD 0x9E3779B9u

// Injection is resolved once per thread here, not per slot as in the adaptive
// kernel. With pinned values the per-slot form costs two live registers for the
// 64-bit index and pushes the window into spilling; with the decision hoisted
// the same window is spill-free.
#define REG_DECL(i) register unsigned r##i asm("v" #i);
#define REG_WR(i)                                                              \
  r##i = (base ^ (unsigned)((i) * REG_GOLD)) ^                                  \
         (injAll | (((unsigned)(i) == injSlot) ? injOne : 0u));                 \
  asm volatile("" : "+r"(r##i));
#define REG_WRC(i)                                                             \
  r##i = ~(base ^ (unsigned)((i) * REG_GOLD)) ^                                 \
         (injAll | (((unsigned)(i) == injSlot) ? injOne : 0u));                 \
  asm volatile("" : "+r"(r##i));
// The second asm names the register at the read as well, so the value is in
// v##i at both ends of the hold rather than only where it was produced.
#define REG_CK(i)                                                              \
  asm volatile("" : "+r"(r##i));                                               \
  { unsigned x = r##i ^ (base ^ (unsigned)((i) * REG_GOLD));                    \
    unsigned ne = x ? 1u : 0u;                                                  \
    bad += ne; xacc |= x;                                                       \
    if (ne && fIdx > (unsigned)(i)) fIdx = (unsigned)(i); }
#define REG_CKC(i)                                                             \
  asm volatile("" : "+r"(r##i));                                               \
  { unsigned x = r##i ^ ~(base ^ (unsigned)((i) * REG_GOLD));                   \
    unsigned ne = x ? 1u : 0u;                                                  \
    bad += ne; xacc |= x;                                                       \
    if (ne && fIdx > (unsigned)(i)) fIdx = (unsigned)(i); }
#define REG_HOLD                                                               \
  if (hold) {                                                                  \
    unsigned long long t0 = clock64();                                          \
    while ((unsigned long long)(clock64() - t0) < hold) {}                      \
  }
#define REG_REPORT(ph)                                                         \
  if (bad) {                                                                   \
    const unsigned e = base ^ (unsigned)(fIdx * REG_GOLD);                      \
    record_n(out, ONCHIP_LV_REG, bad, (unsigned)(slot0 + fIdx), e, e ^ xacc,     \
             ph);                                                               \
  }                                                                            \
  bad = 0; xacc = 0; fIdx = ~0u;

#define MAKE_REG_WAVE(name, LIST, WAVE)                                        \
  __global__ void name(OnchipResult *out, unsigned seed,                       \
                       unsigned long long hold, Inject inj,                     \
                       unsigned long long timeout) {                            \
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;                 \
    {                                                                           \
      __shared__ unsigned s_ok;                                                 \
      __syncthreads();                                                          \
      if (threadIdx.x == 0) {                                                   \
        s_ok = 1u;                                                              \
        atomicAdd(&out->arrivedReg, 1u);                                        \
        const unsigned want = ((WAVE) + 1u) * gridDim.x;                        \
        unsigned long long t0 = clock64();                                      \
        while (atomicAdd(&out->arrivedReg, 0u) < want) {                        \
          if ((unsigned long long)(clock64() - t0) > timeout) {                 \
            s_ok = 0u;                                                          \
            break;                                                              \
          }                                                                     \
        }                                                                       \
      }                                                                         \
      __syncthreads();                                                          \
      if (threadIdx.x == 0) {                                                   \
        if (s_ok == 0u) atomicExch(&out->coverageIncomplete, 1u);               \
        else atomicMax(&out->peakResidentReg, gridDim.x);                       \
      }                                                                         \
    }                                                                           \
    const unsigned base = mix(gid, seed);                                       \
    const unsigned long long slot0 = (unsigned long long)gid * ONCHIP_REG_LANE;  \
    const unsigned injAll = (inj.mode == 2u) ? (1u << inj.bit) : 0u;             \
    unsigned injSlot = ~0u, injOne = 0u;                                        \
    if (inj.mode == 1u && inj.index >= slot0 &&                                 \
        inj.index < slot0 + ONCHIP_REG_LANE) {                                  \
      injSlot = (unsigned)(inj.index - slot0);                                  \
      injOne = 1u << inj.bit;                                                   \
    }                                                                           \
    unsigned bad = 0, xacc = 0, fIdx = ~0u;                                     \
    LIST(REG_DECL)                                                              \
    LIST(REG_WR)                                                                \
    REG_HOLD                                                                    \
    LIST(REG_CK)                                                                \
    REG_REPORT(0)                                                               \
    LIST(REG_WRC)                                                               \
    REG_HOLD                                                                    \
    LIST(REG_CKC)                                                               \
    REG_REPORT(1)                                                               \
  }

MAKE_REG_WAVE(k_reg_w0, REG_W0, 0)
MAKE_REG_WAVE(k_reg_w1, REG_W1, 1)
MAKE_REG_WAVE(k_reg_w2, REG_W2, 2)
MAKE_REG_WAVE(k_reg_w3, REG_W3, 3)

// ---- L2
//
// Each block owns a slice, writes it with kop=1 so the line is dirtied in L2
// with L1 out of the way, and reads it straight back from L2. No cross-block
// dependency, so no barrier is needed.
__global__ void k_l2(OnchipResult *out, unsigned *buf, unsigned long long words,
                     unsigned seed, Inject inj) {
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  const unsigned long long start =
      blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;

  for (int phase = 0; phase < 2; ++phase) {
    for (unsigned long long i = start; i < words; i += stride) {
      unsigned v = mix((unsigned)i, seed);
      if (phase) v = ~v;
      // The corruption is applied by the thread that owns the word, and that
      // same thread verifies it below. Injecting from one thread while another
      // reads makes the self-test racy and it will silently miss.
      v ^= inj.flip(i);
      gstore(buf + i, v, KOP_NO_L1);
    }
    __threadfence();
    unsigned bad = 0, fW = 0, fE = 0, fG = 0, found = 0;
    for (unsigned long long i = start; i < words; i += stride) {
      unsigned e = mix((unsigned)i, seed);
      if (phase) e = ~e;
      unsigned g = gload(buf + i, KOP_NO_L1);
      if (g != e) {
        ++bad;
        if (!found) { found = 1; fW = (unsigned)i; fE = e; fG = g; }
      }
    }
    if (bad) record_n(out, ONCHIP_LV_L2, bad, fW, fE, fG, phase);
  }
}

// -------------------------------------------------------------------- host

struct Ctx {
  OnchipResult *pinned = nullptr;
  OnchipResult *device = nullptr;   // counters stay in device memory: in-kernel
                                    // atomics on mapped host memory would cross
                                    // PCIe and dominate the latency budget
  unsigned levels = 0;
  int nsm = 0;
  int slbGrid = 0, slbThreads = 0;
  size_t slbBytes = 0;
  int regGrid = 0, regThreads = 0, regN = 0;
  size_t regCapacity = 0;
  int regNumRegs = 0;
  size_t regLocalBytes = 0;
  bool regPinned = false;   // named-register waves rather than the adaptive kernel
  unsigned *l1buf = nullptr;
  unsigned l1Words = 0;
  unsigned *l2buf = nullptr;
  unsigned long long l2Words = 0;
  int l2Grid = 0, l2Threads = 0;
  unsigned long long timeout = 0, regHold = 0;
  int dev = 0;
  char ident[96] = "";      // "<pci bdf> GPU-<uuid>", as the driver prints it
  unsigned kmsg = 0;
  unsigned reported = 0;    // level bits already sent, plus the bit below
  bool ready = false;
};

enum { KMSG_INCOMPLETE_BIT = 1u << ONCHIP_LV_COUNT };

static Ctx g;

extern "C" void onchip_check_set_kmsg(int enable) {
  g.kmsg = enable ? 1u : 0u;
}

extern "C" int onchip_check_kmsg_ready(void) {
  int fd = open("/dev/itrctl", O_RDWR | O_CLOEXEC);
  if (fd < 0) return 0;
  close(fd);
  return 1;
}

// Minimal copy of the ctrl-device XID ioctl ABI (see ixdriver
// common/{aip_ioctl,thunk,itr_xid_type}.h). Kept local so this file stays a
// single-file drop-in with no driver header dependency.
enum { XID_CP_HUNG = 38, UMDXID_MSG_LEN = 120, ITRCTL_XID_OCCUR = 200 };
struct OnchipXidOccurArgs {
  uint32_t cardid;
  uint32_t xid;
  char msg[UMDXID_MSG_LEN];
};
#define ONCHIP_XID_OCCUR_IOCTL \
  _IOW(0x81, ITRCTL_XID_OCCUR, struct OnchipXidOccurArgs)

extern "C" int onchip_check_report_xid(unsigned cardid, unsigned xid,
                                       const char *message) {
  int fd = open("/dev/itrctl", O_RDWR | O_CLOEXEC);
  if (fd < 0)
    return -1;
  OnchipXidOccurArgs args = {};
  args.cardid = (uint32_t)cardid;
  args.xid = (uint32_t)xid;
  snprintf(args.msg, sizeof args.msg, "%s", message ? message : "");
  args.msg[UMDXID_MSG_LEN - 1] = '\0';
  const int rc = ioctl(fd, ONCHIP_XID_OCCUR_IOCTL, &args);
  close(fd);
  return rc;
}

// Raise XID_CP_HUNG through /dev/itrctl. The driver prepends BDF/UUID/XID, so
// msg only carries the onchip detail. msg must stay NUL-terminated within
// UMDXID_MSG_LEN (KMD rejects a non-zero final byte). Never fatal.
static void xid_report(const char *msg) {
  (void)onchip_check_report_xid((unsigned)g.dev, XID_CP_HUNG, msg);
}

// Each level is announced once per init session. A caller that checks around
// every workload would otherwise burn through xid_log_max_count on a bad card
// and silence later XIDs. Tying it to the session rather than the process is
// what a multi-device sweep needs: one init/free per card, so a fault on card 0
// does not silence the same level on card 5.
static void report_kmsg(int rc, const OnchipResult *r) {
  if (!g.kmsg) return;
  char msg[UMDXID_MSG_LEN];

  if (rc == ONCHIP_FAULT) {
    unsigned levels = 0;
    for (int i = 0; i < ONCHIP_LV_COUNT; ++i)
      if (r->errors[i]) levels |= 1u << i;
    if ((levels & ~g.reported) == 0) return;
    g.reported |= levels;
    unsigned x = r->firstExpected ^ r->firstGot;
    snprintf(msg, sizeof msg,
             "worker_health_check FAULT SLB=%u REG=%u L1=%u L2=%u first=%s "
             "b=%u t=%u w=%u e=%08x g=%08x xor=%08x bits=%d",
             r->errors[ONCHIP_LV_SLB], r->errors[ONCHIP_LV_REG],
             r->errors[ONCHIP_LV_L1], r->errors[ONCHIP_LV_L2],
             onchip_level_name(r->firstLevel), r->firstBlock, r->firstThread,
             r->firstWord, r->firstExpected, r->firstGot, x,
             __builtin_popcount(x));
  } else {
    // Incomplete coverage: the hardware is not accused of anything, the run
    // simply did not cover every CU and so proves nothing either way. Still
    // raised as XID_CP_HUNG so the same alert path sees it.
    if (g.reported & KMSG_INCOMPLETE_BIT) return;
    g.reported |= KMSG_INCOMPLETE_BIT;
    char slb[32] = "", reg[32] = "";
    if (g.levels & ((1u << ONCHIP_LV_SLB) | (1u << ONCHIP_LV_L1)))
      snprintf(slb, sizeof slb, " SLB %u/%d", r->arrived, g.slbGrid);
    if (g.levels & (1u << ONCHIP_LV_REG))
      snprintf(reg, sizeof reg, " REG %u/%d", r->arrivedReg, g.regGrid);
    snprintf(msg, sizeof msg,
             "worker_health_check INCOMPLETE:%s%s (%s, GPU probably shared)",
             slb, reg,
             r->coverageIncomplete ? "rendezvous timeout"
                                   : "rendezvous uncleared");
  }
  xid_report(msg);
}

extern "C" const char *onchip_check_last_error(void) { return g_err; }

extern "C" const char *onchip_level_name(int level) {
  switch (level) {
  case ONCHIP_LV_SLB: return "SLB(shared)";
  case ONCHIP_LV_REG: return "REG";
  case ONCHIP_LV_L1: return "L1";
  case ONCHIP_LV_L2: return "L2";
  default: return "?";
  }
}

// Register kernels have to be instantiated at compile time, so a set of
// candidate widths is built and init picks the widest one the compiler can hold
// entirely in registers, rejecting any that spills. Holding a value costs 1.07
// registers here, so the ceiling sits just under the 256 a lane has: 239 fits
// with all 256 allocated, 240 spills. The dense entries near the top are where
// the decision actually gets made; the ones below are fallbacks for a compiler
// that needs more per value, and 256 is kept for one that needs less. Which
// widths fit is not monotonic in N, so the list is probed rather than derived.
typedef void (*RegKernel)(OnchipResult *, unsigned, unsigned long long, Inject,
                          unsigned long long);
static const int kRegN[] = {32,  64,  128, 192, 224,
                            232, 236, 239, 240, 256};
static const RegKernel kRegK[] = {
    k_reg<32>,  k_reg<64>,  k_reg<128>, k_reg<192>, k_reg<224>,
    k_reg<232>, k_reg<236>, k_reg<239>, k_reg<240>, k_reg<256>};
#define REG_CFGS (int)(sizeof(kRegN) / sizeof(kRegN[0]))

extern "C" int onchip_check_init(unsigned levelMask) {
  if (g.ready) return ONCHIP_OK;
  g.levels = levelMask ? levelMask : 0xFu;

  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  g.nsm = p.multiProcessorCount;
  // Rendezvous budget. Derived from the device rather than hardcoded, but the
  // arithmetic does not give the 1 ms it looks like: clock64() ticks at the
  // core clock, measured at ~1.5 GHz here, while cudaDeviceProp reports 500
  // MHz, so this is roughly 0.3 ms. That is still far longer than a resident
  // grid needs to meet, and short enough not to hang, which is all the barrier
  // requires of it.
  g.timeout = (unsigned long long)p.clockRate;
  g.regHold = 0;
  g.dev = dev;

  const unsigned char *u = (const unsigned char *)p.uuid.bytes;
  snprintf(g.ident, sizeof g.ident,
           "%04x:%02x:%02x.0 GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
           "%02x%02x%02x%02x%02x%02x", p.pciDomainID, p.pciBusID, p.pciDeviceID,
           u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10],
           u[11], u[12], u[13], u[14], u[15]);
  // Only ever turns reporting on, so an earlier onchip_check_set_kmsg(1) is not
  // undone by an unset environment.
  const char *kenv = getenv("ONCHIP_CHECK_KMSG");
  if (kenv && *kenv && strcmp(kenv, "0") != 0) g.kmsg = 1u;

  // ---- SLB sizing: largest power-of-two block size, chosen by asking the
  // driver how many such blocks stay resident, so the grid covers the whole SLB.
  // The full 4096, against the platform's general advice to write 1024-thread
  // CTAs: that advice is about occupancy, and occupancy here is pinned at one
  // block per CU by the shared memory claim either way. Measured, 4096 sweeps
  // the SLB in 31 us against 52 us for 1024, for identical coverage.
  g.slbThreads = p.maxThreadsPerBlock;
  CK(cudaFuncSetAttribute((const void *)k_slb_l1,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)p.sharedMemPerBlock));
  size_t perCU = p.sharedMemPerMultiprocessor ? p.sharedMemPerMultiprocessor
                                              : p.sharedMemPerBlock;
  size_t bestSz = 0, bestCover = 0;
  int bestBlocks = 0;
  size_t top = 4096;
  while (top * 2 <= p.sharedMemPerBlock) top *= 2;
  for (size_t sz = top; sz >= 4096; sz /= 2) {
    int b = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &b, (const void *)k_slb_l1, g.slbThreads, sz) != cudaSuccess)
      continue;
    if (b < 1) continue;
    size_t cover = (size_t)b * sz;
    if (cover > perCU) cover = perCU;
    if (cover > bestCover) { bestCover = cover; bestSz = sz; bestBlocks = b; }
    if (bestCover >= perCU) break;
  }
  if (!bestSz) {
    snprintf(g_err, sizeof(g_err), "no usable shared memory configuration");
    return ONCHIP_CUDA_ERROR;
  }
  g.slbBytes = bestSz;
  g.slbGrid = g.nsm * bestBlocks;

  // ---- register sizing: what is testable is (resident threads) x (registers
  // held per thread), and the two trade off against each other, so both are
  // swept and the driver decides which combinations are actually resident.
  // Fixing the thread count first would have left most of the file untested.
  g.regCapacity = (size_t)p.regsPerBlock * 4 * g.nsm;

  // A candidate counts as covering N registers per thread only if the compiler
  // actually kept all N in registers. Occupancy cannot establish that: spilling
  // lowers the register count, so a spilled kernel reports the same or better
  // residency and would still win the sweep below, quietly turning part of this
  // level into a local-memory test while the reported coverage stayed at 100%.
  // The attributes do not depend on the block size, so they are read once per
  // candidate rather than inside the sweep.
  int regFits[REG_CFGS];
  int regRegs[REG_CFGS];
  size_t regLocal[REG_CFGS];
  for (int i = 0; i < REG_CFGS; ++i) {
    cudaFuncAttributes fa;
    regFits[i] = 0;
    regRegs[i] = 0;
    regLocal[i] = 0;
    if (cudaFuncGetAttributes(&fa, (const void *)kRegK[i]) != cudaSuccess)
      continue;
    regRegs[i] = fa.numRegs;
    regLocal[i] = fa.localSizeBytes;
    regFits[i] = (fa.localSizeBytes == 0 && fa.numRegs >= kRegN[i]) ? 1 : 0;
  }

  size_t bestRegCover = 0;
  for (int t = 64; t <= p.maxThreadsPerBlock; t *= 2) {
    for (int i = 0; i < REG_CFGS; ++i) {
      if (!regFits[i]) continue;
      int b = 0;
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &b, (const void *)kRegK[i], t, 0) != cudaSuccess)
        continue;
      if (b < 1) continue;
      if (b * t > p.maxThreadsPerMultiProcessor)
        b = p.maxThreadsPerMultiProcessor / t;
      size_t cover = (size_t)b * t * kRegN[i] * 4;
      if (cover > g.regCapacity / g.nsm) cover = g.regCapacity / g.nsm;
      // On ties prefer the wider block: same coverage with far fewer blocks and
      // a much lower rendezvous cost.
      if (cover >= bestRegCover) {
        bestRegCover = cover;
        g.regN = kRegN[i];
        g.regThreads = t;
        g.regGrid = g.nsm * b;
        g.regNumRegs = regRegs[i];
        g.regLocalBytes = regLocal[i];
      }
    }
  }
  if (!g.regN) {
    // Every candidate spilled or could not be queried. Fall back to the
    // narrowest one and report its attributes rather than a coverage figure
    // nothing backs: regLocalBytes != 0 in the config is the caller's signal
    // that the register verdict is weaker than the others.
    g.regN = kRegN[0];
    g.regThreads = 256;
    g.regGrid = g.nsm;
    g.regNumRegs = regRegs[0];
    g.regLocalBytes = regLocal[0];
  }

  // ---- prefer the named-register waves, which cover the whole lane rather than
  // the part the allocator agreed to hold. All four have to be spill-free: a
  // window that spills would be reading its values back from local memory, and
  // the point of naming them was to know exactly which registers were tested.
  {
    const void *waves[4] = {(const void *)k_reg_w0, (const void *)k_reg_w1,
                            (const void *)k_reg_w2, (const void *)k_reg_w3};
    int usable = 1, maxRegs = 0;
    size_t spill = 0;
    for (int i = 0; i < 4; ++i) {
      cudaFuncAttributes fa;
      if (cudaFuncGetAttributes(&fa, waves[i]) != cudaSuccess) { usable = 0; break; }
      if (fa.localSizeBytes != 0) usable = 0;
      if (fa.numRegs > maxRegs) maxRegs = fa.numRegs;
      spill += fa.localSizeBytes;
    }
    // Residency is what makes the coverage per CU rather than per block, so the
    // widest block that still fits is picked exactly as above.
    int bestT = 0, bestB = 0;
    if (usable) {
      for (int t = 64; t <= p.maxThreadsPerBlock; t *= 2) {
        int b = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &b, (const void *)k_reg_w0, t, 0) != cudaSuccess)
          continue;
        if (b < 1) continue;
        if (b * t > p.maxThreadsPerMultiProcessor)
          b = p.maxThreadsPerMultiProcessor / t;
        if (b * t >= bestB * bestT) { bestT = t; bestB = b; }
      }
      if (!bestT) usable = 0;
    }
    if (usable) {
      g.regPinned = true;
      g.regN = ONCHIP_REG_LANE;   // every register of the lane, by name
      g.regThreads = bestT;
      g.regGrid = g.nsm * bestB;
      g.regNumRegs = maxRegs;
      g.regLocalBytes = spill;
    }
  }

  // ---- L1: a private slice per CU, sized to the measured 64KB L1 of ivcore11.
  g.l1Words = 64 * 1024 / 4;
  CK(cudaMalloc(&g.l1buf, (size_t)g.l1Words * 4 * g.slbGrid));

  // ---- L2: a working set the size of the cache, so every line is touched.
  size_t l2Bytes = p.l2CacheSize > 0 ? (size_t)p.l2CacheSize : (16u << 20);
  l2Bytes &= ~(size_t)4095;
  g.l2Words = l2Bytes / 4;
  CK(cudaMalloc(&g.l2buf, l2Bytes));
  g.l2Threads = 256;
  g.l2Grid = g.nsm * 8;

  CK(cudaHostAlloc(&g.pinned, sizeof(OnchipResult), cudaHostAllocDefault));
  CK(cudaMalloc(&g.device, sizeof(OnchipResult)));
  memset(g.pinned, 0, sizeof(OnchipResult));

  g.ready = true;

  // Warm up so the first real call pays no module load cost.
  OnchipResult tmp;
  int rc = onchip_check_run(&tmp, 1u, 0);
  if (rc == ONCHIP_CUDA_ERROR) { g.ready = false; return rc; }
  return ONCHIP_OK;
}

extern "C" int onchip_check_config(OnchipConfig *c) {
  if (!g.ready) return ONCHIP_NOT_READY;
  c->cuCount = g.nsm;
  c->slbGrid = g.slbGrid;
  c->slbThreads = g.slbThreads;
  c->slbBytesPerBlock = g.slbBytes;
  c->slbBytesTotal = (g.levels & (1u << ONCHIP_LV_SLB))
                         ? (size_t)g.slbGrid * g.slbBytes
                         : 0;
  c->regGrid = g.regGrid;
  c->regThreads = g.regThreads;
  c->regPerThread = g.regN;
  c->regBytesTotal = (g.levels & (1u << ONCHIP_LV_REG))
                         ? (size_t)g.regGrid * g.regThreads * g.regN * 4
                         : 0;
  if (c->regBytesTotal > g.regCapacity) c->regBytesTotal = g.regCapacity;
  c->regBytesCapacity = g.regCapacity;
  c->regNumRegs = g.regNumRegs;
  c->regLocalBytes = g.regLocalBytes;
  c->l1BytesTotal = (g.levels & (1u << ONCHIP_LV_L1))
                        ? (size_t)g.l1Words * 4 * g.slbGrid
                        : 0;
  c->l2BytesTotal = (g.levels & (1u << ONCHIP_LV_L2)) ? g.l2Words * 4 : 0;
  return ONCHIP_OK;
}

static int enqueue(unsigned seed, Inject inj, cudaStream_t stream) {
  if (!g.ready) return ONCHIP_NOT_READY;
  CK(cudaMemsetAsync(g.device, 0, sizeof(OnchipResult), stream));

  if (g.levels & ((1u << ONCHIP_LV_SLB) | (1u << ONCHIP_LV_L1))) {
    int doSLB = (g.levels & (1u << ONCHIP_LV_SLB)) ? 1 : 0;
    int doL1 = (g.levels & (1u << ONCHIP_LV_L1)) ? 1 : 0;
    k_slb_l1<<<g.slbGrid, g.slbThreads, g.slbBytes, stream>>>(
        g.device, (unsigned)(g.slbBytes / 4), seed, inj, g.timeout,
        g.l1buf, g.l1Words, doSLB, doL1);
    CK(cudaGetLastError());
  }
  if (g.levels & (1u << ONCHIP_LV_REG)) {
    if (g.regPinned) {
      // Four windows of 64 named registers, in order on one stream: they share
      // the rendezvous counter and each waits for its own multiple of it.
      k_reg_w0<<<g.regGrid, g.regThreads, 0, stream>>>(g.device, seed, g.regHold,
                                                       inj, g.timeout);
      CK(cudaGetLastError());
      k_reg_w1<<<g.regGrid, g.regThreads, 0, stream>>>(g.device, seed, g.regHold,
                                                       inj, g.timeout);
      CK(cudaGetLastError());
      k_reg_w2<<<g.regGrid, g.regThreads, 0, stream>>>(g.device, seed, g.regHold,
                                                       inj, g.timeout);
      CK(cudaGetLastError());
      k_reg_w3<<<g.regGrid, g.regThreads, 0, stream>>>(g.device, seed, g.regHold,
                                                       inj, g.timeout);
      CK(cudaGetLastError());
    } else {
      for (int i = 0; i < REG_CFGS; ++i) {
        if (kRegN[i] != g.regN) continue;
        kRegK[i]<<<g.regGrid, g.regThreads, 0, stream>>>(
            g.device, seed, g.regHold, inj, g.timeout);
        CK(cudaGetLastError());
      }
    }
  }
  if (g.levels & (1u << ONCHIP_LV_L2)) {
    k_l2<<<g.l2Grid, g.l2Threads, 0, stream>>>(g.device, g.l2buf, g.l2Words,
                                               seed, inj);
    CK(cudaGetLastError());
  }

  CK(cudaMemcpyAsync(g.pinned, g.device, sizeof(OnchipResult),
                     cudaMemcpyDeviceToHost, stream));
  return ONCHIP_OK;
}

static int verdict(OnchipResult *out) {
  *out = *g.pinned;
  unsigned tot = 0;
  for (int i = 0; i < ONCHIP_LV_COUNT; ++i) tot += out->errors[i];
  int rc = ONCHIP_OK;
  if (tot)
    rc = ONCHIP_FAULT;
  else if (out->coverageIncomplete)
    rc = ONCHIP_INCOMPLETE;
  else if ((g.levels & ((1u << ONCHIP_LV_SLB) | (1u << ONCHIP_LV_L1))) &&
           out->peakResident != (unsigned)g.slbGrid)
    rc = ONCHIP_INCOMPLETE;
  else if ((g.levels & (1u << ONCHIP_LV_REG)) &&
           out->peakResidentReg != (unsigned)g.regGrid)
    rc = ONCHIP_INCOMPLETE;
  if (rc != ONCHIP_OK) report_kmsg(rc, out);
  return rc;
}

extern "C" int onchip_check_run(OnchipResult *out, unsigned seed,
                                cudaStream_t stream) {
  Inject none = {0u, 0u, 0ull};
  int rc = enqueue(seed, none, stream);
  if (rc != ONCHIP_OK) return rc;
  CK(cudaStreamSynchronize(stream));
  return verdict(out);
}

extern "C" int onchip_check_launch(unsigned seed, cudaStream_t stream) {
  Inject none = {0u, 0u, 0ull};
  return enqueue(seed, none, stream);
}

extern "C" int onchip_check_collect(OnchipResult *out, cudaStream_t stream) {
  if (!g.ready) return ONCHIP_NOT_READY;
  CK(cudaStreamSynchronize(stream));
  return verdict(out);
}

extern "C" int onchip_check_run_inject_mode(OnchipResult *out, unsigned seed,
                                            unsigned mode, unsigned bit,
                                            unsigned long long index,
                                            cudaStream_t stream) {
  Inject inj = {mode, bit, index};
  int rc = enqueue(seed, inj, stream);
  if (rc != ONCHIP_OK) return rc;
  CK(cudaStreamSynchronize(stream));
  *out = *g.pinned;
  unsigned tot = 0;
  for (int i = 0; i < ONCHIP_LV_COUNT; ++i) tot += out->errors[i];
  return tot ? ONCHIP_FAULT : ONCHIP_OK;
}

extern "C" int onchip_check_run_inject(OnchipResult *out, unsigned seed,
                                       cudaStream_t stream) {
  return onchip_check_run_inject_mode(out, seed, 1u, 3u, 12345ull, stream);
}

extern "C" void onchip_check_free(void) {
  if (g.pinned) cudaFreeHost(g.pinned);
  if (g.device) cudaFree(g.device);
  if (g.l1buf) cudaFree(g.l1buf);
  if (g.l2buf) cudaFree(g.l2buf);
  // Reporting is a caller preference, not session state, so it survives the
  // reset while the deduplication bitmask does not. A multi-device sweep is
  // one init/free per device around a single set_kmsg call, and clearing the
  // switch here would silence every device after the first.
  unsigned kmsg = g.kmsg;
  g = Ctx();
  g.kmsg = kmsg;
}

// ==========================================================================
// Validator
// ==========================================================================
//
// A single-point injection only proves the reporting path works; a test that
// happened to check one word would pass it. The decisive check here is to
// corrupt EVERY location and confirm the reported error count matches, to the
// exact unit, the number of locations that should have been read back. Anything
// skipped shows up as a shortfall.

static int v_fail = 0;
static int v_verbose = 1;

static void v_expect(const char *what, unsigned long long got,
                     unsigned long long want) {
  bool ok = (got == want);
  if (!ok) v_fail++;
  if (v_verbose || !ok)
    printf("    %-46s %12llu %s %llu\n", what, got, ok ? "==" : "!= EXPECTED",
           want);
}

extern "C" int onchip_check_validate(int verbose) {
  v_fail = 0;
  v_verbose = verbose;
  if (!g.ready) {
    int rc = onchip_check_init(0);
    if (rc != ONCHIP_OK) return -1;
  }

  OnchipConfig c;
  onchip_check_config(&c);
  cudaStream_t s;
  if (cudaStreamCreate(&s) != cudaSuccess) return -1;
  OnchipResult r;

  const unsigned long long slbWords = c.slbBytesTotal / 4;
  const unsigned long long regWords = c.regBytesTotal / 4;
  const unsigned long long l1Words = c.l1BytesTotal / 4;
  const unsigned long long l2Words = c.l2BytesTotal / 4;

  if (verbose) {
    printf("  locations under test: SLB=%llu REG=%llu L1=%llu L2=%llu words\n",
           slbWords, regWords, l1Words, l2Words);
    printf("  register kernel: %d registers per thread, %zu bytes spilled\n\n",
           c.regNumRegs, c.regLocalBytes);
  }

  // ---- 1. a clean run must be silent
  if (verbose) printf("[1] clean run reports nothing\n");
  int rc = onchip_check_run(&r, 0xABCDu, s);
  v_expect("return code (0 = OK)", (unsigned)rc, 0);
  // Anything spilled is held in local memory, so it would be verified against
  // DRAM rather than the register file while REG coverage still read 100%. The
  // sizing loop rejects such a kernel, which leaves this as the assertion that
  // it did.
  v_expect("register spill (bytes/thread, must be 0)", c.regLocalBytes, 0);
  for (int i = 0; i < ONCHIP_LV_COUNT; ++i) {
    char buf[64];
    snprintf(buf, sizeof buf, "%s errors", onchip_level_name(i));
    v_expect(buf, r.errors[i], 0);
  }
  v_expect("blocks meeting the coverage rendezvous", r.peakResident,
           (unsigned)c.slbGrid);

  // ---- 2. every location corrupted: the count proves nothing is skipped
  if (verbose)
    printf("\n[2] corrupt EVERY location -> error count must equal "
           "locations x read-back passes\n");
  rc = onchip_check_run_inject_mode(&r, 0xABCDu, 2, 5, 0, s);
  v_expect("return code (1 = FAULT)", (unsigned)rc, 1);
  v_expect("SLB errors  (words x 3 passes)", r.errors[ONCHIP_LV_SLB],
           slbWords * ONCHIP_PASSES_SLB);
  v_expect("REG errors  (registers x 2 passes)", r.errors[ONCHIP_LV_REG],
           regWords * ONCHIP_PASSES_REG);
  v_expect("L1 errors   (words x 4 passes)", r.errors[ONCHIP_LV_L1],
           l1Words * ONCHIP_PASSES_L1);
  v_expect("L2 errors   (words x 2 passes)", r.errors[ONCHIP_LV_L2],
           l2Words * ONCHIP_PASSES_L2);

  // ---- 3. every bit position must be checked, not just some
  if (verbose)
    printf("\n[3] single-bit corruption at each of the 32 bit positions\n");
  int bitFail = 0;
  for (unsigned bit = 0; bit < 32; ++bit) {
    rc = onchip_check_run_inject_mode(&r, 0xABCDu, 2, bit, 0, s);
    bool ok = rc == ONCHIP_FAULT &&
              r.errors[ONCHIP_LV_SLB] == slbWords * ONCHIP_PASSES_SLB &&
              r.errors[ONCHIP_LV_REG] == regWords * ONCHIP_PASSES_REG &&
              r.errors[ONCHIP_LV_L1] == l1Words * ONCHIP_PASSES_L1 &&
              r.errors[ONCHIP_LV_L2] == l2Words * ONCHIP_PASSES_L2;
    if (!ok) { bitFail++; printf("    bit %2u NOT fully detected\n", bit); }
  }
  v_expect("bit positions with incomplete detection", (unsigned)bitFail, 0);

  // ---- 4. individual locations, including the boundaries
  if (verbose)
    printf("\n[4] single-location corruption, sampled across the whole range\n");
  unsigned long long probes[] = {0, 1, 7, 63, 1023, 12345, 32767,
                                 slbWords / 2, slbWords - 1};
  int locFail = 0;
  for (unsigned k = 0; k < sizeof(probes) / sizeof(probes[0]); ++k) {
    unsigned long long idx = probes[k];
    if (idx >= slbWords) continue;
    rc = onchip_check_run_inject_mode(&r, 0xABCDu, 1, 11, idx, s);
    unsigned tot = 0;
    for (int i = 0; i < ONCHIP_LV_COUNT; ++i) tot += r.errors[i];
    if (!(rc == ONCHIP_FAULT && tot > 0)) {
      locFail++;
      printf("    word %llu NOT detected\n", idx);
    }
  }
  v_expect("sampled locations missed", (unsigned)locFail, 0);

  // ---- 5. no false positives after a fault run
  if (verbose) printf("\n[5] clean run after the fault runs is clean again\n");
  rc = onchip_check_run(&r, 0x1234u, s);
  v_expect("return code (0 = OK)", (unsigned)rc, 0);
  unsigned tot = 0;
  for (int i = 0; i < ONCHIP_LV_COUNT; ++i) tot += r.errors[i];
  v_expect("total errors", tot, 0);

  // ---- 6. repeated clean runs must be stable, not intermittently noisy
  if (verbose) printf("\n[6] 200 consecutive clean runs\n");
  unsigned spurious = 0, incomplete = 0;
  for (int i = 0; i < 200; ++i) {
    rc = onchip_check_run(&r, 0x9000u + i, s);
    if (rc == ONCHIP_FAULT) spurious++;
    if (rc == ONCHIP_INCOMPLETE) incomplete++;
  }
  v_expect("runs reporting a spurious fault", spurious, 0);
  v_expect("runs reporting incomplete coverage", incomplete, 0);

  cudaStreamDestroy(s);
  return v_fail;
}

#ifdef ONCHIP_CHECK_STANDALONE
int main(int argc, char **argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  if (cudaSetDevice(dev) != cudaSuccess) {
    printf("no device %d\n", dev);
    return 2;
  }
  if (onchip_check_init(0) != ONCHIP_OK) {
    printf("init failed: %s\n", onchip_check_last_error());
    return 2;
  }
  OnchipConfig c;
  onchip_check_config(&c);
  printf("device %d, CU=%d\n", dev, c.cuCount);
  printf("  SLB: %d blocks x %d thr x %zuKB -> %zuKB covered\n", c.slbGrid,
         c.slbThreads, c.slbBytesPerBlock / 1024, c.slbBytesTotal / 1024);
  printf("  REG: %d blocks x %d thr x %d regs -> %zuKB of %zuKB (%.0f%%)\n",
         c.regGrid, c.regThreads, c.regPerThread, c.regBytesTotal / 1024,
         c.regBytesCapacity / 1024,
         100.0 * c.regBytesTotal / (double)c.regBytesCapacity);
  printf("  L1 : %zuKB covered\n", c.l1BytesTotal / 1024);
  printf("  L2 : %zuKB covered\n\n", c.l2BytesTotal / 1024);

  int failed = onchip_check_validate(1);
  printf("\n%s  (%d check%s failed)\n",
         failed ? "VALIDATION FAILED" : "VALIDATION PASSED", failed,
         failed == 1 ? "" : "s");
  onchip_check_free();
  return failed ? 1 : 0;
}
#endif  // ONCHIP_CHECK_STANDALONE

#endif  // ONCHIP_CHECK_CU
