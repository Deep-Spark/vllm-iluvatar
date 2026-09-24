// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ONCHIP_OK = 0,
  ONCHIP_FAULT = 1,
  ONCHIP_INCOMPLETE = 2,
  ONCHIP_NOT_READY = 3,
  ONCHIP_CUDA_ERROR = 4,
};

enum {
  ONCHIP_LV_SLB = 0,
  ONCHIP_LV_REG,
  ONCHIP_LV_L1,
  ONCHIP_LV_L2,
  ONCHIP_LV_COUNT,
};

typedef struct {
  unsigned errors[ONCHIP_LV_COUNT];
  unsigned firstLevel;
  unsigned firstBlock;
  unsigned firstThread;
  unsigned firstWord;
  unsigned firstExpected;
  unsigned firstGot;
  unsigned firstPhase;
  unsigned claimed;
  unsigned arrived;
  unsigned arrivedReg;
  unsigned peakResident;
  unsigned peakResidentReg;
  unsigned coverageIncomplete;
} OnchipResult;

typedef struct {
  int cuCount;
  int slbGrid, slbThreads;
  size_t slbBytesPerBlock, slbBytesTotal;
  int regGrid, regThreads, regPerThread;
  size_t regBytesTotal, regBytesCapacity;
  size_t l1BytesTotal, l2BytesTotal;
  int regNumRegs;
  size_t regLocalBytes;
} OnchipConfig;

int onchip_check_init(unsigned levelMask);
int onchip_check_config(OnchipConfig *config);
int onchip_check_run(OnchipResult *result, unsigned seed,
                     cudaStream_t stream);
void onchip_check_set_kmsg(int enable);
int onchip_check_kmsg_ready(void);
int onchip_check_report_xid(unsigned cardid, unsigned xid, const char *message);
void onchip_check_free(void);
const char *onchip_check_last_error(void);
const char *onchip_level_name(int level);

#ifdef __cplusplus
}
#endif
