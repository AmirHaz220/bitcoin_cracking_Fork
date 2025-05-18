#include "kernel/bip39.cl"
#include "kernel/common.cl"
#include "kernel/ec.cl"
#include "kernel/sha256.cl"
#include "kernel/sha512.cl"

__kernel void verify(__global uint *mnemonicWords, __global ulong *H,
                     __global ulong *L, __global ulong *msg,
                     __global ulong *output) {
  int gid = get_global_id(0);
  if (gid > 0)
    return;

  ulong inner_data[32] = {0};
  ulong outer_data[32] = {0};
  ulong hmacSeedOutput[8] = {0};

  ulong memHigh = H[0];
  ulong firstMem = L[0];
  ulong memLow = firstMem + gid;

  ulong mnemonicLong[16] = {0};
  ulong pbkdLong[16] = {0};
  uint seedNum[16] = {0};
  uchar mnemonicString[128] = {0};

  uint offset = 0;
  prepareSeedNumber(seedNum, memHigh, memLow);
  prepareSeedString(seedNum, mnemonicString, offset);
  ucharLong(mnemonicString, offset - 1, mnemonicLong, 0);

  #pragma unroll
  for (int lid = 0; lid < 16; lid++) {
    inner_data[lid] = mnemonicLong[lid] ^ IPAD;
    outer_data[lid] = mnemonicLong[lid] ^ OPAD;
  }

  for (int i = 0; i < 16; i++) {
    inner_data[16 + i] = msg[i];
    outer_data[16 + i] = msg[i];
  }

  pbkdf2_hmac_sha512_long(inner_data, outer_data, pbkdLong);
  hmac_sha512_bitcoin_seed(pbkdLong, hmacSeedOutput);

#ifdef DEBUG_PRINTF
  printf("Seed: \"%s\"\n"
         "PBKDF2: \"%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx\"\n"
         "HMAC  : \"%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx\"\n",
         mnemonicString, SHOW_ARR(pbkdLong), SHOW_ARR(hmacSeedOutput));
#endif
}
