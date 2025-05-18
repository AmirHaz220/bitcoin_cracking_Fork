import argparse
import json
import os
import time
from itertools import product

import numpy as np
import pyopencl as cl
from mnemonic import Mnemonic

mnemo = Mnemonic("english")

repeater_workers = 1
local_workers = 256
global_workers = 512
global_workers -= global_workers % local_workers
TW = (global_workers,)
TT = (local_workers,)


def build_message_words(passphrase: str):
    salt = b"mnemonic" + passphrase.encode()
    msg = salt + b"\x00\x00\x00\x01"
    total_len = 128 + len(msg)
    pad_len = (112 - total_len % 128) % 128
    padding = b"\x80" + b"\x00" * (pad_len - 1)
    bit_len = (total_len) * 8
    length_field = bit_len.to_bytes(16, "big")
    final = msg + padding + length_field
    return [int.from_bytes(final[i : i + 8], "big") for i in range(0, len(final), 8)]


def load_candidates(anchors_path: str, unknowns_path: str):
    slots = [None] * 24
    if anchors_path:
        with open(anchors_path, "r") as f:
            data = json.load(f)
        for item in data.get("anchors", []):
            idx = item["index"] - 1
            if "word" in item:
                slots[idx] = [item["word"]]
            elif "guess" in item:
                slots[idx] = item["guess"]
    if unknowns_path and os.path.exists(unknowns_path):
        with open(unknowns_path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            slots[int(k) - 1] = v
    for i in range(24):
        if slots[i] is None:
            slots[i] = mnemo.wordlist
    return slots


def candidate_mnemonics(slots):
    for combo in product(*slots):
        yield list(combo)


def words_to_indices(words):
    return np.array([mnemo.wordlist.index(w) for w in words], dtype=np.int32)


def mnemonic_to_uint64_pair(indices):
    binary_string = "".join(f"{i:011b}" for i in indices)[:-4]
    binary_string = binary_string.ljust(128, "0")
    high = int(binary_string[:64], 2)
    low = int(binary_string[64:], 2)
    return high, low


def run_kernel(program, queue, words, msg_words):
    context = program.context
    kernel = program.verify
    elements = global_workers * 12000
    buf_size = elements * 8

    indices = words_to_indices(words)
    high, low = mnemonic_to_uint64_pair(indices)

    high_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([high], dtype=np.uint64))
    low_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([low], dtype=np.uint64))
    msg_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(msg_words, dtype=np.uint64))
    p_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array([1], dtype=np.uint32))
    output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, buf_size)

    start = time.perf_counter()
    kernel.set_args(p_buf, high_buf, low_buf, msg_buf, output_buf)
    event = cl.enqueue_nd_range_kernel(queue, kernel, TW, TT)
    event.wait()
    exec_time = (event.profile.end - event.profile.start) * 1e-6
    total = global_workers / (time.perf_counter() - start)

    result = np.empty(elements, dtype=np.uint64)
    cl.enqueue_copy(queue, result, output_buf).wait()

    print(f"Kernel {exec_time:.3f} ms, {total:.2f} ops/s")


def build_program(context, *filenames):
    src = "".join(open(f).read() + "\n\n" for f in filenames)
    return cl.Program(context, src).build()


def load_wallets(path="wallets.tsv"):
    if not os.path.exists(path):
        print("wallets.tsv not found, skipping wallet loading")
        return {}
    memoria = {}
    with open(path, "r") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue
            try:
                addr, saldo = linha.split()
                memoria[addr] = float(saldo)
            except ValueError:
                continue
    return memoria


def main():
    parser = argparse.ArgumentParser(description="BIP-39 brute force demo")
    parser.add_argument("--anchors", help="anchors JSON file", default=None)
    parser.add_argument("--unknowns", help="unknowns JSON file", default=None)
    parser.add_argument("--passphrase", help="bip39 passphrase", default="")
    args = parser.parse_args()

    slots = load_candidates(args.anchors, args.unknowns)
    msg_words = build_message_words(args.passphrase)

    try:
        platforms = cl.get_platforms()
        devices = platforms[0].get_devices()
        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        program = build_program(context, "./kernel/main.cl")
        for words in candidate_mnemonics(slots):
            run_kernel(program, queue, words, msg_words)
            break  # demo runs first combination only
    except Exception as e:
        print(f"Erro ao executar kernel: {e}")
        return


if __name__ == "__main__":
    main()
