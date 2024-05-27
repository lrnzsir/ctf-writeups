from mersenne_crack import gen_system, test_system, get_random_from_outs, MersenneTwister
from pwn import remote, process, context
from Crypto.Cipher import AES
from enum import Enum
import random, json, re, os

nleaks = 8
nqueries = 3198
assert nqueries < 3200
assert nqueries * nleaks >= MersenneTwister.N * 32
if not os.path.exists(f"matrix-{nqueries}-{nleaks}.png"):
    gen_system(nqueries, nleaks)

API_CHOICE = Enum("API_CHOICE", ["UploadCircuit", "DisplayCircuit", "PerformMeasurement", "Exit"])


def generate_noise_json() -> dict:
    circuit = {"gates": []}
    controls = random.sample(range(8), k = 4)
    dependents = set(range(8)) - set(controls)
    for c in controls:
        circuit["gates"].append(f"H {c}")
        coupled = random.sample(list(dependents), k=1)
        dependents -= set(coupled)
        for cx in coupled:
            circuit["gates"].append(f"CX {c} {cx}")
    return circuit


def query_mesurement(nqueries: int, timeout=0.01) -> list[int]:
    global io, API_CHOICE
    io.send(f"{API_CHOICE.PerformMeasurement.value}\n".encode() * nqueries)
    data = io.recvrepeat(timeout)
    while len(re.findall(r"[01]{8}".encode(), data)) < nqueries:
        data += io.recvrepeat(timeout)
    return [int(x, 2) for x in re.findall(r"[01]{8}".encode(), data)]


while True:
    uploaded_circuit = generate_noise_json()

    context.log_level = "info"
    # io = process(["python3", "quantum.py"])
    io = remote("193.148.168.30", 5666)

    io.sendlineafter(b"Choice: ", str(API_CHOICE.UploadCircuit.value).encode())
    io.sendlineafter(b"Enter circuit json:", json.dumps(uploaded_circuit).encode())

    outs = query_mesurement(256)
    if len(set(outs)) > 128:
        break
    io.close()

outs.extend(query_mesurement(nqueries - 256))

io.sendline(str(API_CHOICE.Exit.value).encode())
io.recvline_contains(b"Flag:")
ct = bytes.fromhex(io.recvline_contains(b"ct:").split(b" ").pop().decode())
iv = bytes.fromhex(io.recvline_contains(b"iv:").split(b" ").pop().decode())

rng = get_random_from_outs(nqueries, nleaks, outs, verbose=True)

key = sum(rng.genrand_int32() << (32 * i) for i in range(4)).to_bytes(16, "little")
iv_ = sum(rng.genrand_int32() << (32 * i) for i in range(4)).to_bytes(16, "little")
assert iv == iv_

pt = AES.new(key, mode=AES.MODE_CBC, iv=iv).decrypt(ct)
print(pt)
