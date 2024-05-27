from sage.all import BooleanPolynomialRing, vector, GF, Matrix
from sage.matrix.matrix_mod2_dense import from_png, to_png
from typing import Any
import tqdm, random, os


class MersenneTwister:
    N = 624
    M = 397
    MATRIX_A = 0x9908b0df
    MASK = 0xffffffff
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7fffffff

    def __init__(self, state: list[int]) -> None:
        assert len(state) == self.N, f"Expected {self.N} elements, got {len(state)}"
        self.state = state
        self.index = 0
        self.twist()
    
    def twist(self) -> None:
        for i in range(self.N):
            y = (self.state[i] & self.UPPER_MASK) | \
                (self.state[(i + 1) % self.N] & self.LOWER_MASK)
            self.state[i] = self.state[(i + self.M) % self.N] ^ (y >> 1)
            if y & 1:
                self.state[i] ^= self.MATRIX_A
            self.state[i] &= self.MASK
        self.index = 0
    
    def genrand_int32(self) -> int:
        if self.index >= self.N:
            self.twist()
        
        y = self.state[self.index]
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= y >> 18

        self.index += 1
        return y

    @classmethod
    def test_getrandbits(cls) -> None:
        seed = os.urandom(16)
        random.seed(seed)
        
        state = list(random.getstate()[1][:-1])
        mt = MersenneTwister(state)
        
        for i in range(cls.N * 2):
            out = random.getrandbits(32)
            guess = mt.genrand_int32()
            assert guess == out, \
                f"Failed at iteration {i + 1}, with {guess=} != {out=}"
    
    @classmethod
    def test_random(cls) -> None:
        seed = os.urandom(16)
        random.seed(seed)
        
        state = list(random.getstate()[1][:-1])
        mt = MersenneTwister(state)
        
        for i in range(cls.N * 2):
            out = random.random()
            a, b = mt.genrand_int32(), mt.genrand_int32()
            assert a >> 31 == int(out + 0.5), \
                f"Failed at iteration {i + 1}"


class MersenneTwisterSymbolic:
    N = 624
    M = 397
    MATRIX_A = 0x9908b0df

    def __init__(self) -> None:
        self.R = BooleanPolynomialRing(
            self.N * 32, [f"s_{i}_{j}" for i in range(self.N) for j in range(32)]
        )
        self.state = [
            vector(self.R, [self.R.gen(i * 32 + j) for j in range(32)]) 
        for i in range(self.N)]
        self.index = 0
        self.twist()
    
    def twist(self) -> None:
        for i in range(self.N):
            # y = (self.state[i] & 0x80000000) |
            #     (self.state[(i + 1) % self.N] & 0x7fffffff)
            y = vector(
                self.R, [self.state[i][0]] + list(self.state[(i + 1) % self.N])[1:]
            )
            # self.state[i] = self.state[(i + self.M) % self.N] ^ 
            #                 (y >> 1) ^ ((y & 1) * self.MATRIX_A)
            self.state[i] = (
                self.state[(i + self.M) % self.N] + vector(self.R, [0] + list(y)[:-1]) +
                vector(
                    self.R, 
                    [y[-1] if bit else 0 for bit in map(int, f"{self.MATRIX_A:032b}")]
                )
            )
        self.index = 0
    
    def genrand_int32(self) -> Any:
        if self.index >= self.N:
            self.twist()
        
        y = self.state[self.index]
        # y ^= y >> 11
        y += vector(self.R, [0] * 11 + list(y)[:-11])
        # y ^= (y << 7) & 0x9d2c5680
        y0 = vector(self.R, list(y)[7:] + [0] * 7)
        y0 = vector(
            self.R, 
            [y0[i] if bit else 0 for i, bit in enumerate(map(int, f"{0x9d2c5680:032b}"))]
        )
        y += y0
        # y ^= (y << 15) & 0xefc60000
        y0 = vector(self.R, list(y)[15:] + [0] * 15)
        y0 = vector(
            self.R, 
            [y0[i] if bit else 0 for i, bit in enumerate(map(int, f"{0xefc60000:032b}"))]
        )
        y += y0
        # y ^= y >> 18
        y += vector(self.R, [0] * 18 + list(y[:-18]))

        self.index += 1
        return y
    
    @classmethod
    def test_getrandbits(cls) -> None:
        seed = os.urandom(16)
        random.seed(seed)
        
        state = list(random.getstate()[1][:-1])
        mt = MersenneTwisterSymbolic()
        mt.R = GF(2)
        mt.state = [vector(mt.R, list(map(int, f"{state[i]:032b}"))) for i in range(mt.N)]
        mt.twist()
        
        for i in range(mt.N * 2):
            out = random.getrandbits(32)
            guess = int("".join(map(str, mt.genrand_int32())), 2)
            assert guess == out, f"Failed at iteration {i + 1}, with {guess=} != {out=}"
    
    @classmethod
    def test_random(cls) -> None:
        seed = os.urandom(16)
        random.seed(seed)
        
        state = list(random.getstate()[1][:-1])
        mt = MersenneTwisterSymbolic()
        mt.R = GF(2)
        mt.state = [vector(mt.R, list(map(int, f"{state[i]:032b}"))) for i in range(mt.N)]
        mt.twist()
        
        for i in range(mt.N * 2):
            out = random.random()
            a, b = mt.genrand_int32(), mt.genrand_int32()
            assert int(a[0]) == int(out + 0.5), f"Failed at iteration {i + 1}"


def gen_system(nqueries: int, nleaks: int) -> None:
    mt = MersenneTwisterSymbolic()

    print("Generating rows of the system...")
    polys = []
    for _ in tqdm.tqdm(range(nqueries)):
        a, b = mt.genrand_int32(), mt.genrand_int32()
        for poly in a[:nleaks]:
            polys.append(poly)

    print("Generating the coefficient matrix...")
    M = Matrix(GF(2), nqueries * nleaks, len(mt.R.gens()), sparse=False)
    mon_to_id = {mon: i for i, mon in enumerate(mt.R.gens())}

    for i, poly in tqdm.tqdm(enumerate(polys), total=len(polys)):
        for mon in poly.monomials():
            M[i, mon_to_id[mon]] = 1
    
    print("Saving the coefficient matrix...")
    to_png(M, f"matrix-{nqueries}-{nleaks}.png")


def test_system(nqueries: int, nleaks: int) -> None:
    print("Loading the coefficient matrix...")
    M = from_png(f"matrix-{nqueries}-{nleaks}.png")

    print("Generating the right-hand side...")
    seed = os.urandom(16)
    random.seed(seed)

    outs, bits = [], []
    for _ in range(nqueries):
        out = int(random.random() * 2**32)
        outs.append(out >> (32 - nleaks))
        for bit in map(int, f"{out:032b}"[:nleaks]):
            bits.append(bit)

    rhs = vector(GF(2), bits)

    print("Solving the system...")
    sol = M.solve_right(rhs)
    assert M * sol == rhs

    print("Checking the solution...")
    mt_guess = MersenneTwister(
        [int("".join(map(str, sol[i:i + 32])), 2) for i in range(0, len(sol), 32)]
    )

    for out in tqdm.tqdm(outs):
        a, b = mt_guess.genrand_int32(), mt_guess.genrand_int32()
        assert a >> (32 - nleaks) == out

    for _ in tqdm.tqdm(range(1024)):
        a, b = mt_guess.genrand_int32(), mt_guess.genrand_int32()
        _a, _b = random.getrandbits(32), random.getrandbits(32)
        assert a == _a and b == _b


def get_random_from_outs(nqueries: int, nleaks: int, outs: list[int], verbose=False) -> "MersenneTwister":
    assert len(outs) == nqueries, \
        f"Expected {nqueries} outs, got {len(outs)}"
    assert all(0 <= out < 2**nleaks for out in outs), \
        f"Expected {nleaks} bits, got {max(outs).bit_length()}"
    
    if verbose:
        print("Loading the coefficient matrix...")
    M = from_png(f"matrix-{nqueries}-{nleaks}.png")

    bits = []
    for out in outs:
        for bit in map(int, f"{out:0{nleaks}b}"):
            bits.append(bit)
    rhs = vector(GF(2), bits)

    if verbose:
        print("Solving the system...")
    sol = M.solve_right(rhs)
    
    if verbose:
        print("Checking the solution...")
    mt_guess = MersenneTwister(
        [int("".join(map(str, sol[i:i + 32])), 2) for i in range(0, len(sol), 32)]
    )
    
    for out in outs:
        a, b = mt_guess.genrand_int32(), mt_guess.genrand_int32()
        assert a >> (32 - nleaks) == out
    
    return mt_guess
