from typing import Dict, Optional, Tuple, List, TypeAlias
from copy import deepcopy
from itertools import product

import sys

# Define the types for the bits, state, transition, and trellis
Metric: TypeAlias = int
Bit: TypeAlias = int
Bits: TypeAlias = Tuple[Bit, ...]
State: TypeAlias = Bits
Transition: TypeAlias = Tuple[State, Bit]
FSM: TypeAlias = Dict[Transition, Tuple[State, Bits]]
Trellis: TypeAlias = List[Dict[State, Tuple[Metric, Optional[Transition]]]]


def replace_bit(bits: Bits, index: int, value: Bit) -> Bits:
    return bits[:index] + (value,) + bits[index + 1 :]


def generate_diff(seq1: Bits, seq2: Bits) -> str:
    seq1_str = " ".join(str(el) for el in seq1)
    seq2_str = " ".join(str(el) for el in seq2)
    diff_markers = "".join("🗲 " if el1 != el2 else "  " for el1, el2 in zip(seq1, seq2))
    return f"{diff_markers}\n{seq1_str}\n{seq2_str}"


def bits_to_string(bits: Bits) -> str:
    return "".join(str(bit) for bit in bits)


class ConvolutionalEncoder:
    def __init__(self, window_size: int, generators: list[int]) -> None:
        # Define the window size
        self.window_size: int = window_size

        # Define the initial state
        # Initial state is 0...0 (window_size - 1 zeros)
        self.state: State = (0,) * (self.window_size - 1)

        # Check generator polynomials
        # Must have the same length as window_size
        # Each generator must be smaller than 2 ** window_size
        # At least one generator must be greater or equal than 2 ** (window_size - 1)
        if len(generators) != self.window_size:
            raise ValueError(
                f"Number of generators must match window size {self.window_size}, got {len(generators)}"
            )

        if not all(0 <= g < 2**self.window_size for g in generators):
            raise ValueError(
                f"Generators must be between 0 and {2**self.window_size - 1}"
            )

        if not any(w >= 2 ** (self.window_size - 1) for w in generators):
            raise ValueError(
                f"At least one generator must be greater or equal than {2**(self.window_size - 1)}"
            )

        self.generators = sorted(generators, reverse=True)
        self.w = [
            ConvolutionalEncoder.integer_to_bits(w, self.window_size)
            for w in self.generators
        ]

        # Define the state transition and output mapping
        # The key is the (current_state, input_bit)
        # The value is a tuple of (next_state, output_bits)
        self.transitions: FSM = self.generate_transition_table()

    def generate_transition_table(self) -> FSM:
        bits = [0, 1]
        transition_table: FSM = {}
        states: List[State] = self.generate_all_states()

        for state in states:
            for bit in bits:
                next_state = state[1:] + (bit,)
                output_bits = self.calculate_output_bits(state, bit)
                transition_table[(state, bit)] = (next_state, output_bits)

        return transition_table

    def generate_all_states(self) -> List[State]:
        return list(product([0, 1], repeat=self.window_size - 1))

    def calculate_output_bits(self, current_state: State, input_bit: Bit) -> Bits:
        bits: list[Bit] = []
        x: Bits = current_state + (input_bit,)
        x = x[::-1]
        for i in range(self.window_size):
            y: Bit = 0
            for j in range(self.window_size):
                y ^= self.w[i][j] & x[j]
            bits.append(y)

        return tuple(bits)

    @staticmethod
    def integer_to_bits(num: int, length: int) -> Bits:
        return tuple(int(bit) for bit in f"{num:0{length}b}")

    def encode(self, input_bits: Bits) -> Bits:
        output_bits: list[Bit] = []
        for bit in input_bits:
            # Get the next state and output bits from the current state and input bit
            next_state, output = self.transitions[(self.state, bit)]
            output_bits.extend(output)
            # Update the state
            self.state = next_state
        return tuple(output_bits)

    def generate_graphviz(self) -> str:
        graph = "digraph FSM {\n"
        graph += "layout=dot\n"
        graph += 'rankdir="LR"\n'
        graph += 'node [shape="point" label=""] start\n'
        graph += 'node [shape="circle"]\n'
        states = self.generate_all_states()
        for state in states:
            graph += f'"{state}" [label="{bits_to_string(state)}"]\n'
            for bit in [0, 1]:
                next_state, output_bits = self.transitions[(state, bit)]
                graph += f'"{state}" -> "{next_state}" [label="{bit}/{bits_to_string(output_bits)}"]\n'
        graph += f'start -> "{self.state}"\n'
        graph += "}"
        return graph


class ViterbiDecoder:
    def __init__(self, encoder: ConvolutionalEncoder):
        self.encoder: ConvolutionalEncoder = encoder

        # 2 bits for state representation, hence we need (window_size - 1) ^ 2 states
        self.num_states: int = (encoder.window_size - 1) ** 2

    @staticmethod
    def hamming_distance(seq1: Bits, seq2: Bits) -> Metric:
        return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

    def decode(self, received_bits: Bits) -> Bits:
        # Number of encoded bits divided by the window size
        n = len(received_bits) // encoder.window_size
        trellis: Trellis = [{} for _ in range(n + 1)]

        # Initial state with 0 distance
        initial_state: State = (0,) * (encoder.window_size - 1)
        trellis[0][initial_state] = (0, None)

        # Populate the trellis
        for t in range(n):
            for state in trellis[t]:
                for bit in [0, 1]:
                    next_state, output_bits = self.encoder.transitions[(state, bit)]
                    segment_start = encoder.window_size * t
                    segment_end = segment_start + encoder.window_size
                    received_segment = received_bits[segment_start:segment_end]
                    distance = ViterbiDecoder.hamming_distance(
                        output_bits, received_segment
                    )
                    metric = trellis[t][state][0] + distance

                    if (
                        next_state not in trellis[t + 1]
                        or metric < trellis[t + 1][next_state][0]
                    ):
                        trellis[t + 1][next_state] = (metric, (state, bit))

        # Find the state with the smallest metric in the last column of the trellis
        final_state = min(trellis[n], key=lambda x: trellis[n][x][0])
        decoded_bits = []
        current_state = final_state

        # Traceback from the final state to the initial state
        for t in range(n, 0, -1):
            prev_state, bit = trellis[t][current_state][1]  # type: ignore
            decoded_bits.append(bit)
            current_state = prev_state

        decoded_bits.reverse()
        return tuple(decoded_bits)


if __name__ == "__main__":
    encoder = ConvolutionalEncoder(window_size=3, generators=[7,6,5])
    if len(sys.argv) == 2 and sys.argv[1] == "graph":
        print(encoder.generate_graphviz())
        exit()

    input_bits = (1, 1, 0, 1)
    print(f"Input bits:\n{input_bits}")

    encoded_bits = encoder.encode(input_bits)
    print(f"Encoded bits:\n{encoded_bits}")

    # Introduce bit errors
    error_positions = [3, 7, 11]
    error_bits = deepcopy(encoded_bits)
    for i in error_positions:
        error_bits = replace_bit(error_bits, i, 1 - error_bits[i])

    print(f"Received bits:\n{error_bits}")

    print("Errors:")
    print(generate_diff(encoded_bits, error_bits))

    decoder = ViterbiDecoder(encoder)

    original_bits = decoder.decode(encoded_bits)

    # Sanity check
    if original_bits != input_bits:
        print(f"Original {original_bits}")
        print(f"Decoded {input_bits}")
        raise ValueError("Original bits not equal to input bits, decoder is incorrect")

    decoded_bits = decoder.decode(error_bits)
    print(f"Decoded bits:\n{decoded_bits}")

    print(f"Error corrected? {decoded_bits == input_bits}")
