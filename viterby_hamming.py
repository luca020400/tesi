from typing import Dict, Optional, Tuple, List, TypeAlias
from copy import deepcopy

# Define the types for the bits, state, transition, and trellis
Metric: TypeAlias = int
Bit: TypeAlias = int
Bits: TypeAlias = Tuple[Bit, ...]
State: TypeAlias = Tuple[Bit, ...]
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


WINDOW_SIZE = 3


class ConvolutionalEncoder:
    def __init__(self) -> None:
        # Define the initial state
        # Initial state is 0...0 (WINDOW_SIZE - 1 zeros)
        self.state: State = (0,) * (WINDOW_SIZE - 1)

        # Define the state transition and output mapping
        # The key is the (current_state, input_bit)
        # The value is a tuple of (next_state, output_bits)
        self.transitions: FSM = {
            ((0, 0), 0): ((0, 0), (0, 0, 0)),
            ((0, 0), 1): ((0, 1), (1, 1, 1)),
            ((0, 1), 0): ((1, 0), (1, 1, 0)),
            ((0, 1), 1): ((1, 1), (0, 0, 1)),
            ((1, 0), 0): ((0, 0), (1, 0, 1)),
            ((1, 0), 1): ((0, 1), (0, 1, 0)),
            ((1, 1), 0): ((1, 0), (0, 1, 1)),
            ((1, 1), 1): ((1, 1), (1, 0, 0)),
        }

    def encode(self, input_bits: Bits) -> Bits:
        output_bits: list[Bit] = []
        for bit in input_bits:
            # Get the next state and output bits from the current state and input bit
            next_state, output = self.transitions[(self.state, bit)]
            output_bits.extend(output)
            # Update the state
            self.state = next_state
        return tuple(output_bits)


class ViterbiDecoder:
    def __init__(self, encoder: ConvolutionalEncoder):
        self.encoder: ConvolutionalEncoder = encoder

        # 2 bits for state representation, hence we need (WINDOW_SIZE - 1) ^ 2 states
        self.num_states: int = (WINDOW_SIZE - 1) ** 2

    @staticmethod
    def hamming_distance(seq1: Bits, seq2: Bits) -> Metric:
        return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

    def decode(self, received_bits: Bits) -> Bits:
        # Number of encoded bits divided by the code rate
        n = len(received_bits) // WINDOW_SIZE
        trellis: Trellis = [{} for _ in range(n + 1)]

        # Initial state with 0 distance
        trellis[0][(0, 0)] = (0, None)

        # Populate the trellis
        for t in range(n):
            for state in trellis[t]:
                for bit in [0, 1]:
                    next_state, output_bits = self.encoder.transitions[(state, bit)]
                    segment_start = WINDOW_SIZE * t
                    segment_end = segment_start + WINDOW_SIZE
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


encoder = ConvolutionalEncoder()
input_bits = (1, 1, 0, 1, 1, 1)
print(f"Input bits:\n{input_bits}")

encoded_bits = encoder.encode(input_bits)
print(f"Encoded bits:\n{encoded_bits}")

# Introduce 2 bit flips
error_positions = [2, 6]
error_bits = deepcopy(encoded_bits)
for i in error_positions:
    error_bits = replace_bit(error_bits, i, 1 - error_bits[i])

print(f"Received bits:\n{error_bits}")

print("Errors:")
print(generate_diff(encoded_bits, error_bits))

decoder = ViterbiDecoder(encoder)
decoded_bits = decoder.decode(error_bits)
print(f"Decoded bits:\n{decoded_bits}")

print(f"Error corrected? {decoded_bits == input_bits}")
