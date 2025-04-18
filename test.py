def octal_to_binary(octal):
    binary = bin(int(octal, 8))[2:]
    return [int(b) for b in binary.zfill(len(octal) * 3)]


def conv_encoder(input_bits, generators):
    """
    Convolutional encoder function.

    :param input_bits: list of input bits to be encoded
    :param generators: list of generator polynomials in octal form
    :return: list of encoded bits
    """
    # Convert generators from octal to binary
    generators_bin = [octal_to_binary(gen) for gen in generators]
    constraint_length = max(len(gen) for gen in generators_bin)
    print("Constraint length:", constraint_length)

    # Zero-padding the input bits
    input_padded = [0] * (constraint_length - 1) + input_bits

    encoded_bits = []

    for i in range(len(input_bits)):
        window = input_padded[i : i + constraint_length]

        for gen in generators_bin:
            encoded_bit = 0
            for j in range(len(gen)):
                encoded_bit ^= gen[j] & window[j]
            encoded_bits.append(encoded_bit)

    return encoded_bits


# Example usage
input_bits = [1, 1, 0, 1]  # Example input sequence
generators = ["7", "6", "5", "4", "3"]  # Example generators in octal form

encoded_bits = conv_encoder(input_bits, generators)
print("Encoded bits:", encoded_bits)
