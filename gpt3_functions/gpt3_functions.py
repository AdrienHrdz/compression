import numpy as np 

def unzigzag(zigzag):
    # Create an empty 8 by 8 matrix
    matrix = [[0] * 8 for _ in range(8)]

    # Initialize the variables to keep track of the current position
    x, y = 0, 0

    # Flag to determine the direction of the zigzag
    up = True

    # Loop until all values in the zigzag list have been visited
    for i in range(len(zigzag)):
        # Get the current value from the zigzag list
        value = zigzag[i]

        # Update the matrix
        matrix[x][y] = value

        # Update the position
        if up:
            x -= 1
            y += 1
            if x < 0:
                x = 0
                y += 2
                up = False
            if y >= 8:
                x += 2
                y -= 1
                up = False
        else:
            x += 1
            y -= 1
            if y < 0:
                y = 0
                x += 2
                up = True
            if x >= 8:
                x -= 1
                y += 2
                up = True

    return matrix

def divide_image(image):
    # Create a list to store the blocks
    blocks = []

    # Divide the image into 8 by 8 blocks
    for i in range(0, 512, 8):
        for j in range(0, 512, 8):
            # Crop the block from the original image
            block = image[i:i+8, j:j+8]

            # Add the block to the list of blocks
            blocks.append(block)

    return blocks

def reconstruct_image(blocks):
    # Create an empty array to store the reconstructed image
    image = np.zeros((512, 512), dtype=np.uint8)

    # Iterate over the blocks and reconstruct the image
    for i in range(0, 512, 8):
        for j in range(0, 512, 8):
            # Get the current block from the list of blocks
            block = blocks[(i // 8) * (512 // 8) + (j // 8)]

            # Insert the block into the reconstructed image
            image[i:i+8, j:j+8] = block

    return image

def zigzag(matrix):
    # Create a list to store the zigzag-coded values
    zigzag = []

    # Initialize the variables to keep track of the current position
    x, y = 0, 0

    # Flag to determine the direction of the zigzag
    up = True

    # Loop until all values in the matrix have been visited
    while x < 8 and y < 8:
        # Add the current value to the zigzag list
        zigzag.append(matrix[x][y])

        # Update the position
        if up:
            x -= 1
            y += 1
            if x < 0:
                x = 0
                y += 2
                up = False
            if y >= 8:
                x += 2
                y -= 1
                up = False
        else:
            x += 1
            y -= 1
            if y < 0:
                y = 0
                x += 2
                up = True
            if x >= 8:
                x -= 1
                y += 2
                up = True

    return zigzag


def huffman_encode(string):
    """
    Encodes a string using Huffman coding.

    Args:
        string (str): The string to be encoded.

    Returns:
        encoded (str): The Huffman-encoded version of the string.
        frequencies (dict): A dictionary of character frequencies.
    """
    # Create a dictionary to store the frequencies of each character
    frequencies = {}

    # Loop through the characters in the string and update the frequencies
    for c in string:
        if c in frequencies:
            frequencies[c] += 1
        else:
            frequencies[c] = 1

    # Create a list of tuples containing the characters and their frequencies
    freq_list = [(freq, char) for char, freq in frequencies.items()]

    # Sort the list of tuples by frequency
    freq_list.sort()

    # Create a Huffman tree
    while len(freq_list) > 1:
        # Pop the two lowest-frequency characters from the list
        char1 = freq_list.pop(0)
        char2 = freq_list.pop(0)

        # Create a new node with the combined frequency of the two characters
        node = (char1[0] + char2[0], char1, char2)

        # Insert the new node into the list
        freq_list.insert(0, node)

    # Traverse the Huffman tree to create the encoding table
    encoding_table = {}
    def traverse(node, path):
        # If the node is a leaf, add the character and its encoded value to the table
        if len(node) == 2:
            encoding_table[node[1]] = path
        else:
            # If the node is not a leaf, traverse its children
            traverse(node[1], path + "0")
            traverse(node[2], path + "1")

    traverse(freq_list[0], "")

    # Create the encoded string by looking up the encoded values in the table
    encoded = ""
    for c in string:
        encoded += encoding_table[c]

    # Return the encoded message and the frequency dictionary
    return encoded, frequencies

def huffman_decode(encoded, frequencies):
    # Create a Huffman tree
    freq_list = [(freq, char) for char, freq in frequencies.items()]
    while len(freq_list) > 1:
        char1 = freq_list.pop(0)
        char2 = freq_list.pop(0)
        node = (char1[0] + char2[0], char1, char2)
        freq_list.insert(0, node)

    # Traverse the Huffman tree to create the decoding table
    decoding_table = {}
    def traverse(node, path):
        if len(node) == 2:
            decoding_table[path] = node[1]
        else:
            traverse(node[1], path + "0")
            traverse(node[2], path + "1")

    traverse(freq_list[0], "")

    # Create the decoded string by looking up the decoded values in the table
    decoded = ""
    i = 0
    while i < len(encoded):
        for j in range(i, len(encoded) + 1):
            if encoded[i:j] in decoding_table:
                decoded += decoding_table[encoded[i:j]]
                i = j
                break

    return decoded


# if __name__ == "__main__":
#     print(huffman_encode.__doc__)
    
#     pass