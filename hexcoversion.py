def generate_hex(opcodes, filename="output.hex"):
    with open(filename, "w") as hex_file:
        for i, opcode in enumerate(opcodes):
            hex_value = f":10{format(i, '04X')}00{opcode.encode().hex().upper()}00"
            hex_file.write(hex_value + "\n")

opcodes = ["ADR", "ADD", "LDR", "MOV", "STR", "CMP", "BX", "BL", "SUB", "BNE", "SVC", "BCC"]
generate_hex(opcodes)
print("HEX file generated successfully.")
