import pandas as pd
from Bio import SeqIO, PDB

# Function to parse protein sequences and structures
def parse_protein_data(sequence_file, structure_file):
    sequences = [str(record.seq) for record in SeqIO.parse(sequence_file, "fasta")]
    structure_parser = PDB.PDBParser(QUIET=True)
    structure = structure_parser.get_structure("protein", structure_file)
    # Add preprocessing steps as needed
    return sequences, structure

# Example usage
sequences, structure = parse_protein_data("protein_sequences.fasta", "protein_structure.pdb")

# Function for data cleaning and transformation
def preprocess_data(sequences, structure):
    # Perform data cleaning and transformation steps
    # Example: remove duplicates, handle missing values, normalize data
    preprocessed_sequences = sequences  # Placeholder, replace with actual preprocessing
    transformed_structure = structure  # Placeholder, replace with actual preprocessing
    return preprocessed_sequences, transformed_structure

# Example usage
preprocessed_sequences, transformed_structure = preprocess_data(sequences, structure)

# Save preprocessed data to file
def save_preprocessed_data(sequences, structure, output_dir):
    # Save preprocessed data to specified output directory
    pd.DataFrame({"Sequence": sequences}).to_csv(f"{output_dir}/preprocessed_sequences.csv", index=False)
    # Save transformed structure to PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(f"{output_dir}/transformed_structure.pdb")

# Example usage
output_directory = "preprocessed_data"
save_preprocessed_data(preprocessed_sequences, transformed_structure, output_directory)
