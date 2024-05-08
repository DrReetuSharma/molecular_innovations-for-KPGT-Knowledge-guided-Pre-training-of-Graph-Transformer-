import pandas as pd
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO, PDB

# Function to parse protein sequences and structures
def parse_protein_data(sequence_file, structure_file):
    try:
        sequences = [str(record.seq) for record in SeqIO.parse(sequence_file, "fasta")]
        structure_parser = PDB.PDBParser(QUIET=True)
        structure = structure_parser.get_structure("protein", structure_file)
    except FileNotFoundError as e:
        print(f"Error: {e}. File not found.")
        return None, None
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None, None

    return sequences, structure

# Example usage
sequences, structure = parse_protein_data("protein_sequences.fasta", "protein_structure.pdb")
if sequences is None or structure is None:
    exit(1)  # Exit program if data parsing failed

# Function for data cleaning and transformation
def preprocess_data(sequences, structure):
    # Convert sequences and structure to DataFrame for processing
    df_sequences = pd.DataFrame(sequences, columns=['Sequence'])
    df_structure = pd.DataFrame(structure, columns=['Structure'])

    # Perform data cleaning steps
    #  remove duplicates, handle missing values
    df_sequences.drop_duplicates(inplace=True)
    df_sequences.dropna(inplace=True)

    # Perform data transformation steps
    # Example: normalize data using StandardScaler
    scaler = StandardScaler()
    normalized_structure = scaler.fit_transform(df_structure[['Structure']])

    # Convert normalized_structure back to original format
    transformed_structure = normalized_structure.flatten()

    return df_sequences.values.tolist(), transformed_structure.tolist()

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
