#!/usr/bin/env python3

import pandas as pd
import polars as pl
import numpy as np
import pysam
import os 
import sys
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
        prog="dmode",
        description="Modkit transcriptome accelerator: A command line tool to accelerate the modkit evaluation of transcriptome aligned reads.",
    )

parser.add_argument(
        "--reference","-r",dest="reference",type= str,
        help="Path to reference file",
        default=""
    )

parser.add_argument(
        "--bam","-b",dest="bam",type= str,
        help="Path to bam file",
        default=""
    )

parser.add_argument(
    "--threads","-t",dest="threads", type= int,
    help="Number of cores for computation",
    default = 8
)
#mod threshold is not recommended to be changed
# because it is already optimized in modkit
# and changing it might lead to unexpected results
"""
parser.add_argument(
    "--mod_threshold", "--mod-threshold", "-f", dest="mod_threshold", type=float,
    help="Threshold for modkit modification probability (mod threshold)",
    default=0.8
)
"""
parser.add_argument(
    "--filter_threshold", "--filter-threshold", dest="filter_threshold", type=float,
    help="Filter threshold for modkit (minimum probability to count a call as modified)",
    default = 0.98
)

parser.add_argument(
    "--mod_threshold_flags", "--mod-threshold-flags", dest="mod_threshold_flags", type=str,
    help="Modification-specific threshold flags for modkit (e.g., '--mod-threshold m:0.8 --mod-threshold a:0.8')",
    default=""
)

args = parser.parse_args()


# Read a Fasta file with Pysam
def concatenate_transcriptome_fasta(fasta_file):
    """
    Concatenate all reference sequences from a transcriptome FASTA file into a single sequence 
    separated by 10 'N' characters, while generating auxiliary index dictionaries for reconstructing 
    the original FASTA structure from the concatenated version.

    This function is optimized for large transcriptome FASTA files by using file-based 
    sinks instead of in-memory dictionaries to reduce RAM usage.

    Parameters
    ----------
    fasta_file : str
        Path to the input FASTA file containing transcriptome references.

    Returns
    -------
    tuple of str
        A tuple containing four output file paths:
        1. reconstruction_dict_name : str
            Tab-delimited file mapping each base index in the concatenated FASTA to its 
            original reference index and position.
            Columns: ["index", "reference_index", "original_index"]
        2. concat_fasta_file_name : str
            The generated concatenated FASTA file with all references joined by "NNNNNNNNNN".
        3. index_dict_name : str
            File mapping reference indices to their corresponding reference names.
            Columns: ["index", "reference"]
        4. reference_dict_name : str
            File mapping each reference name to its start and end coordinates 
            within the concatenated FASTA.
            Columns: ["key", "start", "end"]

    Notes
    -----
    - Each reference sequence in the input FASTA is appended to the concatenated FASTA, 
      followed by 10 'N' characters as delimiters.
    - The `reconstruction_dict` provides a direct mapping from concatenated FASTA positions 
      back to the original transcriptome coordinates.
    - The additional index and reference mapping files facilitate efficient reconstruction 
      and downstream processing (e.g., coordinate translation, BAM file rewriting).
    - The function writes progress bars using `tqdm` for large datasets.

    Example
    -------
    >>> concatenate_transcriptome_fasta("mouse_transcriptome.fa")
    ('mouse_transcriptome_reconstruction_dict.txt',
     'mouse_transcriptome_concatenated.fa',
     'mouse_transcriptome_index_dict.txt',
     'mouse_transcriptome_reference_dict.txt')
    """
    fasta = pysam.FastaFile(fasta_file)
    # Initialize the index dictionairies for reconstruction of the original fasta from concatenated fasta 
    # between concatenated and original fasta
    references = fasta.references
    reference_dict = {}
    reference_index_dict = {}
    start_position_tracker = 0
    # Writing dictionairy storing the index of the base in concatenated fasta, 
    # the index of reference tag in fasta file and 
    # the index of the base within transcript in origninal fasta
    # File writing is written as a sink to reduce RAM usage
    reconstruction_dict_name = f"{fasta_file.replace('.fa','').replace('.fasta','')}_reconstruction_dict.txt"
    with open(reconstruction_dict_name,"w") as out_dict:
        out_dict.write("index\treference_index\toriginal_index\n")
        for reference_index, reference in tqdm(enumerate(references), total= len(references)):
            reference_index_dict[reference_index] = reference
            reference_length = fasta.get_reference_length(reference)
            reference_dict[reference] = (start_position_tracker, start_position_tracker + reference_length + 10 - 1)
            # Next for loop writes single indices for each position in transition from original to concatenated fasta
            for index in range(start_position_tracker, start_position_tracker + reference_length + 10):
                out_dict.write(f"{index}\t{reference_index}\t{index - start_position_tracker}\n")
            start_position_tracker += reference_length + 10 
        # Addition of last nucleotide to avoid boundary issues
        last_start_position_tracker = start_position_tracker - reference_length - 10 
        last_index = start_position_tracker
        out_dict.write(f"{last_index}\t{reference_index}\t{last_index - last_start_position_tracker}\n")
    # Write concatenated fasta file 
    concat_fasta_file_name = f"{fasta_file.replace('.fa','').replace('.fasta','')}_concatenated.fa"
    with open(concat_fasta_file_name, "w") as out_fasta:
        out_fasta.write(">concatenated_reference\n")
        for reference_index, reference in tqdm(enumerate(references), total= len(references)):
            seq = fasta.fetch(reference)
            seq = seq + "NNNNNNNNNN"
            out_fasta.write(seq)
        out_fasta.write("\n")
    # For each index of reference tag 
    # write an index/tag dictionairy entry to reduce RAM usage 
    # Like that Reference tags have to be stored only once,
    # which is a major memory usage reduction 
    index_dict_name = f"{fasta_file.replace('.fa','').replace('.fasta','')}_index_dict.txt"
    with open(index_dict_name, "w") as out_dict:
        out_dict.write("index\treference\n")
        for key in tqdm(reference_index_dict):
            out_dict.write(f"{key}\t{reference_index_dict[key]}\n")
    reference_dict_name = f"{fasta_file.replace('.fa','').replace('.fasta','')}_reference_dict.txt"
    # Dictionairy storing start and end of each reference tag on the conatenated fasta. 
    # The dictionairy is used to construct the concatenated fasta.
    # Used to rewrite bam file
    with open(reference_dict_name, "w") as out_dict:
        out_dict.write("key\tstart\tend\n")
        for key in tqdm(reference_dict):
            out_dict.write(f"{key}\t{reference_dict[key][0]}\t{reference_dict[key][1]}\n")
    fasta.close()
    return reconstruction_dict_name, concat_fasta_file_name, index_dict_name, reference_dict_name

def modify_bam_file_with_concatenated_reference(bam_file, reference_dict_file, output_bam_file):
    """
    Rewrite a BAM file so that all alignments reference a single concatenated reference 
    instead of multiple individual transcript references.

    This function remaps each read’s reference coordinates from the original 
    transcriptome FASTA to the corresponding coordinates on a concatenated 
    reference FASTA (as generated by `concatenate_transcriptome_fasta`). 

    It updates the BAM header to include only one reference (`concatenated_reference`)
    and adjusts all read positions accordingly.

    Parameters
    ----------
    bam_file : str
        Path to the input BAM file aligned against the original (multi-reference) transcriptome.
    reference_dict_file : str
        Path to the reference dictionary file generated by `concatenate_transcriptome_fasta`.
        This file must contain three tab-separated columns: 
        ["key", "start", "end"], where each key corresponds to a reference name
        and its start/end positions within the concatenated FASTA.
    output_bam_file : str
        Path to the output BAM file to be written. 
        The resulting BAM will contain a single reference named 'concatenated_reference'.

    Returns
    -------
    None
        The function writes the modified BAM file to disk and does not return any objects.

    Notes
    -----
    - The function assumes the BAM file uses the same reference names as those
      present in the `reference_dict_file`.
    - Each read’s `reference_start` is offset by the start coordinate of its 
      corresponding transcript in the concatenated FASTA.
    - The BAM header is rewritten to include a single reference entry with
      total length equal to the sum of all concatenated transcript lengths.
    - Reads aligned to transcripts not present in the reference dictionary
      are reported via a printed warning and remain unmodified.

    Example
    -------
    >>> modify_bam_file_with_concatenated_reference(
    ...     "aligned_reads.bam",
    ...     "mouse_transcriptome_reference_dict.txt",
    ...     "aligned_reads_concatenated.bam"
    ... )
    # Output: A single-reference BAM file aligned to 'concatenated_reference'
    """
    # Read reference dict
    reference_df = pd.read_csv(reference_dict_file, sep="\t", header=0)
    reference_dict = {}
    for key,start,end in zip(reference_df["key"],reference_df["start"],reference_df["end"]):
        reference_dict[key] = (start,end)
    # Initialize original bam file
    bam = pysam.AlignmentFile(bam_file, "rb")
    header = bam.header.to_dict()
    # Modify the header to match only one reference
    header['SQ'] = [{'SN': 'concatenated_reference', 'LN': sum([reference_dict[ref][1] - reference_dict[ref][0] + 1 for ref in reference_dict])}]
    out_bam = pysam.AlignmentFile(output_bam_file, "wb", header=header)
    # Convert every transcript specific coordinate on the original fasta
    # to a coordinate on the concatenated fasta
    for read in tqdm(bam.fetch(until_eof=True)):
        transcript = bam.get_reference_name(read.reference_id)
        if transcript in reference_dict:
            new_start = read.reference_start + reference_dict[transcript][0]
            read.reference_id = 0  # Since we have only one reference now
            read.reference_start = new_start
        else:
            print("Transcript not in reference!!")
        out_bam.write(read)
    bam.close()
    out_bam.close()
    return


def modify_bed_file_with_concatenated_reference(bed_file, reconstruction_dict_file, reference_index_dict_file, output_bed_file):
    """
    Convert genomic coordinates in a BED file aligned to a concatenated reference 
    back to their original transcriptome coordinates.

    This function uses the reconstruction dictionary and reference index mapping 
    (generated by `concatenate_transcriptome_fasta`) to translate BED file intervals 
    from the concatenated FASTA coordinate system back to per-transcript coordinates.  
    It outputs a BED file with corrected transcript names and start/end positions.

    Parameters
    ----------
    bed_file : str
        Path to the input BED file containing features aligned to the concatenated reference.
    reconstruction_dict_file : str
        Path to the reconstruction dictionary file generated by 
        `concatenate_transcriptome_fasta`.  
        This file maps each base index in the concatenated FASTA to its original 
        transcript index and position.  
        Expected columns: ["index", "reference_index", "original_index"]
    reference_index_dict_file : str
        Path to the reference index dictionary file (also from 
        `concatenate_transcriptome_fasta`), mapping integer reference indices to 
        transcript names.  
        Expected columns: ["index", "reference"]
    output_bed_file : str
        Path to the output BED file that will contain remapped transcript-specific coordinates.

    Returns
    -------
    pandas.DataFrame
        The reconstructed BED data as a pandas DataFrame with transcript names 
        and transcriptome-based start/end coordinates.

    Notes
    -----
    - This function is intended for reversing coordinate transformations applied when 
      mapping to a concatenated reference.  
    - It assumes that the BED coordinates (`start`, `end`) correspond to indices 
      in the concatenated FASTA, and that the reconstruction dictionary is indexed 
      at single-base resolution.
    - The function supports large input files efficiently via Polars for reading 
      the reconstruction dictionary (`rechunk=False`, `low_memory=True`).
    - The output BED preserves the original structure, with updated coordinates 
      in columns 1, 2, 6, and 7.

    Example
    -------
    >>> modify_bed_file_with_concatenated_reference(
    ...     "aligned_features_concatenated.bed",
    ...     "mouse_transcriptome_reconstruction_dict.txt",
    ...     "mouse_transcriptome_index_dict.txt",
    ...     "aligned_features_transcriptome.bed"
    ... )
    # Output: BED file with coordinates restored to original transcriptome positions
    """
    #Initilize reconstruction dict
    reconstruction_df = pl.read_csv(
    reconstruction_dict_file,
    separator="\t",
    has_header=True,
    rechunk=False,          # avoids immediate memory consolidation
    low_memory=True         # allows Polars to relax type inference
    )
    # Initialize reference index/tag dictionairy 
    reference_index_df = pd.read_csv(reference_index_dict_file, sep="\t", header=0)
    reference_index_dict = {}
    for index, reference in zip(reference_index_df["index"],reference_index_df["reference"]):
        reference_index_dict[int(index)] = str(reference)
    df = pd.read_csv(bed_file, sep="\t", header=None)
    # Modify the start and end positions based on the reconstruction dict
    reconstructed_transcripts = []
    reconstructed_starts = []
    reconstructed_ends = []
    # Reconstruct transcript id from reference index/tag 
    # dictionairy using index from reconstruction dictionairy.
    # Reconstruct original starts and ends on transcripts from reconstruction dictionairy.
    for start, end in tqdm(zip(df[1],df[2]), total = df.shape[0]):
        reconstructed_transcripts.append(reference_index_dict[int(reconstruction_df["reference_index"][start])])
        reconstructed_starts.append(reconstruction_df["original_index"][start])
        reconstructed_ends.append(reconstruction_df["original_index"][end])# + diff_start_end)
    df[0] = reconstructed_transcripts 
    df[1] = reconstructed_starts
    df[6] = reconstructed_starts
    df[2] = reconstructed_ends
    df[7] = reconstructed_ends
    df.to_csv(output_bed_file, sep="\t", header=False, index=False)
    return df

# Extract meta.id from the BAM filename
meta_id = os.path.basename(args.bam).split('.')[0]

original_fasta_file = args.reference
reconstruction_dict_file, concat_fasta_name, reference_index_dict_file, reference_dict_file = concatenate_transcriptome_fasta(args.reference)

# Use consistent naming with meta.id prefix
output_bam_file = f"{meta_id}.transcriptome_concatenated.bam"
modify_bam_file_with_concatenated_reference(bam_file=args.bam, reference_dict_file=reference_dict_file, output_bam_file=output_bam_file)
os.system(f"samtools index --csi {output_bam_file}")

original_bed_file = f"{output_bam_file.replace('.bam','.bed')}"
# Call modkit pileup with filter-threshold and modification-specific thresholds
modkit_cmd = (
    f"modkit pileup \"{output_bam_file}\" \"{original_bed_file}\" "
    f"-t {args.threads} --filter-threshold {args.filter_threshold} "
    f"{args.mod_threshold_flags} "
    f"--log-filepath {meta_id}.transcriptome.log"
)   
os.system(modkit_cmd)

output_bed_file = f"{meta_id}.transcriptome_modified.bed"
modify_bed_file_with_concatenated_reference(bed_file=original_bed_file, reconstruction_dict_file=reconstruction_dict_file, reference_index_dict_file=reference_index_dict_file, output_bed_file=output_bed_file)

"""# Clean up intermediate files
intermediate_files = [
    reconstruction_dict_file,
    reference_index_dict_file,
    reference_dict_file,
    original_bed_file
]
for file in intermediate_files:
    if os.path.exists(file):
        os.remove(file)
"""