#!/usr/bin/env python3
"""
Create HepG2 dataset for IGF2BP1 cross-cell-line analysis.

Methodology (from PrismNet paper):
- Positive samples: Top 5000 IDR peaks with >=40% icSHAPE coverage
- Negative samples: 10000 random transcriptome regions with >=40% icSHAPE coverage

Usage:
    python tools/create_hepg2_dataset.py
"""

import os
import sys
import random
import argparse
from collections import defaultdict

import numpy as np
import pyBigWig
from pyfaidx import Fasta

# Configuration
WINDOW_SIZE = 101
MIN_ICSHAPE_COVERAGE = 0.40  # 40% coverage required
NUM_POSITIVES = 5000
NUM_NEGATIVES = 10000
RANDOM_SEED = 1024

# File paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BED_FILE = os.path.join(PROJECT_ROOT, "data/ENCFF442USD.bed")
GENOME_FILE = os.path.join(PROJECT_ROOT, "data/reference/hg38.fa")
GTF_FILE = os.path.join(PROJECT_ROOT, "data/gencode.v44.annotation.gtf")
ICSHAPE_PLUS = os.path.join(PROJECT_ROOT, "icSHAPE/HepG2-plus.bw")
ICSHAPE_MINUS = os.path.join(PROJECT_ROOT, "icSHAPE/HepG2-minus.bw")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/clip_data/IGF2BP1_HepG2.tsv")


def parse_bed(bed_file):
    """Parse BED file and return list of peaks."""
    peaks = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.strip().split('\t')
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            name = fields[3]
            # Signal strength is in column 7 (log2 fold enrichment)
            signal = float(fields[6]) if len(fields) > 6 else 0.0
            strand = fields[5] if len(fields) > 5 else '+'
            peaks.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'name': name,
                'signal': signal,
                'strand': strand
            })
    return peaks


def get_window_center(start, end):
    """Get center position of a region."""
    return (start + end) // 2


def expand_to_window(center, window_size=WINDOW_SIZE):
    """Expand from center to get window coordinates."""
    half = window_size // 2
    return center - half, center + half + 1  # +1 for 101nt


def get_sequence(genome, chrom, start, end, strand='+'):
    """Extract sequence from genome, convert to RNA."""
    try:
        seq = str(genome[chrom][start:end]).upper()
        if strand == '-':
            # Reverse complement
            complement = {'A': 'U', 'C': 'G', 'G': 'C', 'T': 'A', 'U': 'A', 'N': 'N'}
            seq = ''.join(complement.get(base, 'N') for base in reversed(seq))
        else:
            # Just convert T to U
            seq = seq.replace('T', 'U')
        return seq
    except (KeyError, ValueError):
        return None


def get_icshape_values(bw_plus, bw_minus, chrom, start, end, strand='+'):
    """Query icSHAPE values from BigWig files."""
    try:
        bw = bw_plus if strand == '+' else bw_minus
        values = bw.values(chrom, start, end)
        if values is None:
            return None
        # Convert nan to -1 (missing)
        result = []
        for v in values:
            if v is None or np.isnan(v):
                result.append(-1.0)
            else:
                result.append(round(v, 3))
        return result
    except Exception:
        return None


def calculate_coverage(icshape_values):
    """Calculate fraction of positions with valid icSHAPE values."""
    if not icshape_values:
        return 0.0
    valid = sum(1 for v in icshape_values if v >= 0)
    return valid / len(icshape_values)


def format_icshape(values):
    """Format icSHAPE values as comma-separated string."""
    return ','.join(str(v) for v in values)


def parse_gtf_exons(gtf_file):
    """Parse GTF file and extract exon coordinates per transcript."""
    print("Parsing GTF file (this may take a while)...")
    transcripts = defaultdict(list)

    with open(gtf_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                print(f"  Processed {i:,} lines...")

            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            if fields[2] != 'exon':
                continue

            chrom = fields[0]
            start = int(fields[3]) - 1  # Convert to 0-based
            end = int(fields[4])
            strand = fields[6]

            # Parse attributes to get transcript_id
            attrs = fields[8]
            transcript_id = None
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('transcript_id'):
                    transcript_id = attr.split('"')[1]
                    break

            if transcript_id:
                transcripts[transcript_id].append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'strand': strand
                })

    print(f"  Found {len(transcripts):,} transcripts")
    return transcripts


def get_transcript_regions(transcripts, min_length=WINDOW_SIZE):
    """Convert transcript exons to list of valid sampling regions."""
    regions = []

    for transcript_id, exons in transcripts.items():
        # Sort exons by position
        exons = sorted(exons, key=lambda x: x['start'])

        for exon in exons:
            length = exon['end'] - exon['start']
            if length >= min_length:
                regions.append({
                    'chrom': exon['chrom'],
                    'start': exon['start'],
                    'end': exon['end'],
                    'strand': exon['strand'],
                    'transcript_id': transcript_id
                })

    return regions


def regions_overlap(start1, end1, start2, end2):
    """Check if two regions overlap."""
    return start1 < end2 and start2 < end1


def sample_negative_regions(transcript_regions, peaks, genome, bw_plus, bw_minus,
                            num_samples=NUM_NEGATIVES, window_size=WINDOW_SIZE):
    """Sample negative regions from transcriptome avoiding peaks."""
    print(f"Sampling {num_samples} negative regions...")

    # Build peak index by chromosome for fast lookup
    peak_index = defaultdict(list)
    for peak in peaks:
        peak_index[peak['chrom']].append((peak['start'], peak['end']))

    # Filter regions by chromosome (only those in peak_index or with icSHAPE)
    valid_chroms = set(peak_index.keys())
    filtered_regions = [r for r in transcript_regions if r['chrom'] in valid_chroms]

    if not filtered_regions:
        print("Warning: No valid regions found for negative sampling")
        return []

    negatives = []
    attempts = 0
    max_attempts = num_samples * 100

    random.shuffle(filtered_regions)
    region_idx = 0

    while len(negatives) < num_samples and attempts < max_attempts:
        attempts += 1

        if attempts % 10000 == 0:
            print(f"  Attempt {attempts}, found {len(negatives)} negatives...")

        # Pick a random region
        region = filtered_regions[region_idx % len(filtered_regions)]
        region_idx += 1

        # Pick a random position within the region
        max_start = region['end'] - window_size
        if max_start <= region['start']:
            continue

        start = random.randint(region['start'], max_start)
        end = start + window_size

        # Check for overlap with peaks
        chrom = region['chrom']
        overlaps = False
        for peak_start, peak_end in peak_index.get(chrom, []):
            if regions_overlap(start, end, peak_start, peak_end):
                overlaps = True
                break

        if overlaps:
            continue

        # Get sequence
        seq = get_sequence(genome, chrom, start, end, region['strand'])
        if not seq or len(seq) != window_size or 'N' in seq:
            continue

        # Get icSHAPE values
        icshape = get_icshape_values(bw_plus, bw_minus, chrom, start, end, region['strand'])
        if not icshape or len(icshape) != window_size:
            continue

        # Check coverage
        coverage = calculate_coverage(icshape)
        if coverage < MIN_ICSHAPE_COVERAGE:
            continue

        # Valid negative sample
        negatives.append({
            'chrom': chrom,
            'start': start,
            'end': end,
            'strand': region['strand'],
            'seq': seq,
            'icshape': icshape,
            'transcript_id': region['transcript_id']
        })

    print(f"  Found {len(negatives)} valid negatives in {attempts} attempts")
    return negatives


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Creating HepG2 dataset for IGF2BP1")
    print("=" * 60)

    # Load genome
    print("\nLoading reference genome...")
    genome = Fasta(GENOME_FILE)

    # Load icSHAPE BigWig files
    print("Loading icSHAPE data...")
    bw_plus = pyBigWig.open(ICSHAPE_PLUS)
    bw_minus = pyBigWig.open(ICSHAPE_MINUS)

    # Parse peaks
    print("\nParsing IDR peaks...")
    peaks = parse_bed(BED_FILE)
    print(f"  Found {len(peaks)} peaks")

    # Sort by signal strength
    peaks = sorted(peaks, key=lambda x: x['signal'], reverse=True)

    # Process positive samples
    print("\nProcessing positive samples...")
    positives = []

    for i, peak in enumerate(peaks):
        if i % 500 == 0:
            print(f"  Processing peak {i}/{len(peaks)}...")

        # Get window centered on peak
        center = get_window_center(peak['start'], peak['end'])
        start, end = expand_to_window(center)

        # Get sequence
        seq = get_sequence(genome, peak['chrom'], start, end, peak['strand'])
        if not seq or len(seq) != WINDOW_SIZE:
            continue
        if 'N' in seq:
            continue

        # Get icSHAPE values
        icshape = get_icshape_values(bw_plus, bw_minus, peak['chrom'], start, end, peak['strand'])
        if not icshape or len(icshape) != WINDOW_SIZE:
            continue

        # Check coverage
        coverage = calculate_coverage(icshape)
        if coverage < MIN_ICSHAPE_COVERAGE:
            continue

        positives.append({
            'chrom': peak['chrom'],
            'start': start,
            'end': end,
            'strand': peak['strand'],
            'signal': peak['signal'],
            'seq': seq,
            'icshape': icshape
        })

        # Stop if we have enough
        if len(positives) >= NUM_POSITIVES:
            break

    print(f"  Found {len(positives)} valid positives (target: {NUM_POSITIVES})")

    # Parse GTF for negative sampling
    transcripts = parse_gtf_exons(GTF_FILE)
    transcript_regions = get_transcript_regions(transcripts)
    print(f"  Found {len(transcript_regions)} exon regions for sampling")

    # Sample negatives
    negatives = sample_negative_regions(
        transcript_regions, peaks, genome, bw_plus, bw_minus,
        num_samples=NUM_NEGATIVES
    )

    # Write output
    print(f"\nWriting output to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        # Header
        f.write("Type\tName\tSeq\ticshape\tScore\tlabel\n")

        # Positives
        for i, pos in enumerate(positives):
            name = f"{pos['chrom']}|{pos['start']}|{pos['end']}|{pos['signal']:.3f}"
            icshape_str = format_icshape(pos['icshape'])
            f.write(f"A\t{name}\t{pos['seq']}\t{icshape_str}\t{pos['signal']:.3f}\t1\n")

        # Negatives
        for i, neg in enumerate(negatives):
            name = f"{neg['chrom']}|{neg['start']}|{neg['end']}|0.0"
            icshape_str = format_icshape(neg['icshape'])
            f.write(f"B\t{name}\t{neg['seq']}\t{icshape_str}\t0.0\t0\n")

    print(f"\nDone! Created dataset with:")
    print(f"  - {len(positives)} positive samples")
    print(f"  - {len(negatives)} negative samples")
    print(f"  - Total: {len(positives) + len(negatives)} samples")

    # Cleanup
    bw_plus.close()
    bw_minus.close()


if __name__ == "__main__":
    main()
