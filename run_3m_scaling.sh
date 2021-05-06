#!/bin/bash
echo "!! Running Baseline GPU ONLY !!"
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 1 0

echo "!! Running test for 3M alignments with Host Stealing ON !!"
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 1 1
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 2 1
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 3 1
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 4 1
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 5 1
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 6 1
echo "!! Running test for 3M alignments with Host Stealing OFF !!"
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 1 0
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 2 0
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 3 0
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 4 0
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 5 0
./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file 100 6 0

