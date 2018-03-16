#code from max
echo und los


mkdir ../datasets/PSSM_files/
mkdir ../datasets/PSSM_files/single_fastas/
mkdir ../datasets/PSSM_files/PSSMs/

cd ../scripts/
python3 pssm_inputfile_separator.py

cd ../datasets/PSSM_files/single_fastas/

for b in *.fasta

do psiblast -query $b -evalue 0.01 -db /home/u2360/swissprot.fa -num_iterations 3 -out_ascii_pssm ../PSSMs/$b.pssm -num_threads=8

done
echo fertig
