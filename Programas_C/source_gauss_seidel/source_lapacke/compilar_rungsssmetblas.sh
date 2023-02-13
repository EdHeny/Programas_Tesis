# chmod +x file_name.sh
gcc -Wall -c nlao_ur_fn.c
gcc -Wall -c jacobiMethodLAL.c
gcc -Wall jacobiMethodLAL.o nlao_ur_fn.o -lblas -o jacmetlap