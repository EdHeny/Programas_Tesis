# chmod +x file_name.sh
gcc -Wall -c nlao_ur_fn.c
gcc -Wall -c jacobi_method_nr.c
gcc jacobi_method_nr.o nlao_ur_fn.o -lm -o jametnr