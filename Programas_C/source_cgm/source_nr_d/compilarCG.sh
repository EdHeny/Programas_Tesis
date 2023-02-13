# chmod +x file_name.sh
# sh file_name.sh
gcc -Wall -c nlao_ur_fn.c
gcc -Wall -c cg_method_nr.c
gcc -Wall cg_method_nr.o nlao_ur_fn.o -lm -o cgmetnr