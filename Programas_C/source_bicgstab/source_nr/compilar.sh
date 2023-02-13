gcc -Wall -c nlao_ur_fn.c
gcc -Wall -c bicgstab_method_nr.c
gcc bicgstab_method_nr.o nlao_ur_fn.o -lm -o runbicgstab