run: test strops	
	g++ -O3 test.o string_ops.o

strops:
	g++ -O3 -mavx2 -mavx -Wpsabi -msse4 -mfma -fno-tree-vectorize -c string_ops.cpp

test: test.cpp
	g++ -O3 -c test.cpp

clean:
	rm string_ops.o test.o a.out