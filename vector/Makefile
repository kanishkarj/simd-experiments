run: test vector	
	g++ -O3 test.o parvec.o

vector:
	g++ -O3 -mavx2 -mavx -Wpsabi -msse4 -mfma -fno-tree-vectorize -c parvec.cpp

test: 
	g++ -O3 -c test.cpp

clean:
	rm parvec.o test.o a.out