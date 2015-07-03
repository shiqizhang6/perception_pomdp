CFLAGS=-I.
DEPS= Parser.h Simulator.h
OBJ= test.o Parser.o Simulator.o

%.o: %.cpp $(DEPS)
	g++ -c -o $@ $< $(CFLAGS) 

test: $(OBJ)
	g++ -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o *~ test 
