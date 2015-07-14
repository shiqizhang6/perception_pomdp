CFLAGS=-I.
DEPS= PolicyInterpreter.h Simulator.h Properties.h utilities.h
OBJ= test.o PolicyInterpreter.o Simulator.o ModelWriter.o ModelLoader.o
LIBS=-lboost_filesystem -lboost_system # the second is needed coz it's a 64 OS

%.o: %.cpp $(DEPS)
	g++ -c -o $@ $< $(CFLAGS) 

test: $(OBJ)
	g++ -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f *.o *~ test 
