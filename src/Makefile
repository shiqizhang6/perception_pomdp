CFLAGS=-I.
DEPS= PolicyInterpreter.h PomdpModel.h Properties.h utilities.h Simulator.h FeatureSelector.h
OBJ= PolicyInterpreter.o PomdpModel.o ModelWriter.o ModelLoader.o Simulator.o FeatureSelector.o test.o 
LIBS=-lboost_filesystem -lboost_system # the second is needed coz it's a 64 OS

%.o: %.cpp $(DEPS)
	g++ -c -o $@ $< $(CFLAGS) 

test: $(OBJ)
	g++ -o $@ $^ $(CFLAGS) $(LIBS)

clean:
	rm -f *.o *~ test 
