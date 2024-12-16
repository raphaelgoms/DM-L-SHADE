HEADER = de.h
TARGET = lshade_exe

OBJS := $(patsubst src/%.cc,src/%.o,$(wildcard src/*.cc))
CC = g++
OPTION = -O3

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(OPTION) -lm 

%.o: %.cc $(HEADER)
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf *.o