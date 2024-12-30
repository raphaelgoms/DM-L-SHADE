HEADER = de.h
TARGET = solver

OBJS := $(patsubst src/%.cc,src/%.o,$(wildcard src/*.cc))
CC = g++
OPTION = -std=c++14 -O3
LIB_DIR = /home/raphael/Code/DM-L-SHADE/src/lib 
INC_DIR = /home/raphael/Code/DM-L-SHADE/src/include 

# Link the static library directly
LDFLAGS = $(LIB_DIR)/libpyclustering.a -lm

# Add -I to specify the header file directory
CFLAGS = -I$(INC_DIR)

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(OPTION) $(LDFLAGS)

%.o: %.cc $(HEADER)
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf *.o