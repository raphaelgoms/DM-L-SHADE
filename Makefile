HEADER = de.h
TARGET = solver

OBJS := $(patsubst %.cc,%.o,$(shell find src -name '*.cc'))
CC = g++
OPTION = -std=c++14 -O3
LIB_DIR = ./src/lib
INC_DIR = ./src/include

# Link the static library directly
LDFLAGS = $(LIB_DIR)/libpyclustering.a -lm

# Add -I to specify the header file directory
CFLAGS = -I$(INC_DIR) -I$(INC_DIR)/pyclustering

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(OPTION) $(LDFLAGS)

%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf src/*.o $(TARGET)
