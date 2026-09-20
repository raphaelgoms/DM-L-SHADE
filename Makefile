TARGET = solver

SRC_DIR = src
BUILD_DIR = build
THIRD_PARTY_DIR = third_party/pyclustering

SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

CC = g++
OPTION = -std=c++14 -O3
LIB_DIR = $(THIRD_PARTY_DIR)/lib
INC_DIR = $(THIRD_PARTY_DIR)/include

# Link the static library directly
LDFLAGS = $(LIB_DIR)/libpyclustering.a -lm

# -I$(SRC_DIR) lets any file under src/ include another (e.g.
# "problems/problem.h" or "algorithm/algorithm.h") by its path from the
# src/ root, regardless of which subdirectory the including file lives in.
CFLAGS = -I$(SRC_DIR) -I$(INC_DIR) -I$(INC_DIR)/pyclustering

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(OPTION) $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
