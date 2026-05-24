CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -I. -Wall -Wextra -Wno-unknown-pragmas

TARGET = test_ifan_stage1
SRCS = ifan_stage1.cpp ifan_stage1_engines.cpp test_ifan_stage1.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) ifan_stage1.hpp ifan_stage1_engines.hpp
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe
