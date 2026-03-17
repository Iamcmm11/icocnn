CXX = g++
CXXFLAGS = -std=c++11 -O2 -I. -I.. -Wall

SRCS = ico_conv_layer2_5.cpp test_ico_conv_layer2_5.cpp
TARGET = test_ico_conv_layer2_5

DEBUG_SRCS = ico_conv_layer2_5.cpp test_ico_conv_layer2_5_debug.cpp
DEBUG_TARGET = test_ico_conv_layer2_5_debug

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(DEBUG_TARGET): $(DEBUG_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	del /Q $(TARGET).exe $(DEBUG_TARGET).exe 2>nul || rm -f $(TARGET) $(DEBUG_TARGET)

.PHONY: all clean
