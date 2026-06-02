CXX = g++
CXXFLAGS = -std=c++11 -O2 -I. -I.. -Wall
C8T6_DEFS = -DICO_LAYER2_5_TIME_STEPS=6 -DICO_LAYER2_5_CIN=8 -DICO_LAYER2_5_COUT=8

SRCS = ico_conv_layer2_5.cpp test_ico_conv_layer2_5.cpp
TARGET = test_ico_conv_layer2_5
C8T6_TARGET = test_ico_conv_layer2_5_c8t6

DEBUG_SRCS = ico_conv_layer2_5.cpp test_ico_conv_layer2_5_debug.cpp
DEBUG_TARGET = test_ico_conv_layer2_5_debug

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

c8t6: $(SRCS)
	$(CXX) $(CXXFLAGS) $(C8T6_DEFS) -o $(C8T6_TARGET) $^

$(DEBUG_TARGET): $(DEBUG_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	del /Q $(TARGET).exe $(C8T6_TARGET).exe $(DEBUG_TARGET).exe 2>nul || rm -f $(TARGET) $(C8T6_TARGET) $(DEBUG_TARGET)

.PHONY: all c8t6 clean
