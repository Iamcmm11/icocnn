CXX = g++
CXXFLAGS = -std=c++11 -O2 -I. -Wall

SRCS = ico_conv_layer0.cpp test_ico_conv.cpp
TARGET = test_ico_conv

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	del /Q $(TARGET).exe 2>nul || rm -f $(TARGET)

.PHONY: all clean
