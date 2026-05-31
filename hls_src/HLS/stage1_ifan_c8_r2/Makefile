CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -I. -Wall -Wextra -Wno-unknown-pragmas

TARGET = test_ifan_stage1
MABA_TARGET = test_feature_maba
POST_TARGET = test_post_maba
SRCS = ifan_stage1.cpp ifan_stage1_engines.cpp test_ifan_stage1.cpp
MABA_SRCS = ifan_stage1_maba.cpp test_feature_maba.cpp
POST_SRCS = ifan_stage1_post.cpp test_post_maba.cpp

.PHONY: all clean run maba run-maba post run-post

all: $(TARGET)

$(TARGET): $(SRCS) ifan_stage1.hpp ifan_stage1_engines.hpp
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

maba: $(MABA_TARGET)

$(MABA_TARGET): $(MABA_SRCS) ifan_stage1.hpp ifan_stage1_maba.hpp
	$(CXX) $(CXXFLAGS) $(MABA_SRCS) -o $(MABA_TARGET)

post: $(POST_TARGET)

$(POST_TARGET): $(POST_SRCS) ifan_stage1.hpp ifan_stage1_maba.hpp ifan_stage1_post.hpp
	$(CXX) $(CXXFLAGS) $(POST_SRCS) -o $(POST_TARGET)

run: $(TARGET)
	./$(TARGET)

run-maba: $(MABA_TARGET)
	./$(MABA_TARGET)

run-post: $(POST_TARGET)
	./$(POST_TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe $(MABA_TARGET) $(MABA_TARGET).exe $(POST_TARGET) $(POST_TARGET).exe
