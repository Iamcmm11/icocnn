CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -I. -Ifull_stage1_legacy -Ifrontend_dual_feature -Ifeature_maba -Ipost_maba -Wall -Wextra -Wno-unknown-pragmas

TARGET = test_ifan_dual_frontend
FULL_TARGET = test_ifan_stage1
MABA_TARGET = test_feature_maba
POST_TARGET = test_post_maba
SRCS = frontend_dual_feature/ifan_dual_frontend.cpp full_stage1_legacy/ifan_stage1_engines.cpp frontend_dual_feature/test_ifan_dual_frontend.cpp
FULL_SRCS = full_stage1_legacy/ifan_stage1.cpp full_stage1_legacy/ifan_stage1_engines.cpp full_stage1_legacy/test_ifan_stage1.cpp
MABA_SRCS = feature_maba/ifan_stage1_maba.cpp feature_maba/test_feature_maba.cpp
POST_SRCS = post_maba/ifan_stage1_post.cpp post_maba/test_post_maba.cpp

.PHONY: all clean run full run-full maba run-maba post run-post

all: $(TARGET)

$(TARGET): $(SRCS) frontend_dual_feature/ifan_dual_frontend.hpp full_stage1_legacy/ifan_stage1.hpp full_stage1_legacy/ifan_stage1_engines.hpp
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

full: $(FULL_TARGET)

$(FULL_TARGET): $(FULL_SRCS) full_stage1_legacy/ifan_stage1.hpp full_stage1_legacy/ifan_stage1_engines.hpp
	$(CXX) $(CXXFLAGS) $(FULL_SRCS) -o $(FULL_TARGET)

maba: $(MABA_TARGET)

$(MABA_TARGET): $(MABA_SRCS) full_stage1_legacy/ifan_stage1.hpp feature_maba/ifan_stage1_maba.hpp
	$(CXX) $(CXXFLAGS) $(MABA_SRCS) -o $(MABA_TARGET)

post: $(POST_TARGET)

$(POST_TARGET): $(POST_SRCS) full_stage1_legacy/ifan_stage1.hpp feature_maba/ifan_stage1_maba.hpp post_maba/ifan_stage1_post.hpp
	$(CXX) $(CXXFLAGS) $(POST_SRCS) -o $(POST_TARGET)

run: $(TARGET)
	./$(TARGET)

run-full: $(FULL_TARGET)
	./$(FULL_TARGET)

run-maba: $(MABA_TARGET)
	./$(MABA_TARGET)

run-post: $(POST_TARGET)
	./$(POST_TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe $(FULL_TARGET) $(FULL_TARGET).exe $(MABA_TARGET) $(MABA_TARGET).exe $(POST_TARGET) $(POST_TARGET).exe
