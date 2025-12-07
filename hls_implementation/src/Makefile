# HLS Layer0 IcoConv 测试 Makefile

# 编译器配置
CC = g++
CFLAGS = -Wall -Wextra -O2 -std=c++11
INCLUDES = -I../include
LDFLAGS = -lm

# 目录
SRC_DIR = .
BUILD_DIR = ../build
BIN_DIR = ../bin

# 源文件
SOURCES = ico_conv_layer0.cpp utils.cpp test_ico_conv.cpp
OBJECTS = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
TARGET = $(BIN_DIR)/test_ico_conv

# 默认目标
all: directories $(TARGET)

# 创建目录
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# 链接
$(TARGET): $(OBJECTS)
	@echo "链接 $@..."
	@$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "✓ 编译完成: $@"

# 编译
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "编译 $<..."
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 运行测试
run: $(TARGET)
	@echo ""
	@echo "运行测试..."
	@cd $(BIN_DIR) && ./test_ico_conv

# 清理
clean:
	@echo "清理编译文件..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✓ 清理完成"

# 帮助
help:
	@echo "HLS Layer0 测试 Makefile"
	@echo ""
	@echo "可用目标:"
	@echo "  all      - 编译所有文件 (默认)"
	@echo "  run      - 编译并运行测试"
	@echo "  clean    - 清理编译文件"
	@echo "  help     - 显示此帮助信息"

.PHONY: all directories run clean help
