# Cross3D Preprocessing - C Implementation Makefile
# Author: Cross3D C Implementation
# Date: 2024

#==============================================================================
# Compiler Settings
#==============================================================================
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Debug build
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -std=c99 -DDEBUG

#==============================================================================
# Directories
#==============================================================================
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin
OUTPUT_DIR = output

#==============================================================================
# Source Files
#==============================================================================
SOURCES = $(SRC_DIR)/main.c \
          $(SRC_DIR)/audio_reader.c \
          $(SRC_DIR)/fft.c \
          $(SRC_DIR)/gcc_phat.c \
          $(SRC_DIR)/srp_map.c \
          $(SRC_DIR)/test_data.c

OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SOURCES))

#==============================================================================
# Target
#==============================================================================
TARGET = $(BIN_DIR)/cross3d_preprocess

#==============================================================================
# Rules
#==============================================================================
.PHONY: all clean debug run dirs

all: dirs $(TARGET)

debug: CFLAGS = $(DEBUG_CFLAGS)
debug: clean dirs $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete: $@"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

dirs:
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	@if not exist $(OUTPUT_DIR) mkdir $(OUTPUT_DIR)

clean:
	@echo "Cleaning..."
	@if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR)
	@if exist $(BIN_DIR) rmdir /s /q $(BIN_DIR)
	@if exist $(OUTPUT_DIR) rmdir /s /q $(OUTPUT_DIR)
	@echo "Clean complete"

run: all
	@echo "Running $(TARGET)..."
	@$(TARGET)

#==============================================================================
# Dependencies
#==============================================================================
$(OBJ_DIR)/main.o: $(SRC_DIR)/main.c $(INC_DIR)/config.h $(INC_DIR)/types.h \
                   $(INC_DIR)/audio_reader.h $(INC_DIR)/fft.h \
                   $(INC_DIR)/gcc_phat.h $(INC_DIR)/srp_map.h $(INC_DIR)/test_data.h

$(OBJ_DIR)/audio_reader.o: $(SRC_DIR)/audio_reader.c $(INC_DIR)/audio_reader.h \
                           $(INC_DIR)/config.h $(INC_DIR)/types.h

$(OBJ_DIR)/fft.o: $(SRC_DIR)/fft.c $(INC_DIR)/fft.h $(INC_DIR)/config.h $(INC_DIR)/types.h

$(OBJ_DIR)/gcc_phat.o: $(SRC_DIR)/gcc_phat.c $(INC_DIR)/gcc_phat.h $(INC_DIR)/fft.h \
                       $(INC_DIR)/config.h $(INC_DIR)/types.h

$(OBJ_DIR)/srp_map.o: $(SRC_DIR)/srp_map.c $(INC_DIR)/srp_map.h $(INC_DIR)/gcc_phat.h \
                      $(INC_DIR)/config.h $(INC_DIR)/types.h

$(OBJ_DIR)/test_data.o: $(SRC_DIR)/test_data.c $(INC_DIR)/test_data.h \
                        $(INC_DIR)/config.h $(INC_DIR)/types.h

#==============================================================================
# Help
#==============================================================================
help:
	@echo "Cross3D Preprocessing Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the project (default)"
	@echo "  debug   - Build with debug flags"
	@echo "  clean   - Remove build files"
	@echo "  run     - Build and run the program"
	@echo "  help    - Show this help message"
