CC = g++
CXXFLAGS = -O3 -fopenmp

TARGET = main.elf
SOURCES = main.cpp

BIN_DIR := bin
SRC_DIR := src

all: $(TARGET)

$(TARGET): ${SRC_DIR}/${SOURCES}
	@mkdir -p ${BIN_DIR}
	@$(CC) $(CXXFLAGS) $(SRC_DIR)/$(SOURCES) -o $(BIN_DIR)/$(TARGET)

clean:
	@rm -rf ${BIN_DIR}

run: ${TARGET}
	@./${BIN_DIR}/${TARGET}

help:
	@echo "  make        - компиляция программы"
	@echo "  make clean  - удаление папки c исполняемым файлом"
	@echo "  make run    - запуск программы"
	@echo "  make help   - показать эту справку"
