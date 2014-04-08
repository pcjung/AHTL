
CC = icc
CXX = icpc

CFLAGS = -fast -openmp -ipo

LDFLAGS = -openmp -ipo -lm -fPIC

AR = ar

AHTL = ahtl
LIB_AHTL = lib$(AHTL)

LINKAGE = dynamic

ifeq ($(LINKAGE),static)
TARGET = $(LIB_AHTL).a
LIB_DEP = $(HOME)/$(LIB_DIR)/$(TARGET)
endif

ifeq ($(LINKAGE),dynamic)
TARGET = $(LIB_AHTL).so
LIB_DEP =
endif

ISA = avx

ifeq ($(ISA),sse)
CFLAGS += -DSIMD -DSSE
endif
ifeq ($(ISA),avx)
CFLAGS += -DSIMD -DAVX
endif
ifeq ($(ISA),mic)
CFLAGS += -DSIMD -DMIC
endif

SRC_DIR = src
LIB_DIR = lib
INC_DIR = include
TEST_DIR = test
