HOME = ../..

include $(HOME)/define.mk

OBJS = unittest.o

PROGS = unittest

default: all

all: $(PROGS)

LIBS += -L$(HOME)/$(LIB_DIR) -l$(AHTL)

unittest : $(OBJS) $(LIB_DEP)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

%.o: %.cpp
	 $(CXX) $(CFLAGS) -c $< -o $@ -I$(HOME)/$(INC_DIR)

clean:
	rm -f $(PROGS) $(OBJS) 
