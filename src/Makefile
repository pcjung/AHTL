HOME = ..

include $(HOME)/define.mk

SRCS := \
				binary.cpp\
				fixed.cpp\
				linear.cpp\
				sorting.cpp\
				partition.cpp\
				
				
OBJS := ${SRCS:.cpp=.o}

defaults: all

all: $(HOME)/$(LIB_DIR)/$(TARGET)

$(HOME)/$(LIB_DIR)/$(TARGET): $(TARGET)
	cp $< $(HOME)/$(LIB_DIR)/$<

$(LIB_AHTL).a: $(OBJS)
	$(AR) cr $@ $(OBJS)
	$(RANLIB) $@

$(LIB_AHTL).so: $(OBJS)
	$(CXX) --shared -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp  
	$(CXX) $(CFLAGS) -c $< -o $@ -I$(HOME)/$(INC_DIR)/

clean:
	rm -f $(HOME)/$(LIB_DIR)/$(TARGET) $(TARGET) $(OBJS)
