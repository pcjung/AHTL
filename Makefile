include define.mk

default: all

all: $(TARGET) tests

$(TARGET):
	@$(MAKE) -C $(SRC_DIR)

tests:
	@$(MAKE) -C $(TEST_DIR)

clean:
	@$(MAKE) -C $(SRC_DIR) clean
	@$(MAKE) -C $(TEST_DIR) clean
