HOME = ..

include $(HOME)/define.mk

APPS := \
        unit \
#				uniform \
#				zipf \
#				text \


# image, some other things

defaults: all

all:
	@$(foreach APP, $(APPS), \
		$(MAKE) -C $(APP);)

clean:
	@$(foreach APP, $(APPS), \
		$(MAKE) -C $(APP) clean;)
