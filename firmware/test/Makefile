CFLAGS ?= -std=c11 -Wall -Wextra -Werror -O1 -g -fsanitize=undefined
test_control: test_control.c ../main/control.c ../main/control.h
	$(CC) $(CFLAGS) -o $@ test_control.c ../main/control.c
	./test_control
.PHONY: test_control
