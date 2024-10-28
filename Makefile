CC = gcc
CFLAGS = -Wall -Wextra -Werror -DDEBUG -g
# CFLAGS = -Wall -Wextra -Werror -g
LDFLAGS = -lm
TARGET = HopfieldNetworks
SRCS = HopfieldNetworks.c utils.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean