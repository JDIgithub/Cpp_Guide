###############S
## Variables   ##
###############
# target: dependencies
#	commandsS

# The commands are a series of steps typically to make the target,
# 	need to start with a tab character, not space
# The dependencies are file names, separated by spaces.
# 	These files need to exist before the commands for the target are run.


DEBUG = 1
EXECUTABLE_NAME = main

CXX_STANDARD = c++17
CXX_WARNINGS = -Wall -Wextra -Wpedantic
CXX = g++
CXXFLAGS = $(CXX_WARNINGS) -std=$(CXX_STANDARD)
LDFLAGS = -lmath

ifeq ($(DEBUG), 1)
CXXFLAGS += -g -O0
else
CXXFLAGS += -O3
endif

COMPILER_CALL = $(CXX) $(CXXFLAGS)

build: myLib.o main.o
	$(COMPILER_CALL) main.o myLib.o -o $(EXECUTABLE_NAME)

main.o:
	$(COMPILER_CALL) main.cpp -c main.o
myLib.o:
	$(COMPILER_CALL) myLib.cpp -c myLib.o

execute:
	./main

# clean is often used as a target that removes the output of other targets
clean:
	rm -f *.o
	rm -f main
