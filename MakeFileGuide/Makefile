###############S
## Variables   ##
###############
# target: dependenciesS
#	commandsS

# The commands are a series of steps typically to make the target,
# 	need to start with a tab character, not space
# The dependencies are file names, separated by spaces.
# 	These files need to exist before the commands for the target are run.

# These flags will be set only if the user did not set them via terminal command already  (make all DEBUG=0)
DEBUG ?= 1
ENABLE_WARNINGS ?= 1
WARNINGS_AS_ERRORS ?= 0

INCLUDE_DIR = include
SOURCE_DIR = src
BUILD_DIR = build

CXX_STANDARD = c++17

ifeq ($(ENABLE_WARNINGS),1)
CXX_WARNINGS = -Wall -Wextra -Wpedantic
else
CXX_WARNINGS = 
endif
ifeq ($(WARNINGS_AS_ERRORS),1)
CXX_WARNINGS += -Werror
endif

CXX = g++
CXXFLAGS = $(CXX_WARNINGS) -std=$(CXX_STANDARD)
CPPFLAGS = -I $(INCLUDE_DIR)					# Where the compiler should look for includes
LDFLAGS = 

ifeq ($(DEBUG), 1)
CXXFLAGS += -g -O0
EXECUTABLE_NAME = mainDebug
else
CXXFLAGS += -O3
EXECUTABLE_NAME = mainRelease
endif


CXX_COMPILER_CALL = $(CXX) $(CXXFLAGS) $(CPPFLAGS)

CXX_SOURCES = $(wildcard $(SOURCE_DIR)/*.cpp)									# This will get us:	 								main.cpp myLib.cpp
CXX_OBJECTS = $(patsubst $(SOURCE_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CXX_SOURCES))	# Substitute .cpp with .o in the CXX_SOURCES list: 	main.o myLib.o

# Phony targets: they do not correspond to existing file name

all: create build


# ?= only sets variables if they have not yet been set
DEBUG ?= 666
# We can also check if variable is defined with ifdef

# -p so it will not throw an error if folder already exists
#  @ so we will not see the console output
# If we need to use more terminal commands that are dependend on each other we have to chain them with &&
create:
	@mkdir -p build    
	cd build && mkdir -p test

build: create $(CXX_OBJECTS)
	$(CXX_COMPILER_CALL) $(CXX_OBJECTS) $(LDFLAGS) -o $(BUILD_DIR)/$(EXECUTABLE_NAME)

execute:
	./$(BUILD_DIR)/$(EXECUTABLE_NAME)

# clean is often used as a target that removes the output of other targets
clean:
	rm -f $(BUILD_DIR)/*.o
	rm -f $(BUILD_DIR)/$(EXECUTABLE_NAME)

# $@: the file name of the target
# $<: the name of the first dependency
# $^: the names of all prerequisites

# PATTERNS -------------------
# For every *.cpp file in our directory we will generate corresponding object file:
$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	$(CXX_COMPILER_CALL) -c $< -o $@

# Almost like template function that will do this:
#main.o: main.cpp
#	$(CXX_COMPILER_CALL) -c main.cpp -o main.o


## PHONY targets
.PHONY: all create build execute clean